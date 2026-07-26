#if UNITY_EDITOR
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using Aexis.Samples.Json.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

public static class AIImageAndroidSmokeTest
{
    private const string AdbPathEnvironmentVariable = "AIIMAGE_ANDROID_ADB_PATH";
    private const string AdbSerialEnvironmentVariable = "AIIMAGE_ANDROID_ADB_SERIAL";
    private const string ReportPathEnvironmentVariable = "AIIMAGE_ANDROID_SMOKE_REPORT";
    private const string QwenBuildOutputEnvironmentVariable = "AIIMAGE_QWEN35_BUILD_OUTPUT";
    private const string SmokeReportFileName = "aiimage_android_smoke.json";
    private const string RunnerReportFileName = "aiimage_android_runner_smoke.json";
    private const string RunnerInputDirectoryName = "aiimage-smoke";
    private const string SmokeMarker = "[AIIMAGE_ANDROID_SMOKE] ready";

    public static void BuildInstallRunAndroidBatch()
    {
        RunAndroidSmokeBatch(buildApk: true, requireVulkan: true, runDefaultRunners: false);
    }

    public static void InstallRunExistingAndroidApkBatch()
    {
        RunAndroidSmokeBatch(buildApk: false, requireVulkan: false, runDefaultRunners: false);
    }

    public static void InstallRunExistingAndroidVulkanApkBatch()
    {
        RunAndroidSmokeBatch(buildApk: false, requireVulkan: true, runDefaultRunners: false);
    }

    public static void BuildInstallRunAndroidDefaultRunnersVulkanBatch()
    {
        RunAndroidSmokeBatch(buildApk: true, requireVulkan: true, runDefaultRunners: true);
    }

    public static void BuildInstallRunAndroidDefaultRunnersVulkanDeviceQualificationBatch()
    {
        RunAndroidSmokeBatch(
            buildApk: true,
            requireVulkan: true,
            runDefaultRunners: true,
            runQwenDeviceQualification: true);
    }

    public static void InstallRunExistingAndroidDefaultRunnersVulkanBatch()
    {
        RunAndroidSmokeBatch(buildApk: false, requireVulkan: true, runDefaultRunners: true);
    }

    private static void RunAndroidSmokeBatch(
        bool buildApk,
        bool requireVulkan,
        bool runDefaultRunners,
        bool runQwenDeviceQualification = false)
    {
        var stopwatch = Stopwatch.StartNew();
        var reportPath = ResolveReportPath();
        var report = new JObject
        {
            ["schema"] = "aiimage.android-smoke/v1",
            ["started_utc"] = DateTime.UtcNow.ToString("O"),
            ["unity_version"] = Application.unityVersion,
            ["valid"] = false
        };
        var success = false;
        try
        {
            var adbPath = ResolveAdbPath();
            var serial = ResolveSerial(adbPath);
            var apkPath = ResolveApkPath();
            report["adb_path"] = adbPath;
            report["device_serial"] = serial;
            report["apk_path"] = apkPath;
            report["require_vulkan"] = requireVulkan;
            report["run_default_runners"] = runDefaultRunners;
            report["run_qwen_device_qualification"] = runQwenDeviceQualification;

            if (buildApk)
                BuildNativeAndroidSmokeApk(apkPath, report);
            else
                report["reused_existing_apk"] = true;
            if (!File.Exists(apkPath))
                throw new FileNotFoundException("Android smoke APK was not produced.", apkPath);
            report["apk_bytes"] = new FileInfo(apkPath).Length;

            var connect = RequiresAdbConnect(serial)
                ? RunAdb(adbPath, "connect " + Quote(serial), TimeSpan.FromSeconds(20))
                : new CommandResult(0, "not required for the selected Android device", string.Empty);
            report["adb_connect"] = connect.ToJson();
            EnsureSuccess(connect, "ADB connect failed");
            var deviceState = RunAdb(adbPath, "-s " + Quote(serial) + " get-state", TimeSpan.FromSeconds(15));
            report["device_state"] = deviceState.ToJson();
            if (!string.Equals(deviceState.StandardOutput.Trim(), "device", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("ADB device is not ready: " + deviceState.StandardOutput + deviceState.StandardError);

            var packageInfo = ReadPackageInfo(apkPath);
            report["package"] = packageInfo.PackageName;
            report["activity"] = packageInfo.ActivityName;

            var clearLog = RunAdb(adbPath, "-s " + Quote(serial) + " logcat -c", TimeSpan.FromSeconds(15));
            report["clear_logcat"] = clearLog.ToJson();
            EnsureSuccess(clearLog, "Failed to clear Android logcat");

            var install = RunAdb(adbPath, "-s " + Quote(serial) + " install -r -g " + Quote(apkPath), TimeSpan.FromMinutes(3));
            report["install"] = install.ToJson();
            EnsureSuccess(install, "APK installation failed");
            if (!install.StandardOutput.Contains("Success", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("APK installation did not report Success: " + install.StandardOutput);

            if (runDefaultRunners)
            {
                var clearApplicationData = RunAdb(
                    adbPath,
                    "-s " + Quote(serial) + " shell pm clear " + Quote(packageInfo.PackageName),
                    TimeSpan.FromSeconds(30));
                report["clear_application_data"] = clearApplicationData.ToJson();
                EnsureSuccess(clearApplicationData, "Could not clear Android application data before runner smoke");
                PushRunnerSmokeInputs(adbPath, serial, packageInfo, report);
            }

            var start = RunAdb(
                adbPath,
                "-s " + Quote(serial) + " shell am start -W -S -n " + Quote(packageInfo.ComponentName)
                + " --ez aiimage_smoke true"
                + (runDefaultRunners ? " --ez aiimage_runner_smoke true" : string.Empty)
                + (runQwenDeviceQualification ? " --ez aiimage_runner_qwen_device_qualification true" : string.Empty),
                TimeSpan.FromSeconds(45));
            report["start"] = start.ToJson();
            EnsureSuccess(start, "APK launch failed");

            var logcatPath = Path.Combine(
                Path.GetDirectoryName(reportPath),
                Path.GetFileNameWithoutExtension(reportPath) + ".logcat.txt");
            var smokeReady = false;
            var smokeFileReady = false;
            var runnerReportReady = false;
            var processObserved = false;
            var latestLogcat = default(CommandResult);
            var smokeFile = default(CommandResult);
            var runnerReportFile = default(CommandResult);
            var latestLogcatCaptured = false;
            var attemptLimit = runDefaultRunners ? 600 : 30;
            for (var attempt = 0; attempt < attemptLimit; attempt++)
            {
                var pid = RunAdb(adbPath, "-s " + Quote(serial) + " shell pidof " + Quote(packageInfo.PackageName), TimeSpan.FromSeconds(10));
                processObserved |= pid.ExitCode == 0 && !string.IsNullOrWhiteSpace(pid.StandardOutput);
                if (!runDefaultRunners || attempt % 10 == 0 || attempt == attemptLimit - 1)
                {
                    latestLogcat = RunAdb(adbPath, "-s " + Quote(serial) + " logcat -d -v threadtime", TimeSpan.FromSeconds(30));
                    latestLogcatCaptured = true;
                    File.WriteAllText(logcatPath, latestLogcat.StandardOutput + latestLogcat.StandardError);
                    smokeReady = latestLogcat.StandardOutput.Contains(SmokeMarker, StringComparison.Ordinal);
                }
                smokeFile = RunAdb(
                    adbPath,
                    "-s " + Quote(serial) + " shell cat /sdcard/Android/data/" + packageInfo.PackageName + "/files/" + SmokeReportFileName,
                    TimeSpan.FromSeconds(20));
                smokeFileReady = smokeFile.ExitCode == 0
                    && smokeFile.StandardOutput.Contains("\"status\": \"ready\"", StringComparison.Ordinal);
                if (runDefaultRunners)
                {
                    runnerReportFile = RunAdb(
                        adbPath,
                        "-s " + Quote(serial) + " shell cat /sdcard/Android/data/" + packageInfo.PackageName + "/files/" + RunnerReportFileName,
                        TimeSpan.FromSeconds(20));
                    runnerReportReady = runnerReportFile.ExitCode == 0
                        && IsRunnerReportComplete(runnerReportFile.StandardOutput);
                    if (runnerReportReady)
                        break;
                }
                else if (smokeReady || smokeFileReady)
                    break;
                Thread.Sleep(TimeSpan.FromSeconds(2));
            }

            var fatal = latestLogcatCaptured && ContainsFatalAndroidRuntime(latestLogcat.StandardOutput);
            report["process_observed"] = processObserved;
            report["smoke_marker_observed"] = smokeReady;
            report["smoke_file_ready"] = smokeFileReady;
            report["smoke_file"] = smokeFile.ToJson();
            report["runner_report_ready"] = runnerReportReady;
            report["runner_report_file"] = runnerReportFile.ToJson();
            report["logcat_path"] = logcatPath;
            report["fatal_android_runtime"] = fatal;
            if (!smokeReady && !smokeFileReady)
                throw new TimeoutException("Android Player did not write the smoke report or emit the smoke-ready marker. See " + logcatPath);
            if (fatal)
                throw new InvalidOperationException("Android logcat contains a fatal runtime failure. See " + logcatPath);
            if (!processObserved)
                throw new InvalidOperationException("Android Player process was not observed running.");
            if (!smokeFileReady)
                throw new InvalidOperationException("Android smoke report file is missing or invalid: " + smokeFile.StandardOutput + smokeFile.StandardError);

            var playerSmoke = JObject.Parse(smokeFile.StandardOutput);
            var playerGraphicsApi = (string)playerSmoke["graphicsDeviceType"];
            report["player_graphics_api"] = playerGraphicsApi;
            report["vulkan_detection_log_observed"] = latestLogcat.StandardOutput.Contains("Vulkan detection: 1", StringComparison.Ordinal);
            if (requireVulkan && !string.Equals(playerGraphicsApi, GraphicsDeviceType.Vulkan.ToString(), StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "Android Player did not acquire Vulkan. Runtime graphicsDeviceType=" + playerGraphicsApi
                    + ". See " + logcatPath);

            if (runDefaultRunners)
            {
                if (!runnerReportReady)
                    throw new TimeoutException("Android default runner smoke did not finish. See " + logcatPath);
                var runnerReport = JObject.Parse(runnerReportFile.StandardOutput);
                report["runner_report"] = runnerReport;
                var runnerValid = (bool?)runnerReport["valid"] ?? false;
                if (!runnerValid)
                    throw new InvalidOperationException("Android default runner smoke failed: " + runnerReportFile.StandardOutput);
            }

            var memoryInfo = RunAdb(
                adbPath,
                "-s " + Quote(serial) + " shell dumpsys meminfo " + Quote(packageInfo.PackageName),
                TimeSpan.FromSeconds(30));
            report["memory_info"] = memoryInfo.ToJson();
            EnsureSuccess(memoryInfo, "Could not collect Android process memory information");
            var pssMatch = Regex.Match(memoryInfo.StandardOutput, @"(?:TOTAL PSS:|TOTAL:)\s*(?<kb>[0-9,]+)", RegexOptions.IgnoreCase);
            if (pssMatch.Success
                && long.TryParse(pssMatch.Groups["kb"].Value.Replace(",", string.Empty), out var pssKb))
            {
                report["process_pss_kb"] = pssKb;
            }

            success = true;
        }
        catch (Exception exception)
        {
            report["error"] = exception.ToString();
            UnityEngine.Debug.LogException(exception);
        }
        finally
        {
            stopwatch.Stop();
            report["valid"] = success;
            report["ended_utc"] = DateTime.UtcNow.ToString("O");
            report["elapsed_ms"] = stopwatch.ElapsedMilliseconds;
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath));
            File.WriteAllText(reportPath, report.ToString());
            UnityEngine.Debug.Log("Android smoke report: " + reportPath + " valid=" + success);
            EditorApplication.Exit(success ? 0 : 1);
        }
    }

    private static string ResolveAdbPath()
    {
        var path = Environment.GetEnvironmentVariable(AdbPathEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(path))
            throw new InvalidOperationException(AdbPathEnvironmentVariable + " must contain an Android adb.exe path.");
        path = Path.GetFullPath(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Configured Android adb.exe is missing.", path);
        return path;
    }

    private static string ResolveSerial(string adbPath)
    {
        var serial = Environment.GetEnvironmentVariable(AdbSerialEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(serial))
            return serial.Trim();

        var devices = RunAdb(adbPath, "devices", TimeSpan.FromSeconds(15));
        EnsureSuccess(devices, "Could not enumerate Android devices");
        var matches = Regex.Matches(devices.StandardOutput, "^(?<serial>[^\\s]+)\\s+device$", RegexOptions.Multiline)
            .Cast<Match>()
            .Select(match => match.Groups["serial"].Value)
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        if (matches.Length == 1)
            return matches[0];
        if (matches.Length == 0)
            throw new InvalidOperationException("No Android device is connected. Set " + AdbSerialEnvironmentVariable + " for a TCP device before running the smoke test.");
        throw new InvalidOperationException("Multiple Android devices are connected. Set " + AdbSerialEnvironmentVariable + " to select one.");
    }

    private static string ResolveApkPath()
    {
        var path = Environment.GetEnvironmentVariable(QwenBuildOutputEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(path))
            path = Path.Combine(ProjectRoot, "Tools", "Qwen35NcnnBaseline", "build", "mobile", "android", "Qwen35MobileValidation.apk");
        return Path.GetFullPath(path);
    }

    private static string ResolveReportPath()
    {
        var path = Environment.GetEnvironmentVariable(ReportPathEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(path))
            return Path.GetFullPath(path);
        return Path.Combine(ProjectRoot, "Logs", "AIImage_Android_Smoke.json");
    }

    private static void BuildNativeAndroidSmokeApk(string apkPath, JObject report)
    {
        const BuildTarget target = BuildTarget.Android;
        var originalBuildAppBundle = EditorUserBuildSettings.buildAppBundle;
        try
        {
            EditorUserBuildSettings.buildAppBundle = false;

            report["test_variant"] = new JObject
            {
                ["purpose"] = "android-arm64-vulkan-install-startup-smoke",
                ["graphics_api"] = GraphicsDeviceType.Vulkan.ToString(),
                ["android_architecture"] = AndroidArchitecture.ARM64.ToString(),
                ["application_entry"] = "GameActivity"
            };

            var buildReport = AIImageReducedModelBuild.BuildMain2(
                target,
                apkPath,
                BuildOptions.Development | BuildOptions.CleanBuildCache);
            report["build"] = new JObject
            {
                ["result"] = buildReport.summary.result.ToString(),
                ["total_errors"] = buildReport.summary.totalErrors,
                ["total_warnings"] = buildReport.summary.totalWarnings,
                ["total_bytes"] = buildReport.summary.totalSize,
                ["output_path"] = buildReport.summary.outputPath
            };
            if (buildReport.summary.result != BuildResult.Succeeded || buildReport.summary.totalErrors != 0)
                throw new InvalidOperationException(
                    "Android native smoke APK build failed: result=" + buildReport.summary.result
                    + " errors=" + buildReport.summary.totalErrors);
        }
        finally
        {
            EditorUserBuildSettings.buildAppBundle = originalBuildAppBundle;
            AssetDatabase.SaveAssets();
        }
    }

    private static void PushRunnerSmokeInputs(string adbPath, string serial, PackageInfo packageInfo, JObject report)
    {
        var faceInput = Path.Combine(ProjectRoot, "ref", "02.png");
        var sceneInput = Path.Combine(ProjectRoot, "ref", "03.jpg");
        if (!File.Exists(faceInput) || !File.Exists(sceneInput))
            throw new FileNotFoundException("Android default runner smoke input is missing from the ref directory.");

        var deviceDirectory = "/sdcard/Android/data/" + packageInfo.PackageName + "/files/" + RunnerInputDirectoryName;
        var createDirectory = RunAdb(adbPath, "-s " + Quote(serial) + " shell mkdir -p " + Quote(deviceDirectory), TimeSpan.FromSeconds(20));
        report["runner_input_directory"] = createDirectory.ToJson();
        EnsureSuccess(createDirectory, "Could not create the Android runner smoke input directory");

        var pushFace = RunAdb(adbPath, "-s " + Quote(serial) + " push " + Quote(faceInput) + " " + Quote(deviceDirectory + "/face.png"), TimeSpan.FromMinutes(2));
        var pushScene = RunAdb(adbPath, "-s " + Quote(serial) + " push " + Quote(sceneInput) + " " + Quote(deviceDirectory + "/scene.jpg"), TimeSpan.FromMinutes(2));
        report["runner_input_face"] = pushFace.ToJson();
        report["runner_input_scene"] = pushScene.ToJson();
        EnsureSuccess(pushFace, "Could not push the CodeFormer/GFPGAN/YOLO/CLIP runner smoke input");
        EnsureSuccess(pushScene, "Could not push the Matting/QWEN runner smoke input");
    }

    private static bool IsRunnerReportComplete(string json)
    {
        try
        {
            var status = (string)JObject.Parse(json)["status"];
            return string.Equals(status, "passed", StringComparison.Ordinal)
                || string.Equals(status, "completed_with_failures", StringComparison.Ordinal)
                || string.Equals(status, "failed", StringComparison.Ordinal);
        }
        catch
        {
            return false;
        }
    }

    private static PackageInfo ReadPackageInfo(string apkPath)
    {
        var aaptPath = ResolveAaptPath();
        var result = RunProcess(aaptPath, "dump badging " + Quote(apkPath), TimeSpan.FromSeconds(30));
        EnsureSuccess(result, "Could not inspect Android APK manifest");
        var package = Regex.Match(result.StandardOutput, "^package: name='(?<value>[^']+)'", RegexOptions.Multiline).Groups["value"].Value;
        var activity = Regex.Match(result.StandardOutput, "^launchable-activity: name='(?<value>[^']+)'", RegexOptions.Multiline).Groups["value"].Value;
        if (string.IsNullOrWhiteSpace(package) || string.IsNullOrWhiteSpace(activity))
            throw new InvalidDataException("Android APK manifest does not contain a launchable activity.");
        return new PackageInfo(package, activity);
    }

    private static string ResolveAaptPath()
    {
        var androidPlayerRoot = Path.Combine(EditorApplication.applicationContentsPath, "PlaybackEngines", "AndroidPlayer", "SDK", "build-tools");
        if (!Directory.Exists(androidPlayerRoot))
            throw new DirectoryNotFoundException("Unity Android build-tools directory is missing: " + androidPlayerRoot);
        var candidates = Directory.GetFiles(androidPlayerRoot, "aapt.exe", SearchOption.AllDirectories)
            .OrderByDescending(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (candidates.Length == 0)
            throw new FileNotFoundException("Unity Android SDK aapt.exe was not found.");
        return candidates[0];
    }

    private static CommandResult RunAdb(string adbPath, string arguments, TimeSpan timeout)
    {
        return RunProcess(adbPath, arguments, timeout);
    }

    private static bool RequiresAdbConnect(string serial)
    {
        return Regex.IsMatch(serial ?? string.Empty, "^[^\\s:]+:\\d+$", RegexOptions.CultureInvariant);
    }

    private static CommandResult RunProcess(string fileName, string arguments, TimeSpan timeout)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        using (var process = Process.Start(startInfo))
        {
            if (process == null)
                throw new InvalidOperationException("Failed to start process: " + fileName);
            var standardOutput = process.StandardOutput.ReadToEnd();
            var standardError = process.StandardError.ReadToEnd();
            if (!process.WaitForExit((int)timeout.TotalMilliseconds))
            {
                try { process.Kill(); } catch { }
                throw new TimeoutException("Process timed out: " + fileName + " " + arguments);
            }
            return new CommandResult(process.ExitCode, standardOutput, standardError);
        }
    }

    private static bool ContainsFatalAndroidRuntime(string logcat)
    {
        return logcat.Contains("FATAL EXCEPTION", StringComparison.Ordinal)
            || logcat.Contains("AndroidRuntime: Process:", StringComparison.Ordinal)
            || logcat.Contains("Fatal signal", StringComparison.Ordinal);
    }

    private static void EnsureSuccess(CommandResult result, string message)
    {
        if (result.ExitCode != 0)
            throw new InvalidOperationException(message + ": " + result.StandardOutput + result.StandardError);
    }

    private static string Quote(string value)
    {
        return "\"" + (value ?? string.Empty).Replace("\"", "\\\"") + "\"";
    }

    private static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

    private readonly struct PackageInfo
    {
        public readonly string PackageName;
        public readonly string ActivityName;
        public string ComponentName => PackageName + "/" + ActivityName;

        public PackageInfo(string packageName, string activityName)
        {
            PackageName = packageName;
            ActivityName = activityName;
        }
    }

    private readonly struct CommandResult
    {
        public readonly int ExitCode;
        public readonly string StandardOutput;
        public readonly string StandardError;

        public CommandResult(int exitCode, string standardOutput, string standardError)
        {
            ExitCode = exitCode;
            StandardOutput = standardOutput ?? string.Empty;
            StandardError = standardError ?? string.Empty;
        }

        public JObject ToJson()
        {
            return new JObject
            {
                ["exit_code"] = ExitCode,
                ["stdout"] = StandardOutput,
                ["stderr"] = StandardError
            };
        }
    }
}
#endif
