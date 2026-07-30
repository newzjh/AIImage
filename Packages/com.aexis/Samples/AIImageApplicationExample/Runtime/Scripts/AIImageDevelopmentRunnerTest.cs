#if UNITY_EDITOR || DEVELOPMENT_BUILD
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Aexis.Samples.Async;
using AIImage.Qwen35;
using UnityEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

[Serializable]
public sealed class AIImageDevelopmentRunnerTestReport
{
    public string schema = "aexis-development-runner-test/v1";
    public string status;
    public string startedUtc;
    public string completedUtc;
    public int runnerTimeoutSeconds;
    public string reportPath;
    public string unityVersion;
    public string platform;
    public string operatingSystem;
    public string deviceModel;
    public string graphicsDeviceType;
    public string graphicsDeviceName;
    public int systemMemoryMb;
    public string persistentDataPath;
    public string sourceImagePath;
    public string sourceImageName;
    public int sourceWidth;
    public int sourceHeight;
    public List<AIImageDevelopmentRunnerTestEntry> runners = new List<AIImageDevelopmentRunnerTestEntry>();
}

[Serializable]
public sealed class AIImageDevelopmentRunnerTestEntry
{
    public string id;
    public string displayName;
    public string modelGroup;
    public string status;
    public string detail;
    public long elapsedMs;
    public int outputWidth;
    public int outputHeight;
    public int personCount;
    public float maskCoverage01;
}

/// <summary>
/// Development-build runner smoke coverage for the image currently selected in MainView2.
/// It deliberately never downloads a model; missing local payloads are reported as skipped.
/// </summary>
public static class AIImageDevelopmentRunnerTest
{
    public const int RunnerTimeoutSeconds = 10 * 60;

    private const string QwenPrompt =
        "Describe this image in one concise sentence. State only what is visible and do not explain your reasoning.";

    public static async UniTask<AIImageDevelopmentRunnerTestReport> RunAsync(
        AIImagePageHost host,
        Texture2D sourceImage,
        string sourceImagePath,
        CancellationToken cancellationToken,
        Action<int, int, string> onProgress = null)
    {
        if (host == null)
            throw new ArgumentNullException(nameof(host));
        if (sourceImage == null)
            throw new ArgumentNullException(nameof(sourceImage));

        var controller = new Controller(host, sourceImage, sourceImagePath, cancellationToken, onProgress);
        return await controller.RunAsync();
    }

    public static void RevealReport(string reportPath)
    {
        if (string.IsNullOrWhiteSpace(reportPath) || !File.Exists(reportPath))
            return;

        var absoluteReportPath = Path.GetFullPath(reportPath);

        try
        {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            var reportDirectory = Path.GetDirectoryName(absoluteReportPath);
            if (string.IsNullOrWhiteSpace(reportDirectory) || !Directory.Exists(reportDirectory))
                return;
            Process.Start(new ProcessStartInfo
            {
                FileName = reportDirectory,
                UseShellExecute = true
            });
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
            Process.Start(new ProcessStartInfo
            {
                FileName = "/usr/bin/open",
                Arguments = "-R \"" + absoluteReportPath.Replace("\"", "\\\"") + "\"",
                UseShellExecute = false
            });
#elif UNITY_STANDALONE_LINUX
            Process.Start(new ProcessStartInfo
            {
                FileName = "xdg-open",
                Arguments = "\"" + Path.GetDirectoryName(absoluteReportPath)?.Replace("\"", "\\\"") + "\"",
                UseShellExecute = false
            });
#elif UNITY_ANDROID && !UNITY_EDITOR
            RevealAndroidReport(absoluteReportPath);
#elif UNITY_IOS && !UNITY_EDITOR
            RevealIosReport(absoluteReportPath);
#else
            UnityEngine.Debug.Log("Aexis development runner report: " + absoluteReportPath);
#endif
        }
        catch (Exception exception)
        {
            UnityEngine.Debug.LogWarning("Could not reveal development runner report: " + exception.Message);
        }
    }

#if UNITY_ANDROID && !UNITY_EDITOR
    private static void RevealAndroidReport(string reportPath)
    {
        try
        {
            using (var unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (var activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
            using (var intent = new AndroidJavaObject("android.content.Intent", "android.intent.action.VIEW"))
            using (var uriClass = new AndroidJavaClass("android.net.Uri"))
            {
                var uri = uriClass.CallStatic<AndroidJavaObject>("parse", "file://" + reportPath.Replace("\\", "/"));
                intent.Call<AndroidJavaObject>("setDataAndType", uri, "application/json");
                intent.Call<AndroidJavaObject>("addFlags", 1); // FLAG_GRANT_READ_URI_PERMISSION
                activity.Call("startActivity", intent);
            }
        }
        catch (Exception exception)
        {
            UnityEngine.Debug.LogWarning(
                "Aexis development runner report is available at " + reportPath
                + ". Android file-manager launch was rejected: " + exception.Message);
        }
    }
#endif

#if UNITY_IOS && !UNITY_EDITOR
    private static void RevealIosReport(string reportPath)
    {
        try
        {
            AIImageReportReveal(reportPath);
        }
        catch (Exception exception)
        {
            UnityEngine.Debug.LogWarning(
                "Aexis development runner report is available at " + reportPath
                + ". iOS report preview could not be opened: " + exception.Message);
        }
    }

    [System.Runtime.InteropServices.DllImport("__Internal")]
    private static extern void AIImageReportReveal(string path);
#endif

    private sealed class Controller
    {
        private const int TotalRunnerCount = 9;

        private readonly AIImagePageHost _host;
        private readonly Texture2D _sourceImage;
        private readonly CancellationToken _cancellationToken;
        private readonly Action<int, int, string> _onProgress;
        private readonly AIImageDevelopmentRunnerTestReport _report;
        private YoloSegResult _yoloResult;
        private bool _hasYoloMask;
        private int _completedRunnerCount;

        public Controller(
            AIImagePageHost host,
            Texture2D sourceImage,
            string sourceImagePath,
            CancellationToken cancellationToken,
            Action<int, int, string> onProgress)
        {
            _host = host;
            _sourceImage = sourceImage;
            _cancellationToken = cancellationToken;
            _onProgress = onProgress;
            var timestamp = DateTime.UtcNow;
            _report = new AIImageDevelopmentRunnerTestReport
            {
                status = "running",
                startedUtc = timestamp.ToString("O"),
                runnerTimeoutSeconds = RunnerTimeoutSeconds,
                reportPath = Path.Combine(
                    Application.persistentDataPath,
                    "AexisDevelopmentRunnerTest_" + timestamp.ToString("yyyyMMdd_HHmmss") + ".json"),
                unityVersion = Application.unityVersion,
                platform = Application.platform.ToString(),
                operatingSystem = SystemInfo.operatingSystem,
                deviceModel = SystemInfo.deviceModel,
                graphicsDeviceType = SystemInfo.graphicsDeviceType.ToString(),
                graphicsDeviceName = SystemInfo.graphicsDeviceName,
                systemMemoryMb = SystemInfo.systemMemorySize,
                persistentDataPath = Application.persistentDataPath,
                sourceImagePath = sourceImagePath ?? string.Empty,
                sourceImageName = sourceImage.name ?? string.Empty,
                sourceWidth = sourceImage.width,
                sourceHeight = sourceImage.height
            };
        }

        public async UniTask<AIImageDevelopmentRunnerTestReport> RunAsync()
        {
            WriteReport();
            try
            {
                await RunClipAsync();
                await RunCodeFormerAsync();
                await RunGfpganAsync();
                await RunRealEsrganAsync();
                await RunMattingAsync();
                await RunQwenAsync();
                await RunYoloAsync();
                await RunDeepFillAsync();
                await RunSdInpaintingAsync();
                _report.status = HasFailures() ? "completed_with_failures" : "completed";
            }
            catch (OperationCanceledException) when (_cancellationToken.IsCancellationRequested)
            {
                _report.status = "cancelled";
            }
            catch (Exception exception)
            {
                _report.status = "failed";
                AddUnexpectedFailure(exception);
                UnityEngine.Debug.LogException(exception);
            }
            finally
            {
                DestroyYoloResult();
                _report.completedUtc = DateTime.UtcNow.ToString("O");
                WriteReport();
            }

            return _report;
        }

        private async UniTask RunClipAsync()
        {
            var runner = _host.ClipRunner;
            if (runner == null)
            {
                SkipUnavailableRunner("clip_mobileclip_s0", "CLIP MobileCLIP S0", AIImageModelGroupId.ClipMobileClipS0, "CLIP runner component is unavailable.");
                return;
            }

            var originalForceTexture = runner.forceFullRenderTexturePath;
            var originalDisallowAccess = runner.disallowBufferAccess;
            var originalDisallowOutputs = runner.disallowBufferOutputs;
            var originalDisallowMaterialization = runner.disallowBufferToTextureMaterialization;
            await RunEntryAsync(
                "clip_mobileclip_s0",
                "CLIP MobileCLIP S0",
                AIImageModelGroupId.ClipMobileClipS0,
                async token =>
                {
                    runner.forceFullRenderTexturePath = true;
                    runner.disallowBufferAccess = true;
                    runner.disallowBufferOutputs = true;
                    runner.disallowBufferToTextureMaterialization = true;
                    var result = await runner.ProcessAsync(_sourceImage, token);
                    var passed = string.IsNullOrWhiteSpace(result.error)
                        && result.scores != null && result.scores.Length > 0
                        && !string.IsNullOrWhiteSpace(result.bestLabel);
                    return RunnerOutcome.Create(
                        passed,
                        string.IsNullOrWhiteSpace(result.error)
                            ? "bestLabel=" + (result.bestLabel ?? string.Empty)
                            + " | probability=" + result.bestProbability.ToString("0.000000")
                            : result.error,
                        result.elapsedMs);
                },
                () =>
                {
                    runner.forceFullRenderTexturePath = originalForceTexture;
                    runner.disallowBufferAccess = originalDisallowAccess;
                    runner.disallowBufferOutputs = originalDisallowOutputs;
                    runner.disallowBufferToTextureMaterialization = originalDisallowMaterialization;
                });
        }

        private async UniTask RunCodeFormerAsync()
        {
            var runner = _host.CodeFormerReproRunner;
            if (runner == null)
            {
                SkipUnavailableRunner("codeformer", "CodeFormer", AIImageModelGroupId.CodeFormerDefault, "CodeFormer runner component is unavailable.");
                return;
            }

            var originalDebugDump = runner.enableDebugDump;
            var originalFaceDebugDump = runner.enableFaceRegionDebugDump;
            await RunEntryAsync(
                "codeformer",
                "CodeFormer",
                AIImageModelGroupId.CodeFormerDefault,
                async token =>
                {
                    runner.enableDebugDump = false;
                    runner.enableFaceRegionDebugDump = false;
                    var result = await runner.ProcessAsync(_sourceImage, token);
                    return RunnerOutcome.FromTexture(result.texture, result.error, result.elapsedMs);
                },
                () =>
                {
                    runner.enableDebugDump = originalDebugDump;
                    runner.enableFaceRegionDebugDump = originalFaceDebugDump;
                });
        }

        private async UniTask RunGfpganAsync()
        {
            var runner = _host.GfpganReproRunner;
            if (runner == null)
            {
                SkipUnavailableRunner("gfpgan", "GFPGAN", AIImageModelGroupId.GfpganDefault, "GFPGAN runner component is unavailable.");
                return;
            }

            var originalDebugDump = runner.enableFaceRegionDebugDump;
            var originalDisallowAccess = runner.disallowBufferAccess;
            var originalDisallowOutputs = runner.disallowBufferOutputs;
            var originalDisallowMaterialization = runner.disallowBufferToTextureMaterialization;
            await RunEntryAsync(
                "gfpgan",
                "GFPGAN",
                AIImageModelGroupId.GfpganDefault,
                async token =>
                {
                    runner.enableFaceRegionDebugDump = false;
                    runner.disallowBufferAccess = true;
                    runner.disallowBufferOutputs = true;
                    runner.disallowBufferToTextureMaterialization = true;
                    var result = await runner.ProcessAsync(_sourceImage, token);
                    return RunnerOutcome.FromTexture(result.texture, result.error, result.elapsedMs);
                },
                () =>
                {
                    runner.enableFaceRegionDebugDump = originalDebugDump;
                    runner.disallowBufferAccess = originalDisallowAccess;
                    runner.disallowBufferOutputs = originalDisallowOutputs;
                    runner.disallowBufferToTextureMaterialization = originalDisallowMaterialization;
                });
        }

        private async UniTask RunRealEsrganAsync()
        {
            var runner = _host.RealEsrganReproRunner;
            if (runner == null)
            {
                SkipUnavailableRunner("realesrgan", "Real-ESRGAN", ResolveRealEsrganModelGroup(null), "Real-ESRGAN runner component is unavailable.");
                return;
            }

            var group = ResolveRealEsrganModelGroup(runner);
            var originalProfile = runner.enableGpuLayerProfiling;
            var originalDisallowAccess = runner.disallowBufferAccess;
            var originalDisallowOutputs = runner.disallowBufferOutputs;
            var originalDisallowMaterialization = runner.disallowBufferToTextureMaterialization;
            await RunEntryAsync(
                "realesrgan",
                "Real-ESRGAN",
                group,
                async token =>
                {
                    runner.enableGpuLayerProfiling = false;
                    runner.disallowBufferAccess = true;
                    runner.disallowBufferOutputs = true;
                    runner.disallowBufferToTextureMaterialization = true;
                    var result = await runner.ProcessAsync(_sourceImage, token);
                    return RunnerOutcome.FromTexture(result.texture, result.error, result.elapsedMs);
                },
                () =>
                {
                    runner.enableGpuLayerProfiling = originalProfile;
                    runner.disallowBufferAccess = originalDisallowAccess;
                    runner.disallowBufferOutputs = originalDisallowOutputs;
                    runner.disallowBufferToTextureMaterialization = originalDisallowMaterialization;
                });
        }

        private async UniTask RunMattingAsync()
        {
            var runner = _host.MattingReproRunner;
            if (runner == null)
            {
                SkipUnavailableRunner("matting", "Matting", AIImageModelGroupId.Matting, "Matting runner component is unavailable.");
                return;
            }

            var originalDebugDump = runner.enableDebugDump;
            var originalForceBuffer = runner.forceBufferConvolution;
            var originalDisallowAccess = runner.disallowBufferAccess;
            var originalDisallowOutputs = runner.disallowBufferOutputs;
            var originalDisallowMaterialization = runner.disallowBufferToTextureMaterialization;
            await RunEntryAsync(
                "matting",
                "Matting",
                AIImageModelGroupId.Matting,
                async token =>
                {
                    runner.enableDebugDump = false;
                    runner.forceBufferConvolution = false;
                    runner.disallowBufferAccess = true;
                    runner.disallowBufferOutputs = true;
                    runner.disallowBufferToTextureMaterialization = true;
                    var result = await runner.ProcessAsync(_sourceImage, token);
                    var passed = string.IsNullOrWhiteSpace(result.error) && result.texture != null && result.matte != null;
                    var outcome = RunnerOutcome.Create(passed, result.error, result.elapsedMs, result.texture);
                    DestroyTexture(result.texture);
                    DestroyTexture(result.matte);
                    return outcome;
                },
                () =>
                {
                    runner.enableDebugDump = originalDebugDump;
                    runner.forceBufferConvolution = originalForceBuffer;
                    runner.disallowBufferAccess = originalDisallowAccess;
                    runner.disallowBufferOutputs = originalDisallowOutputs;
                    runner.disallowBufferToTextureMaterialization = originalDisallowMaterialization;
                });
        }

        private async UniTask RunQwenAsync()
        {
            var group = ResolveQwen35ModelGroup();
            var existingDirectory = ResolveQwen35ModelDirectory(group);
            await RunEntryAsync(
                "qwen35",
                "Qwen3.5",
                group,
                async token =>
                {
                    var directory = ResolveQwen35ModelDirectory(group);
                    if (!HasQwen35ModelPayload(directory))
                        return RunnerOutcome.Skip("Qwen3.5 model payload is incomplete: " + directory);

                    using (var runner = await Qwen35Runner.CreateAsync(directory, 256, true, token))
                    {
                        var result = await runner.GenerateImageAsync(
                            _sourceImage,
                            QwenPrompt,
                            Qwen35SamplingConfig.Greedy(),
                            token);
                        var text = result?.Text?.Trim() ?? string.Empty;
                        var passed = result != null && result.DecoderStepCount > 0 && text.Length > 0;
                        var detail = "maxNewTokens=" + runner.MaxNewTokens
                            + " | decoderSteps=" + (result?.DecoderStepCount ?? 0)
                            + " | visibleTextChars=" + text.Length
                            + " | text=" + Truncate(text, 240);
                        return RunnerOutcome.Create(passed, detail, 0L);
                    }
                },
                skipModelDeliveryPreflight: HasQwen35ModelPayload(existingDirectory));
        }

        private async UniTask RunYoloAsync()
        {
            var runner = _host.YoloSegRunner;
            if (runner == null)
            {
                SkipUnavailableRunner("yolo_person_segmentation", "YOLO person segmentation", AIImageModelGroupId.YoloV8PersonSegmentation, "YOLO runner component is unavailable.");
                return;
            }

            var group = runner.modelVariant == YoloSegNcnnReproRunner.YoloSegModelVariant.Yolo11nSeg
                ? AIImageModelGroupId.Yolo11PersonSegmentation
                : AIImageModelGroupId.YoloV8PersonSegmentation;
            var originalTargetPerson = runner.targetPersonOnly;
            var originalDebugDump = runner.enableDebugDump;
            var originalDisallowAccess = runner.disallowBufferAccess;
            var originalDisallowOutputs = runner.disallowBufferOutputs;
            var originalDisallowMaterialization = runner.disallowBufferToTextureMaterialization;
            await RunEntryAsync(
                "yolo_person_segmentation",
                "YOLO person segmentation",
                group,
                async token =>
                {
                    runner.targetPersonOnly = true;
                    runner.enableDebugDump = false;
                    runner.disallowBufferAccess = true;
                    runner.disallowBufferOutputs = true;
                    runner.disallowBufferToTextureMaterialization = true;
                    _yoloResult = await runner.ProcessAsync(_sourceImage, token);
                    var passed = string.IsNullOrWhiteSpace(_yoloResult.error);
                    _hasYoloMask = passed && _yoloResult.mask != null && _yoloResult.personCount > 0;
                    var detail = string.IsNullOrWhiteSpace(_yoloResult.error)
                        ? _hasYoloMask
                            ? "Person mask ready for inpainting."
                            : "No person mask was produced; dependent inpainting runners will be skipped."
                        : _yoloResult.error;
                    return RunnerOutcome.Create(
                        passed,
                        detail,
                        _yoloResult.elapsedMs,
                        _yoloResult.texture,
                        _yoloResult.personCount,
                        _yoloResult.maskCoverage01);
                },
                () =>
                {
                    runner.targetPersonOnly = originalTargetPerson;
                    runner.enableDebugDump = originalDebugDump;
                    runner.disallowBufferAccess = originalDisallowAccess;
                    runner.disallowBufferOutputs = originalDisallowOutputs;
                    runner.disallowBufferToTextureMaterialization = originalDisallowMaterialization;
                    runner.ReleaseRuntimeResources();
                });
        }

        private async UniTask RunDeepFillAsync()
        {
            var runner = _host.DeepFillV2Runner;
            if (runner == null)
            {
                SkipUnavailableRunner("yolo_deepfillv2", "YOLO + DeepFillV2", AIImageModelGroupId.DeepFillV2Case1Ncnn, "DeepFillV2 runner component is unavailable.");
                return;
            }
            if (!_hasYoloMask)
            {
                SkipDependencyRunner("yolo_deepfillv2", "YOLO + DeepFillV2", AIImageModelGroupId.DeepFillV2Case1Ncnn, "YOLO did not provide a person mask.");
                return;
            }

            var originalBackend = runner.backend;
            var originalDebugDump = runner.enableDebugDump;
            await RunEntryAsync(
                "yolo_deepfillv2",
                "YOLO + DeepFillV2",
                AIImageModelGroupId.DeepFillV2Case1Ncnn,
                async token =>
                {
                    runner.backend = DeepFillV2Backend.NcnnBin;
                    runner.enableDebugDump = false;
                    var result = await runner.ProcessAsync(_sourceImage, _yoloResult.mask, token);
                    return RunnerOutcome.FromTexture(result.texture, result.error, result.elapsedMs);
                },
                () =>
                {
                    runner.Release();
                    runner.backend = originalBackend;
                    runner.enableDebugDump = originalDebugDump;
                });
        }

        private async UniTask RunSdInpaintingAsync()
        {
            var runner = _host.SDInpaintingRunner;
            if (runner == null)
            {
                SkipUnavailableRunner("yolo_sd_inpainting", "YOLO + SD inpainting", AIImageModelGroupId.StableDiffusion, "SD inpainting runner component is unavailable.");
                return;
            }
            if (!_hasYoloMask)
            {
                SkipDependencyRunner("yolo_sd_inpainting", "YOLO + SD inpainting", AIImageModelGroupId.StableDiffusion, "YOLO did not provide a person mask.");
                return;
            }

            var originalDebugDump = runner.enableDebugDump;
            var originalDisallowTempBuffers = runner.disallowInferenceTempComputeBuffers;
            var originalDisallowAccess = runner.disallowBufferAccess;
            var originalDisallowOutputs = runner.disallowBufferOutputs;
            var originalDisallowMaterialization = runner.disallowBufferToTextureMaterialization;
            await RunEntryAsync(
                "yolo_sd_inpainting",
                "YOLO + SD inpainting",
                AIImageModelGroupId.StableDiffusion,
                async token =>
                {
                    runner.enableDebugDump = false;
                    runner.disallowInferenceTempComputeBuffers = true;
                    runner.disallowBufferAccess = true;
                    runner.disallowBufferOutputs = true;
                    runner.disallowBufferToTextureMaterialization = true;
                    var result = await runner.ProcessAsync(
                        _sourceImage,
                        _yoloResult.mask,
                        SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedPositivePrompt,
                        SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedNegativePrompt,
                        SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedStepCount,
                        12345,
                        SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedStrength,
                        SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedGuidanceScale,
                        token);
                    return RunnerOutcome.FromTexture(result.texture, result.error, result.elapsedMs);
                },
                () =>
                {
                    runner.ReleaseRuntimeResources();
                    runner.enableDebugDump = originalDebugDump;
                    runner.disallowInferenceTempComputeBuffers = originalDisallowTempBuffers;
                    runner.disallowBufferAccess = originalDisallowAccess;
                    runner.disallowBufferOutputs = originalDisallowOutputs;
                    runner.disallowBufferToTextureMaterialization = originalDisallowMaterialization;
                });
        }

        private async UniTask RunEntryAsync(
            string id,
            string displayName,
            AIImageModelGroupId groupId,
            Func<CancellationToken, UniTask<RunnerOutcome>> execute,
            Action restore = null,
            bool skipModelDeliveryPreflight = false)
        {
            var entry = new AIImageDevelopmentRunnerTestEntry
            {
                id = id,
                displayName = displayName,
                modelGroup = groupId.ToString(),
                status = "queued",
                detail = string.Empty
            };
            _report.runners.Add(entry);
            WriteReport();

            using (var timeout = CancellationTokenSource.CreateLinkedTokenSource(_cancellationToken))
            {
                timeout.CancelAfter(TimeSpan.FromSeconds(RunnerTimeoutSeconds));
                var stopwatch = Stopwatch.StartNew();
                try
                {
                    NotifyProgress(displayName + ": preparing model");
                    entry.status = "preparing";
                    WriteReport();
                    var group = AIImageModelDelivery.GetGroup(groupId);
                    if (!skipModelDeliveryPreflight
                        && !await AIImageModelDelivery.IsAvailableAsync(group, timeout.Token))
                    {
                        entry.status = "skipped_missing_model";
                        entry.detail = group.DisplayName + " is not installed or bundled in this Player.";
                        return;
                    }

                    if (!skipModelDeliveryPreflight)
                        await AIImageModelDelivery.MaterializeBundledGroupAsync(group, null, timeout.Token);
                    entry.status = "running";
                    entry.detail = "Executing strict texture-path runner.";
                    NotifyProgress(displayName + ": running");
                    WriteReport();
                    var outcome = await execute(timeout.Token);
                    timeout.Token.ThrowIfCancellationRequested();
                    if (outcome.skip)
                    {
                        entry.status = "skipped_missing_model";
                        entry.detail = outcome.detail;
                    }
                    else
                    {
                        entry.status = outcome.passed ? "passed" : "failed";
                        entry.detail = outcome.detail ?? string.Empty;
                        entry.outputWidth = outcome.outputWidth;
                        entry.outputHeight = outcome.outputHeight;
                        entry.personCount = outcome.personCount;
                        entry.maskCoverage01 = outcome.maskCoverage01;
                    }
                    entry.elapsedMs = outcome.elapsedMs > 0 ? outcome.elapsedMs : stopwatch.ElapsedMilliseconds;
                }
                catch (OperationCanceledException) when (_cancellationToken.IsCancellationRequested)
                {
                    throw;
                }
                catch (OperationCanceledException)
                {
                    entry.status = "timed_out";
                    entry.detail = "Exceeded the per-runner timeout of " + RunnerTimeoutSeconds + " seconds.";
                    entry.elapsedMs = stopwatch.ElapsedMilliseconds;
                }
                catch (Exception exception)
                {
                    entry.status = "failed";
                    entry.detail = Truncate(exception.Message, 480);
                    entry.elapsedMs = stopwatch.ElapsedMilliseconds;
                    UnityEngine.Debug.LogWarning("Development runner test failed for " + displayName + ": " + exception);
                }
                finally
                {
                    stopwatch.Stop();
                    try { restore?.Invoke(); } catch (Exception exception) { UnityEngine.Debug.LogWarning(exception); }
                    CompleteRunner(entry);
                }
            }
        }

        private void SkipUnavailableRunner(string id, string displayName, AIImageModelGroupId groupId, string detail)
        {
            AddSkippedRunner(id, displayName, groupId, "skipped_unavailable_runner", detail);
        }

        private void SkipDependencyRunner(string id, string displayName, AIImageModelGroupId groupId, string detail)
        {
            AddSkippedRunner(id, displayName, groupId, "skipped_dependency", detail);
        }

        private void AddSkippedRunner(string id, string displayName, AIImageModelGroupId groupId, string status, string detail)
        {
            var entry = new AIImageDevelopmentRunnerTestEntry
            {
                id = id,
                displayName = displayName,
                modelGroup = groupId.ToString(),
                status = status,
                detail = detail ?? string.Empty,
                elapsedMs = 0L
            };
            _report.runners.Add(entry);
            CompleteRunner(entry);
        }

        private void CompleteRunner(AIImageDevelopmentRunnerTestEntry entry)
        {
            _completedRunnerCount++;
            NotifyProgress(entry.displayName + ": " + entry.status);
            WriteReport();
        }

        private void AddUnexpectedFailure(Exception exception)
        {
            var entry = new AIImageDevelopmentRunnerTestEntry
            {
                id = "orchestrator",
                displayName = "Development runner test",
                status = "failed",
                detail = Truncate(exception?.ToString(), 960)
            };
            _report.runners.Add(entry);
        }

        private void NotifyProgress(string detail)
        {
            _onProgress?.Invoke(_completedRunnerCount, TotalRunnerCount, detail);
        }

        private bool HasFailures()
        {
            for (var index = 0; index < _report.runners.Count; index++)
            {
                var status = _report.runners[index].status;
                if (string.Equals(status, "failed", StringComparison.Ordinal)
                    || string.Equals(status, "timed_out", StringComparison.Ordinal))
                    return true;
            }
            return false;
        }

        private void DestroyYoloResult()
        {
            DestroyTexture(_yoloResult.texture);
            DestroyTexture(_yoloResult.mask);
            DestroyTexture(_yoloResult.overlay);
            _yoloResult = default;
            _hasYoloMask = false;
        }

        private void WriteReport()
        {
            try
            {
                var directory = Path.GetDirectoryName(_report.reportPath);
                if (!string.IsNullOrWhiteSpace(directory))
                    Directory.CreateDirectory(directory);
                File.WriteAllText(_report.reportPath, JsonUtility.ToJson(_report, true));
            }
            catch (Exception exception)
            {
                UnityEngine.Debug.LogWarning("Could not write development runner report: " + exception.Message);
            }
        }
    }

    private readonly struct RunnerOutcome
    {
        public readonly bool passed;
        public readonly bool skip;
        public readonly string detail;
        public readonly long elapsedMs;
        public readonly int outputWidth;
        public readonly int outputHeight;
        public readonly int personCount;
        public readonly float maskCoverage01;

        private RunnerOutcome(
            bool passed,
            bool skip,
            string detail,
            long elapsedMs,
            int outputWidth,
            int outputHeight,
            int personCount,
            float maskCoverage01)
        {
            this.passed = passed;
            this.skip = skip;
            this.detail = detail;
            this.elapsedMs = elapsedMs;
            this.outputWidth = outputWidth;
            this.outputHeight = outputHeight;
            this.personCount = personCount;
            this.maskCoverage01 = maskCoverage01;
        }

        public static RunnerOutcome Create(
            bool passed,
            string detail,
            long elapsedMs,
            Texture output = null,
            int personCount = 0,
            float maskCoverage01 = 0f)
        {
            return new RunnerOutcome(
                passed,
                false,
                detail,
                elapsedMs,
                output == null ? 0 : output.width,
                output == null ? 0 : output.height,
                personCount,
                maskCoverage01);
        }

        public static RunnerOutcome FromTexture(Texture2D output, string error, long elapsedMs)
        {
            var outcome = Create(
                string.IsNullOrWhiteSpace(error) && output != null,
                error,
                elapsedMs,
                output);
            DestroyTexture(output);
            return outcome;
        }

        public static RunnerOutcome Skip(string detail)
        {
            return new RunnerOutcome(false, true, detail, 0L, 0, 0, 0, 0f);
        }
    }

    private static AIImageModelGroupId ResolveRealEsrganModelGroup(RealEsrganNcnnReproRunner runner)
    {
        var modelName = runner?.modelName;
        return string.IsNullOrWhiteSpace(modelName)
               || string.Equals(modelName, "realesr-animevideov3-x4", StringComparison.OrdinalIgnoreCase)
            ? AIImageModelGroupId.RealEsrganX4PlusAnime
            : AIImageModelGroupId.RealEsrganOptionalModels;
    }

    private static AIImageModelGroupId ResolveQwen35ModelGroup()
    {
        const string mobileQ4DirectoryName = "qwen3.5_0.8b_mobile_q4";
        var configured = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            var resolved = Qwen35ModelDirectoryResolver.Resolve(configured, mobileQ4DirectoryName);
            var directoryName = Path.GetFileName(resolved.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            if (string.Equals(directoryName, "qwen3.5_0.8b", StringComparison.OrdinalIgnoreCase))
                return AIImageModelGroupId.Qwen35FullPrecision;
            if (string.Equals(directoryName, "qwen3.5_0.8b_mobile_q8", StringComparison.OrdinalIgnoreCase))
                return AIImageModelGroupId.Qwen35MobileQ8;
        }

        return AIImageModelGroupId.Qwen35MobileQ4;
    }

    private static string ResolveQwen35ModelDirectory(AIImageModelGroupId group)
    {
        var modelDirectoryName = group == AIImageModelGroupId.Qwen35FullPrecision
            ? "qwen3.5_0.8b"
            : group == AIImageModelGroupId.Qwen35MobileQ8
                ? "qwen3.5_0.8b_mobile_q8"
                : "qwen3.5_0.8b_mobile_q4";
        var configured = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            var configuredDirectory = Qwen35ModelDirectoryResolver.Resolve(configured, modelDirectoryName);
            if (HasQwen35ModelPayload(configuredDirectory))
                return configuredDirectory;
        }

        var persistentDirectory = Path.Combine(Application.persistentDataPath, modelDirectoryName);
        if (HasQwen35ModelPayload(persistentDirectory))
            return persistentDirectory;

        var deliveredDirectory = Path.Combine(AIImageModelDelivery.PersistentRoot, "QWEN35", modelDirectoryName);
        if (HasQwen35ModelPayload(deliveredDirectory))
            return deliveredDirectory;

        var playerDirectory = Path.Combine(Application.streamingAssetsPath, "QWEN35", modelDirectoryName);
        if (HasQwen35ModelPayload(playerDirectory))
            return playerDirectory;

        if (Aexis.Samples.AexisSampleStreamingAssets.TryResolveDirectoryPath("QWEN35", out var streamingAssetsDirectory))
        {
            var deployedDirectory = Qwen35ModelDirectoryResolver.Resolve(streamingAssetsDirectory, modelDirectoryName);
            if (HasQwen35ModelPayload(deployedDirectory))
                return deployedDirectory;
        }

#if UNITY_EDITOR
        var projectDirectory = Path.GetFullPath(Path.Combine(
            Application.dataPath,
            "..",
            "Tools",
            "Qwen35NcnnBaseline",
            "_models",
            modelDirectoryName));
        if (HasQwen35ModelPayload(projectDirectory))
            return projectDirectory;
#endif
        return persistentDirectory;
    }

    private static bool HasQwen35ModelPayload(string modelDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory) || !Directory.Exists(modelDirectory))
            return false;

        if (File.Exists(Path.Combine(modelDirectory, Qwen35MobileAssetSet.Q4ManifestFileName))
            || File.Exists(Path.Combine(modelDirectory, Qwen35MobileAssetSet.ManifestFileName)))
        {
            try { return Qwen35MobileAssetSet.TryLoad(modelDirectory) != null; }
            catch { return false; }
        }

        var requiredFiles = new[]
        {
            "model.json", "vocab.txt", "merges.txt",
            "qwen3.5_decoder.ncnn.param", "qwen3.5_decoder.ncnn.bin",
            "qwen3.5_embed_token.ncnn.param", "qwen3.5_embed_token.ncnn.bin",
            "qwen3.5_proj_out.ncnn.param",
            "qwen3.5_vision_embed_patch.ncnn.param", "qwen3.5_vision_embed_patch.ncnn.bin",
            "qwen3.5_vision_embed_pos.ncnn.param", "qwen3.5_vision_embed_pos.ncnn.bin",
            "qwen3.5_vision_encoder.ncnn.param", "qwen3.5_vision_encoder.ncnn.bin"
        };
        for (var index = 0; index < requiredFiles.Length; index++)
        {
            if (!File.Exists(Path.Combine(modelDirectory, requiredFiles[index])))
                return false;
        }
        return true;
    }

    private static string Truncate(string value, int maximumCharacters)
    {
        if (string.IsNullOrWhiteSpace(value))
            return string.Empty;
        var normalized = value.Trim().Replace("\r", " ").Replace("\n", " ");
        return normalized.Length <= maximumCharacters
            ? normalized
            : normalized.Substring(0, maximumCharacters) + "...";
    }

    private static void DestroyTexture(Texture texture)
    {
        if (texture != null)
            UnityEngine.Object.Destroy(texture);
    }
}
#endif
