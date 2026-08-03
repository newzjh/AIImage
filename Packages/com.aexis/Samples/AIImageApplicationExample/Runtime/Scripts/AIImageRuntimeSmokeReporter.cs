using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using AIImage.Qwen35;
using Aexis.Samples.Async;
using UnityEngine;
using UnityEngine.SceneManagement;
using Stopwatch = System.Diagnostics.Stopwatch;

internal sealed class AIImageRuntimeSmokeReporter : MonoBehaviour
{
    private const string IntentExtraName = "aiimage_smoke";
    private const string RunnerIntentExtraName = "aiimage_runner_smoke";
    private const string RunnerStageInputIntentExtraName = "aiimage_runner_smoke_stage_input";
    private const string MainView2AutoInputIntentExtraName = "aiimage_mainview2_auto_input";
    private const string QwenDeviceQualificationIntentExtraName = "aiimage_runner_qwen_device_qualification";
    private const string AndroidReportFileName = "aiimage_android_smoke.json";
    private const string AndroidRunnerReportFileName = "aiimage_android_runner_smoke.json";
    private const string AppleReportFileName = "aiimage_apple_smoke.json";
    private const string AppleRunnerReportFileName = "aiimage_apple_runner_smoke.json";
    private const string RunnerInputDirectoryName = "aiimage-smoke";
    private const string AdbRunnerFaceInputPath = "/data/local/tmp/aiimage-smoke-face.png";
    private const string AdbRunnerSceneInputPath = "/data/local/tmp/aiimage-smoke-scene.jpg";
    private const string FaceInputFileName = "face.png";
    private const string SceneInputFileName = "scene.jpg";
    private const string RunnerSmokeCommandLineArgument = "-aiimage_runner_smoke";
    private const string QwenDeviceQualificationCommandLineArgument = "-aiimage_runner_qwen_device_qualification";
    private const string RunnerReportCommandLineArgument = "-aiimage_runner_smoke_report";
    private const string QuitWhenCompleteCommandLineArgument = "-aiimage_runner_smoke_quit_when_done";
    private const string QwenMultiPersonSmokePrompt =
        "Respond directly without explaining the task or your reasoning. First state the total number of distinct people in the image, then list every person from left to right. Do not count an isolated hand, arm, reflection, or other body fragment as an additional person. Do not stop after the first person.";
    private const int ExpectedQwenPeopleCount = 4;
    // Match Main2's product request. The validated mobile Q4 asset set applies
    // its runtime cap, so this also proves that the mobile tuning is active.
    private const int QwenProductRequestedMaxNewTokens = 256;
    private static string LogPrefix => Application.platform == RuntimePlatform.Android
        ? "[AIIMAGE_ANDROID_SMOKE]"
        : "[AEXIS_RUNTIME_SMOKE]";

    [Serializable]
    private sealed class Report
    {
        public string status;
        public string unityVersion;
        public string platform;
        public string operatingSystem;
        public string scene;
        public string deviceModel;
        public string graphicsDeviceType;
        public string graphicsDeviceName;
        public bool supportsComputeShaders;
        public int maxTextureSize;
        public int maxTextureArraySlices;
        public string persistentDataPath;
    }

    [Serializable]
    private sealed class RunnerResult
    {
        public string id;
        public string modelGroup;
        public string status;
        public string detail;
        public long elapsedMs;
        public int outputWidth;
        public int outputHeight;
        public int personCount;
        public float maskCoverage01;
    }

    [Serializable]
    private sealed class RunnerReport
    {
        public string status;
        public string unityVersion;
        public string platform;
        public string operatingSystem;
        public string deviceModel;
        public string graphicsDeviceType;
        public string graphicsDeviceName;
        public int systemMemoryMb;
        public string persistentDataPath;
        public string faceInputPath;
        public string sceneInputPath;
        public bool nonQwenValid;
        public bool valid;
        public string qwenStatus;
        public string detail;
        public List<RunnerResult> runners = new List<RunnerResult>();
    }

    private bool _runDefaultRunnerSmoke;
    private bool _stageRunnerInputsFromAdb;
    private bool _autoSelectMainView2Input;
    private bool _runQwenDeviceQualification;
    private int _qwenDetectedPersonCount;
    private readonly RunnerReport _runnerReport = new RunnerReport();

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void CreateIfRequested()
    {
#if !UNITY_EDITOR
        var smokeRequested = false;
        var runnerSmokeRequested = false;
        var stageRunnerInputsFromAdb = false;
        var autoSelectMainView2Input = false;
        var qwenDeviceQualificationRequested = false;
#if UNITY_ANDROID
        smokeRequested = IsIntentBooleanSet(IntentExtraName);
        runnerSmokeRequested = IsIntentBooleanSet(RunnerIntentExtraName);
        autoSelectMainView2Input = IsIntentBooleanSet(MainView2AutoInputIntentExtraName);
        stageRunnerInputsFromAdb = runnerSmokeRequested
            && IsIntentBooleanSet(RunnerStageInputIntentExtraName);
        qwenDeviceQualificationRequested = runnerSmokeRequested
            && IsIntentBooleanSet(QwenDeviceQualificationIntentExtraName);
#else
        smokeRequested = HasCommandLineArgument(IntentExtraName);
        runnerSmokeRequested = HasCommandLineArgument(RunnerSmokeCommandLineArgument)
            || IsIosAutorunBuild();
        qwenDeviceQualificationRequested = runnerSmokeRequested
            && (HasCommandLineArgument(QwenDeviceQualificationCommandLineArgument) || IsIosAutorunBuild());
#endif
        if (!smokeRequested && !runnerSmokeRequested && !autoSelectMainView2Input)
            return;

        var gameObject = new GameObject(nameof(AIImageRuntimeSmokeReporter));
        DontDestroyOnLoad(gameObject);
        var reporter = gameObject.AddComponent<AIImageRuntimeSmokeReporter>();
        reporter._runDefaultRunnerSmoke = runnerSmokeRequested;
        reporter._stageRunnerInputsFromAdb = stageRunnerInputsFromAdb;
        reporter._autoSelectMainView2Input = autoSelectMainView2Input;
        reporter._runQwenDeviceQualification = qwenDeviceQualificationRequested;
#endif
    }

    private static bool HasCommandLineArgument(string argument)
    {
        if (string.IsNullOrWhiteSpace(argument))
            return false;

        var values = Environment.GetCommandLineArgs();
        for (var index = 0; index < values.Length; index++)
        {
            if (string.Equals(values[index], argument, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }

    private static bool IsIosAutorunBuild()
    {
#if UNITY_IOS && AEXIS_IOS_RUNTIME_SMOKE_AUTORUN
        return true;
#else
        return false;
#endif
    }

    private static bool IsIntentBooleanSet(string key)
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            using (var unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (var activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
            using (var intent = activity?.Call<AndroidJavaObject>("getIntent"))
            {
                return intent != null && intent.Call<bool>("getBooleanExtra", key, false);
            }
        }
        catch (Exception exception)
        {
            Debug.LogWarning(LogPrefix + " request check failed: " + exception.Message);
        }
#endif
        return false;
    }

    private IEnumerator Start()
    {
        yield return null;
        yield return new WaitForEndOfFrame();

        try
        {
            var report = new Report
            {
                status = "ready",
                unityVersion = Application.unityVersion,
                platform = Application.platform.ToString(),
                operatingSystem = SystemInfo.operatingSystem,
                scene = SceneManager.GetActiveScene().name,
                deviceModel = SystemInfo.deviceModel,
                graphicsDeviceType = SystemInfo.graphicsDeviceType.ToString(),
                graphicsDeviceName = SystemInfo.graphicsDeviceName,
                supportsComputeShaders = SystemInfo.supportsComputeShaders,
                maxTextureSize = SystemInfo.maxTextureSize,
                maxTextureArraySlices = SystemInfo.maxTextureArraySlices,
                persistentDataPath = Application.persistentDataPath
            };
            var json = JsonUtility.ToJson(report, true);
            File.WriteAllText(SmokeReportPath, json);
            Debug.Log(LogPrefix + " ready " + json.Replace('\n', ' '));
        }
        catch (Exception exception)
        {
            Debug.LogError(LogPrefix + " failed: " + exception);
        }

        if (_runDefaultRunnerSmoke)
            RunDefaultRunnerSmokeAsync().Forget();

        if (_autoSelectMainView2Input)
            StageAndSelectMainView2InputAsync().Forget();
    }

    private async UniTaskVoid StageAndSelectMainView2InputAsync()
    {
        try
        {
            StageMainView2InputFromAdb();
            var inputPath = ResolveInputPath(SceneInputFileName);
            if (!File.Exists(inputPath))
                throw new FileNotFoundException("MainView2 auto-select input is missing.", inputPath);

            for (var frame = 0; frame < 120; frame++)
            {
                var host = UnityEngine.Object.FindFirstObjectByType<AIImagePageHost>();
                if (host != null && host.ReloadMainImageFromDisk(inputPath, true))
                {
                    Debug.Log(LogPrefix + " MainView2 auto-selected input=" + inputPath);
                    return;
                }

                await UniTask.Yield(PlayerLoopTiming.Update);
            }

            throw new TimeoutException("AIImagePageHost did not accept the staged MainView2 input within 120 frames.");
        }
        catch (Exception exception)
        {
            Debug.LogError(LogPrefix + " MainView2 auto-select failed: " + exception);
        }
    }

    private async UniTaskVoid RunDefaultRunnerSmokeAsync()
    {
        InitializeRunnerReport();
        WriteRunnerReport();

        Texture2D faceInput = null;
        Texture2D sceneInput = null;
        try
        {
            if (_stageRunnerInputsFromAdb)
                StageRunnerInputsFromAdb();

            var requiredGraphicsApi = RequiredGraphicsApi;
            if (SystemInfo.graphicsDeviceType != requiredGraphicsApi)
                throw new InvalidOperationException(
                    Application.platform + " runner smoke requires " + requiredGraphicsApi
                    + ". Active API is " + SystemInfo.graphicsDeviceType + ".");

            // Main2 registers this in its page host. The intent-driven harness
            // does not depend on that UI lifecycle, so register it explicitly.
            Aexis.Samples.AexisSampleStreamingAssets.RegisterManifestPathResolver();

            faceInput = LoadInputTexture(FaceInputFileName);
            sceneInput = LoadInputTexture(SceneInputFileName);

            await RunCodeFormerAsync(faceInput);
            await RunGfpganAsync(faceInput);
            await RunMattingAsync(sceneInput);
            await RunRealEsrganAsync(faceInput);
            await RunYoloAndDeepFillAsync(faceInput);
            await RunClipAsync(faceInput);
            // The face input is ref/02.png: four people arranged from left to right.
            // It guards against a valid-looking response that silently stops after
            // describing only the first person.
            if (_runQwenDeviceQualification)
                await RunQwenAsync(faceInput, true);
            else
                _runnerReport.qwenStatus = "not_requested";

            _runnerReport.nonQwenValid = AllNonQwenPassed();
            _runnerReport.valid = _runnerReport.nonQwenValid
                && (!_runQwenDeviceQualification
                    || string.Equals(_runnerReport.qwenStatus, "passed", StringComparison.Ordinal));
            _runnerReport.status = _runnerReport.valid ? "passed" : "completed_with_failures";
            _runnerReport.detail = _runnerReport.valid
                ? "All default Main2 runners passed on Android Vulkan."
                : "See individual runner results; QWEN is blocked when the device compatibility gate is not met.";
        }
        catch (Exception exception)
        {
            _runnerReport.status = "failed";
            _runnerReport.detail = exception.ToString();
            _runnerReport.nonQwenValid = false;
            _runnerReport.valid = false;
            Debug.LogError(LogPrefix + " default runner smoke failed: " + exception);
        }
        finally
        {
            DestroyTexture(faceInput);
            DestroyTexture(sceneInput);
            WriteRunnerReport();
            Debug.Log(LogPrefix + " default runner smoke " + _runnerReport.status
                + " valid=" + _runnerReport.valid
                + " path=" + RunnerReportPath);
            Debug.Log(LogPrefix + " runner-report=" + JsonUtility.ToJson(_runnerReport));
            if (Application.platform == RuntimePlatform.OSXPlayer
                && HasCommandLineArgument(QuitWhenCompleteCommandLineArgument))
            {
                Application.Quit(0);
            }
        }
    }

    private void InitializeRunnerReport()
    {
        _runnerReport.status = "running";
        _runnerReport.unityVersion = Application.unityVersion;
        _runnerReport.platform = Application.platform.ToString();
        _runnerReport.operatingSystem = SystemInfo.operatingSystem;
        _runnerReport.deviceModel = SystemInfo.deviceModel;
        _runnerReport.graphicsDeviceType = SystemInfo.graphicsDeviceType.ToString();
        _runnerReport.graphicsDeviceName = SystemInfo.graphicsDeviceName;
        _runnerReport.systemMemoryMb = SystemInfo.systemMemorySize;
        _runnerReport.persistentDataPath = Application.persistentDataPath;
        _runnerReport.faceInputPath = ResolveInputPath(FaceInputFileName);
        _runnerReport.sceneInputPath = ResolveInputPath(SceneInputFileName);
        _runnerReport.nonQwenValid = false;
        _runnerReport.valid = false;
        _runnerReport.qwenStatus = "not_started";
        _runnerReport.detail = "Preparing default Main2 Android runner validation.";
        _runnerReport.runners.Clear();
    }

    private async UniTask RunCodeFormerAsync(Texture2D input)
    {
        var entry = BeginRunner("codeformer", AIImageModelGroupId.CodeFormerDefault);
        GameObject gameObject = null;
        try
        {
            await MaterializeGroupAsync(AIImageModelGroupId.CodeFormerDefault, entry);
            gameObject = new GameObject("AndroidSmoke_CodeFormer");
            var runner = gameObject.AddComponent<CodeFormerNcnnReproRunner2>();
            runner.enableDebugDump = false;
            runner.enableFaceRegionDebugDump = false;
            var result = await runner.ProcessAsync(input, CancellationToken.None);
            CompleteRunner(entry, string.IsNullOrWhiteSpace(result.error) && result.texture != null, result.error, result.elapsedMs, result.texture);
            DestroyTexture(result.texture);
        }
        catch (Exception exception)
        {
            CompleteRunner(entry, false, exception.Message, 0L, null);
        }
        finally
        {
            DestroyObject(gameObject);
            WriteRunnerReport();
            await UniTask.Yield(PlayerLoopTiming.Update);
        }
    }

    private async UniTask RunRealEsrganAsync(Texture2D input)
    {
        var entry = BeginRunner("realesr_animevideov3_x4", AIImageModelGroupId.RealEsrganX4PlusAnime);
        GameObject gameObject = null;
        Texture2D smallInput = null;
        try
        {
            await MaterializeGroupAsync(AIImageModelGroupId.RealEsrganX4PlusAnime, entry);
            smallInput = ResizeInput(input, 128);
            gameObject = new GameObject("AndroidSmoke_RealEsrgan");
            var runner = gameObject.AddComponent<RealEsrganNcnnReproRunner>();
            runner.modelName = "realesr-animevideov3-x4";
            runner.enableGpuLayerProfiling = false;
            runner.useCommandBuffer = false;
            runner.disallowBufferAccess = true;
            runner.disallowBufferOutputs = true;
            runner.disallowBufferToTextureMaterialization = true;
            var result = await runner.ProcessAsync(smallInput, CancellationToken.None);
            CompleteRunner(entry, string.IsNullOrWhiteSpace(result.error) && result.texture != null, result.error, result.elapsedMs, result.texture);
            DestroyTexture(result.texture);
        }
        catch (Exception exception)
        {
            CompleteRunner(entry, false, exception.Message, 0L, null);
        }
        finally
        {
            DestroyTexture(smallInput);
            DestroyObject(gameObject);
            WriteRunnerReport();
            await UniTask.Yield(PlayerLoopTiming.Update);
        }
    }

    private async UniTask RunGfpganAsync(Texture2D input)
    {
        var entry = BeginRunner("gfpgan", AIImageModelGroupId.GfpganDefault);
        GameObject gameObject = null;
        try
        {
            await MaterializeGroupAsync(AIImageModelGroupId.GfpganDefault, entry);
            gameObject = new GameObject("AndroidSmoke_Gfpgan");
            var runner = gameObject.AddComponent<GfpganNcnnReproRunner>();
            runner.enableFaceRegionDebugDump = false;
            runner.disallowBufferAccess = true;
            runner.disallowBufferOutputs = true;
            runner.disallowBufferToTextureMaterialization = true;
            var result = await runner.ProcessAsync(input, CancellationToken.None);
            CompleteRunner(entry, string.IsNullOrWhiteSpace(result.error) && result.texture != null, result.error, result.elapsedMs, result.texture);
            DestroyTexture(result.texture);
        }
        catch (Exception exception)
        {
            CompleteRunner(entry, false, exception.Message, 0L, null);
        }
        finally
        {
            DestroyObject(gameObject);
            WriteRunnerReport();
            await UniTask.Yield(PlayerLoopTiming.Update);
        }
    }

    private async UniTask RunYoloAndDeepFillAsync(Texture2D input)
    {
        var yoloEntry = BeginRunner("yolov8_person_segmentation", AIImageModelGroupId.YoloV8PersonSegmentation);
        var deepFillEntry = BeginRunner("deepfillv2_case1_ncnn", AIImageModelGroupId.DeepFillV2Case1Ncnn);
        GameObject gameObject = null;
        YoloSegResult yolo = default;
        try
        {
            await MaterializeGroupAsync(AIImageModelGroupId.YoloV8PersonSegmentation, yoloEntry);
            gameObject = new GameObject("AndroidSmoke_YoloDeepFill");
            var yoloRunner = gameObject.AddComponent<YoloSegNcnnReproRunner>();
            yoloRunner.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
            yoloRunner.enableDebugDump = false;
            yoloRunner.targetPersonOnly = true;
            yoloRunner.enableMaskClose = true;
            yoloRunner.enableMaskDilate = true;
            yoloRunner.disallowBufferAccess = true;
            yoloRunner.disallowBufferOutputs = true;
            yoloRunner.disallowBufferToTextureMaterialization = true;
            yolo = await yoloRunner.ProcessAsync(input, CancellationToken.None);
            var yoloPassed = string.IsNullOrWhiteSpace(yolo.error) && yolo.mask != null && yolo.personCount > 0;
            CompleteRunner(yoloEntry, yoloPassed, yolo.error, yolo.elapsedMs, yolo.texture, yolo.personCount, yolo.maskCoverage01);
            if (!yoloPassed)
            {
                CompleteRunner(deepFillEntry, false, "YOLO did not produce a person mask; DeepFillV2 was not started.", 0L, null);
                return;
            }
            _qwenDetectedPersonCount = yolo.personCount;

            await MaterializeGroupAsync(AIImageModelGroupId.DeepFillV2Case1Ncnn, deepFillEntry);
            var deepFillRunner = gameObject.AddComponent<DeepFillV2Runner>();
            deepFillRunner.backend = DeepFillV2Backend.NcnnBin;
            deepFillRunner.enableDebugDump = false;
            var result = await deepFillRunner.ProcessAsync(input, yolo.mask, CancellationToken.None);
            CompleteRunner(deepFillEntry, string.IsNullOrWhiteSpace(result.error) && result.texture != null, result.error, result.elapsedMs, result.texture);
            DestroyTexture(result.texture);
        }
        catch (Exception exception)
        {
            if (string.Equals(yoloEntry.status, "running", StringComparison.Ordinal))
                CompleteRunner(yoloEntry, false, exception.Message, 0L, null);
            if (string.Equals(deepFillEntry.status, "running", StringComparison.Ordinal))
                CompleteRunner(deepFillEntry, false, exception.Message, 0L, null);
        }
        finally
        {
            DestroyTexture(yolo.texture);
            DestroyTexture(yolo.mask);
            DestroyTexture(yolo.overlay);
            DestroyObject(gameObject);
            WriteRunnerReport();
            await UniTask.Yield(PlayerLoopTiming.Update);
        }
    }

    private async UniTask RunMattingAsync(Texture2D input)
    {
        var entry = BeginRunner("matting", AIImageModelGroupId.Matting);
        GameObject gameObject = null;
        try
        {
            await MaterializeGroupAsync(AIImageModelGroupId.Matting, entry);
            gameObject = new GameObject("AndroidSmoke_Matting");
            var runner = gameObject.AddComponent<MatterNcnnReproRunner>();
            runner.enableDebugDump = false;
            runner.forceBufferConvolution = false;
            runner.disallowBufferAccess = true;
            runner.disallowBufferOutputs = true;
            runner.disallowBufferToTextureMaterialization = true;
            var result = await runner.ProcessAsync(input, CancellationToken.None);
            CompleteRunner(entry, string.IsNullOrWhiteSpace(result.error) && result.texture != null && result.matte != null, result.error, result.elapsedMs, result.texture);
            DestroyTexture(result.texture);
            DestroyTexture(result.matte);
        }
        catch (Exception exception)
        {
            CompleteRunner(entry, false, exception.Message, 0L, null);
        }
        finally
        {
            DestroyObject(gameObject);
            WriteRunnerReport();
            await UniTask.Yield(PlayerLoopTiming.Update);
        }
    }

    private async UniTask RunClipAsync(Texture2D input)
    {
        var entry = BeginRunner("clip_mobileclip_s0", AIImageModelGroupId.ClipMobileClipS0);
        GameObject gameObject = null;
        try
        {
            await MaterializeGroupAsync(AIImageModelGroupId.ClipMobileClipS0, entry);
            gameObject = new GameObject("AndroidSmoke_Clip");
            var runner = gameObject.AddComponent<ClipNcnnReproRunner>();
            runner.enableDebugDump = false;
            runner.forceFullRenderTexturePath = true;
            runner.disallowBufferAccess = true;
            runner.disallowBufferOutputs = true;
            runner.disallowBufferToTextureMaterialization = true;
            var result = await runner.ProcessAsync(input, CancellationToken.None);
            var passed = string.IsNullOrWhiteSpace(result.error)
                && result.scores != null
                && result.scores.Length > 0
                && !string.IsNullOrWhiteSpace(result.bestLabel);
            CompleteRunner(entry, passed, result.error, result.elapsedMs, null);
        }
        catch (Exception exception)
        {
            CompleteRunner(entry, false, exception.Message, 0L, null);
        }
        finally
        {
            DestroyObject(gameObject);
            WriteRunnerReport();
            await UniTask.Yield(PlayerLoopTiming.Update);
        }
    }

    private async UniTask RunQwenAsync(Texture2D input, bool deviceQualificationRun)
    {
        var entry = BeginRunner("qwen35_mobile_q4", AIImageModelGroupId.Qwen35MobileQ4);
        try
        {
            var compatibility = deviceQualificationRun
                ? Qwen35DeviceCompatibility.EvaluateCapabilitiesForDeviceQualification(
                    hasValidatedMobileAssets: true,
                    Qwen35DeviceCapabilities.CaptureCurrent(),
                    quantizationBits: 4)
                : Qwen35DeviceCompatibility.EvaluateCapabilities(
                    hasValidatedMobileAssets: true,
                    Qwen35DeviceCapabilities.CaptureCurrent(),
                    quantizationBits: 4);
            if (!compatibility.Supported)
            {
                _runnerReport.qwenStatus = "blocked_device_compatibility";
                entry.status = "blocked";
                entry.detail = string.Join(" | ", compatibility.UnsupportedReasons);
                WriteRunnerReport();
                return;
            }

            await MaterializeGroupAsync(AIImageModelGroupId.Qwen35MobileQ4, entry);
            var directory = Path.Combine(
                AIImageModelDelivery.PersistentRoot,
                "QWEN35",
                "qwen3.5_0.8b_mobile_q4");
            var stopwatch = Stopwatch.StartNew();
            using (var runner = deviceQualificationRun
                ? await Qwen35Runner.CreateForDeviceQualificationAsync(
                    directory,
                    QwenProductRequestedMaxNewTokens,
                    true,
                    CancellationToken.None)
                : await Qwen35Runner.CreateAsync(
                    directory,
                    QwenProductRequestedMaxNewTokens,
                    true,
                    CancellationToken.None))
            {
                var result = await runner.GenerateImageAsync(
                    input,
                    BuildQwenMultiPersonSmokePrompt(_qwenDetectedPersonCount),
                    Qwen35SamplingConfig.Greedy(),
                    CancellationToken.None);
                stopwatch.Stop();
                var passed = HasMeaningfulQwenOutput(result)
                    && MentionsExpectedPeopleCount(result.Text, ExpectedQwenPeopleCount);
                var detail = "tokens=" + string.Join(",", result.TokenIds)
                    + " | maxNewTokens=" + runner.MaxNewTokens
                    + " | stoppedOnEndOfTurn=" + result.StoppedOnEndOfTurn
                    + " | decoderSteps=" + result.DecoderStepCount
                    + " | visibleTextChars=" + (result.Text ?? string.Empty).Length
                    + " | expectedPeople=" + ExpectedQwenPeopleCount
                    + " | detectedPeople=" + _qwenDetectedPersonCount
                    + " | text=" + FormatQwenTextForReport(result.Text);
                _runnerReport.qwenStatus = passed ? "passed" : "failed";
                CompleteRunner(entry, passed, detail, stopwatch.ElapsedMilliseconds, null);
            }
        }
        catch (Exception exception)
        {
            _runnerReport.qwenStatus = "failed";
            CompleteRunner(entry, false, exception.Message, 0L, null);
        }
        finally
        {
            WriteRunnerReport();
            await UniTask.Yield(PlayerLoopTiming.Update);
        }
    }

    private static string BuildQwenMultiPersonSmokePrompt(int detectedPersonCount)
    {
        if (detectedPersonCount <= 0)
            return QwenMultiPersonSmokePrompt;

        return QwenMultiPersonSmokePrompt
            + " A local person detector has verified exactly " + detectedPersonCount
            + " distinct people. Use that verified count when writing the first line and enumerate that many people.";
    }

    private static bool HasMeaningfulQwenOutput(Qwen35GenerationResult result)
    {
        if (result == null || result.TokenIds == null || result.TokenIds.Count < 2
            || result.DecoderStepCount < 2 || result.FinalPosition <= 0)
        {
            return false;
        }

        var text = (result.Text ?? string.Empty).Trim();
        if (text.Length < 2 || text.IndexOf("Iterations", StringComparison.OrdinalIgnoreCase) >= 0)
            return false;

        var meaningfulCharacters = 0;
        for (var index = 0; index < text.Length; index++)
        {
            if (char.IsLetterOrDigit(text[index]))
                meaningfulCharacters++;
        }

        if (meaningfulCharacters < 2)
            return false;

        // A quantization or token-index fault can yield a long run of token
        // brackets or a repeated alphanumeric character. Do not reject normal
        // Markdown emphasis, which can legitimately contain consecutive '*'.
        var consecutiveRepeats = 1;
        for (var index = 1; index < text.Length; index++)
        {
            if (text[index] == text[index - 1])
            {
                consecutiveRepeats++;
                if (consecutiveRepeats >= 4
                    && (text[index] == '[' || text[index] == ']'
                        || char.IsLetterOrDigit(text[index])))
                    return false;
            }
            else
            {
                consecutiveRepeats = 1;
            }
        }

        return true;
    }

    private static string FormatQwenTextForReport(string text)
    {
        const int maximumCharacters = 240;
        var formatted = (text ?? string.Empty)
            .Trim()
            .Replace("\\", "\\\\")
            .Replace("\r", "\\r")
            .Replace("\n", "\\n");
        return formatted.Length <= maximumCharacters
            ? formatted
            : formatted.Substring(0, maximumCharacters) + "...";
    }

    private static bool MentionsExpectedPeopleCount(string text, int expectedPeopleCount)
    {
        var value = (text ?? string.Empty).Trim();
        if (expectedPeopleCount != 4) return false;

        return value.IndexOf("four people", StringComparison.OrdinalIgnoreCase) >= 0
            || value.IndexOf("4 people", StringComparison.OrdinalIgnoreCase) >= 0
            || value.IndexOf("four distinct people", StringComparison.OrdinalIgnoreCase) >= 0
            || value.IndexOf("4 distinct people", StringComparison.OrdinalIgnoreCase) >= 0
            || value.IndexOf("people: 4", StringComparison.OrdinalIgnoreCase) >= 0
            || value.IndexOf("四个人", StringComparison.Ordinal) >= 0
            || value.IndexOf("四人", StringComparison.Ordinal) >= 0
            || value.IndexOf("人物数量：4", StringComparison.Ordinal) >= 0
            || value.IndexOf("人物总数：4", StringComparison.Ordinal) >= 0;
    }

    private async UniTask MaterializeGroupAsync(AIImageModelGroupId id, RunnerResult entry)
    {
        var group = AIImageModelDelivery.GetGroup(id);
        entry.status = "preparing";
        entry.detail = "Materializing " + group.DisplayName + ".";
        WriteRunnerReport();
        await AIImageModelDelivery.MaterializeBundledGroupAsync(
            group,
            progress =>
            {
                entry.detail = progress.Detail + " " + Mathf.RoundToInt(progress.Progress01 * 100f) + "%";
                WriteRunnerReport();
            },
            CancellationToken.None);
        entry.status = "running";
        entry.detail = "Executing default model path.";
        WriteRunnerReport();
    }

    private RunnerResult BeginRunner(string id, AIImageModelGroupId group)
    {
        var entry = new RunnerResult
        {
            id = id,
            modelGroup = group.ToString(),
            status = "queued",
            detail = string.Empty
        };
        _runnerReport.runners.Add(entry);
        WriteRunnerReport();
        return entry;
    }

    private static void CompleteRunner(
        RunnerResult entry,
        bool passed,
        string detail,
        long elapsedMs,
        Texture output,
        int personCount = 0,
        float maskCoverage = 0f)
    {
        entry.status = passed ? "passed" : "failed";
        entry.detail = detail ?? string.Empty;
        entry.elapsedMs = elapsedMs;
        entry.personCount = personCount;
        entry.maskCoverage01 = maskCoverage;
        if (output != null)
        {
            entry.outputWidth = output.width;
            entry.outputHeight = output.height;
        }

        var conciseDetail = entry.detail.Replace('\r', ' ').Replace('\n', ' ');
        if (conciseDetail.Length > 240)
            conciseDetail = conciseDetail.Substring(0, 240);
        Debug.Log(LogPrefix + " runner-complete id=" + entry.id
            + " status=" + entry.status
            + " elapsed_ms=" + entry.elapsedMs
            + " detail=" + conciseDetail);
    }

    private bool AllNonQwenPassed()
    {
        for (var index = 0; index < _runnerReport.runners.Count; index++)
        {
            var entry = _runnerReport.runners[index];
            if (string.Equals(entry.id, "qwen35_mobile_q4", StringComparison.Ordinal)
                || string.Equals(entry.id, "qwen35_mobile_q8", StringComparison.Ordinal))
                continue;
            if (!string.Equals(entry.status, "passed", StringComparison.Ordinal))
                return false;
        }
        return _runnerReport.runners.Count >= 7;
    }

    private Texture2D LoadInputTexture(string fileName)
    {
        var path = ResolveInputPath(fileName);
        if (!File.Exists(path))
            throw new FileNotFoundException("Runtime runner smoke input is missing.", path);
        var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, true)
        {
            name = fileName,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear
        };
        if (!texture.LoadImage(File.ReadAllBytes(path), false))
        {
            DestroyTexture(texture);
            throw new InvalidDataException("Could not decode runtime runner smoke input: " + path);
        }
        return texture;
    }

    private static Texture2D ResizeInput(Texture2D source, int maxLongSide)
    {
        if (source == null)
            return null;
        var longSide = Mathf.Max(source.width, source.height);
        if (longSide <= maxLongSide)
            return source;
        var scale = maxLongSide / (float)longSide;
        var width = Mathf.Max(1, Mathf.RoundToInt(source.width * scale));
        var height = Mathf.Max(1, Mathf.RoundToInt(source.height * scale));
        // Smoke inputs carry raw encoded RGB for the model runners. Keep the resize
        // target Linear so a Gamma/Linear project setting cannot alter the fixture.
        var temporary = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        var previous = RenderTexture.active;
        try
        {
            Graphics.Blit(source, temporary);
            RenderTexture.active = temporary;
            var resized = new Texture2D(width, height, TextureFormat.RGBA32, false, true)
            {
                name = source.name + "_" + maxLongSide,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear
            };
            resized.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            resized.Apply(false, false);
            return resized;
        }
        finally
        {
            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(temporary);
        }
    }

    private static void DestroyObject(UnityEngine.Object value)
    {
        if (value != null)
            UnityEngine.Object.Destroy(value);
    }

    private static void DestroyTexture(Texture value)
    {
        if (value != null)
            UnityEngine.Object.Destroy(value);
    }

    private static UnityEngine.Rendering.GraphicsDeviceType RequiredGraphicsApi => Application.platform == RuntimePlatform.Android
        ? UnityEngine.Rendering.GraphicsDeviceType.Vulkan
        : UnityEngine.Rendering.GraphicsDeviceType.Metal;

    private void StageRunnerInputsFromAdb()
    {
#if UNITY_ANDROID && DEVELOPMENT_BUILD
        Directory.CreateDirectory(RunnerInputDirectoryPath);
        CopyStagedRunnerInput(AdbRunnerFaceInputPath, FaceInputFileName);
        CopyStagedRunnerInput(AdbRunnerSceneInputPath, SceneInputFileName);
#else
        throw new InvalidOperationException("ADB runner input staging is only supported by Android development builds.");
#endif
    }

    private void StageMainView2InputFromAdb()
    {
#if UNITY_ANDROID && DEVELOPMENT_BUILD
        Directory.CreateDirectory(RunnerInputDirectoryPath);
        CopyStagedRunnerInput(AdbRunnerSceneInputPath, SceneInputFileName);
#else
        throw new InvalidOperationException("ADB MainView2 input staging is only supported by Android development builds.");
#endif
    }

    private void CopyStagedRunnerInput(string sourcePath, string destinationFileName)
    {
        if (!File.Exists(sourcePath))
            throw new FileNotFoundException("ADB-staged runner input is missing.", sourcePath);

        var destinationPath = Path.Combine(RunnerInputDirectoryPath, destinationFileName);
        File.Copy(sourcePath, destinationPath, true);
    }

    private string RunnerInputDirectoryPath => Path.Combine(Application.persistentDataPath, RunnerInputDirectoryName);
    private string SmokeReportPath => Path.Combine(
        Application.platform == RuntimePlatform.Android ? AndroidInternalCachePath : Application.persistentDataPath,
        Application.platform == RuntimePlatform.Android ? AndroidReportFileName : AppleReportFileName);
    private string RunnerReportPath
    {
        get
        {
            var configured = GetCommandLineValue(RunnerReportCommandLineArgument);
            if (Application.platform != RuntimePlatform.IPhonePlayer && !string.IsNullOrWhiteSpace(configured))
                return Path.GetFullPath(configured);
            return Path.Combine(
                Application.platform == RuntimePlatform.Android ? AndroidInternalCachePath : Application.persistentDataPath,
                Application.platform == RuntimePlatform.Android ? AndroidRunnerReportFileName : AppleRunnerReportFileName);
        }
    }

    private static string AndroidInternalCachePath
    {
        get
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            using (var unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (var activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
            using (var cacheDirectory = activity?.Call<AndroidJavaObject>("getCacheDir"))
            {
                var path = cacheDirectory?.Call<string>("getAbsolutePath");
                if (!string.IsNullOrWhiteSpace(path))
                    return path;
            }
#endif
            return Application.temporaryCachePath;
        }
    }

    private string ResolveInputPath(string fileName)
    {
        var persistentPath = Path.Combine(RunnerInputDirectoryPath, fileName);
        if (File.Exists(persistentPath) || Application.platform == RuntimePlatform.Android)
            return persistentPath;

        return Path.Combine(Application.streamingAssetsPath, RunnerInputDirectoryName, fileName);
    }

    private static string GetCommandLineValue(string argument)
    {
        var values = Environment.GetCommandLineArgs();
        for (var index = 0; index + 1 < values.Length; index++)
        {
            if (string.Equals(values[index], argument, StringComparison.OrdinalIgnoreCase))
                return values[index + 1];
        }

        return string.Empty;
    }

    private void WriteRunnerReport()
    {
        try
        {
            var directory = Path.GetDirectoryName(RunnerReportPath);
            if (!string.IsNullOrWhiteSpace(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(RunnerReportPath, JsonUtility.ToJson(_runnerReport, true));
        }
        catch (Exception exception)
        {
            Debug.LogWarning(LogPrefix + " could not write runner smoke report: " + exception.Message);
        }
    }
}
