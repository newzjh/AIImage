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

internal sealed class AIImageAndroidRuntimeSmokeReporter : MonoBehaviour
{
    private const string IntentExtraName = "aiimage_smoke";
    private const string RunnerIntentExtraName = "aiimage_runner_smoke";
    private const string QwenDeviceQualificationIntentExtraName = "aiimage_runner_qwen_device_qualification";
    private const string ReportFileName = "aiimage_android_smoke.json";
    private const string RunnerReportFileName = "aiimage_android_runner_smoke.json";
    private const string RunnerInputDirectoryName = "aiimage-smoke";
    private const string FaceInputFileName = "face.png";
    private const string SceneInputFileName = "scene.jpg";
    // Match Main2's product request. The validated mobile Q4 asset set applies
    // its runtime cap, so this also proves that the mobile tuning is active.
    private const int QwenProductRequestedMaxNewTokens = 256;
    private const string LogPrefix = "[AIIMAGE_ANDROID_SMOKE]";

    [Serializable]
    private sealed class Report
    {
        public string status;
        public string unityVersion;
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
        public string graphicsDeviceType;
        public string graphicsDeviceName;
        public int systemMemoryMb;
        public bool nonQwenValid;
        public bool valid;
        public string qwenStatus;
        public string detail;
        public List<RunnerResult> runners = new List<RunnerResult>();
    }

    private bool _runDefaultRunnerSmoke;
    private bool _runQwenDeviceQualification;
    private readonly RunnerReport _runnerReport = new RunnerReport();

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void CreateIfRequested()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        var smokeRequested = IsIntentBooleanSet(IntentExtraName);
        var runnerSmokeRequested = IsIntentBooleanSet(RunnerIntentExtraName);
        if (!smokeRequested && !runnerSmokeRequested)
            return;

        var gameObject = new GameObject(nameof(AIImageAndroidRuntimeSmokeReporter));
        DontDestroyOnLoad(gameObject);
        var reporter = gameObject.AddComponent<AIImageAndroidRuntimeSmokeReporter>();
        reporter._runDefaultRunnerSmoke = runnerSmokeRequested;
        reporter._runQwenDeviceQualification = runnerSmokeRequested
            && IsIntentBooleanSet(QwenDeviceQualificationIntentExtraName);
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
            File.WriteAllText(Path.Combine(Application.persistentDataPath, ReportFileName), json);
            Debug.Log(LogPrefix + " ready " + json.Replace('\n', ' '));
        }
        catch (Exception exception)
        {
            Debug.LogError(LogPrefix + " failed: " + exception);
        }

        if (_runDefaultRunnerSmoke)
            RunDefaultRunnerSmokeAsync().Forget();
    }

    private async UniTaskVoid RunDefaultRunnerSmokeAsync()
    {
        InitializeRunnerReport();
        WriteRunnerReport();

        Texture2D faceInput = null;
        Texture2D sceneInput = null;
        try
        {
            if (SystemInfo.graphicsDeviceType != UnityEngine.Rendering.GraphicsDeviceType.Vulkan)
                throw new InvalidOperationException("Android runner smoke requires Vulkan. Active API is " + SystemInfo.graphicsDeviceType + ".");

            // Main2 registers this in its page host. The intent-driven harness
            // does not depend on that UI lifecycle, so register it explicitly.
            Aexis.Samples.AexisSampleStreamingAssets.RegisterManifestPathResolver();

            faceInput = LoadInputTexture(FaceInputFileName);
            sceneInput = LoadInputTexture(SceneInputFileName);

            await RunCodeFormerAsync(faceInput);
            await RunRealEsrganAsync(faceInput);
            await RunGfpganAsync(faceInput);
            await RunYoloAndDeepFillAsync(faceInput);
            await RunMattingAsync(sceneInput);
            await RunClipAsync(faceInput);
            await RunQwenAsync(sceneInput, _runQwenDeviceQualification);

            _runnerReport.nonQwenValid = AllNonQwenPassed();
            _runnerReport.valid = _runnerReport.nonQwenValid
                && string.Equals(_runnerReport.qwenStatus, "passed", StringComparison.Ordinal);
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
        }
    }

    private void InitializeRunnerReport()
    {
        _runnerReport.status = "running";
        _runnerReport.unityVersion = Application.unityVersion;
        _runnerReport.graphicsDeviceType = SystemInfo.graphicsDeviceType.ToString();
        _runnerReport.graphicsDeviceName = SystemInfo.graphicsDeviceName;
        _runnerReport.systemMemoryMb = SystemInfo.systemMemorySize;
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
        var entry = BeginRunner("realesrgan_x4plus_anime", AIImageModelGroupId.RealEsrganX4PlusAnime);
        GameObject gameObject = null;
        Texture2D smallInput = null;
        try
        {
            await MaterializeGroupAsync(AIImageModelGroupId.RealEsrganX4PlusAnime, entry);
            smallInput = ResizeInput(input, 128);
            gameObject = new GameObject("AndroidSmoke_RealEsrgan");
            var runner = gameObject.AddComponent<RealEsrganNcnnReproRunner>();
            runner.modelName = "realesrgan-x4plus-anime";
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
        var deepFillEntry = BeginRunner("deepfillv2_case1_onnx", AIImageModelGroupId.DeepFillV2Case1Onnx);
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

            await MaterializeGroupAsync(AIImageModelGroupId.DeepFillV2Case1Onnx, deepFillEntry);
            var deepFillRunner = gameObject.AddComponent<DeepFillV2Runner>();
            deepFillRunner.backend = DeepFillV2Backend.OnnxDirect;
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
                    "Describe the image in a few words.",
                    Qwen35SamplingConfig.Greedy(),
                    CancellationToken.None);
                stopwatch.Stop();
                var passed = HasMeaningfulQwenOutput(result);
                var detail = "tokens=" + string.Join(",", result.TokenIds)
                    + " | maxNewTokens=" + runner.MaxNewTokens
                    + " | stoppedOnEndOfTurn=" + result.StoppedOnEndOfTurn
                    + " | decoderSteps=" + result.DecoderStepCount
                    + " | visibleTextChars=" + (result.Text ?? string.Empty).Length
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
    }

    private bool AllNonQwenPassed()
    {
        for (var index = 0; index < _runnerReport.runners.Count; index++)
        {
            var entry = _runnerReport.runners[index];
            if (string.Equals(entry.id, "qwen35_mobile_q8", StringComparison.Ordinal))
                continue;
            if (!string.Equals(entry.status, "passed", StringComparison.Ordinal))
                return false;
        }
        return _runnerReport.runners.Count >= 7;
    }

    private Texture2D LoadInputTexture(string fileName)
    {
        var path = Path.Combine(RunnerInputDirectoryPath, fileName);
        if (!File.Exists(path))
            throw new FileNotFoundException("Android runner smoke input is missing.", path);
        var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, true)
        {
            name = fileName,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear
        };
        if (!texture.LoadImage(File.ReadAllBytes(path), false))
        {
            DestroyTexture(texture);
            throw new InvalidDataException("Could not decode Android runner smoke input: " + path);
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
        var temporary = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.ARGB32);
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

    private string RunnerInputDirectoryPath => Path.Combine(Application.persistentDataPath, RunnerInputDirectoryName);
    private string RunnerReportPath => Path.Combine(Application.persistentDataPath, RunnerReportFileName);

    private void WriteRunnerReport()
    {
        try
        {
            File.WriteAllText(RunnerReportPath, JsonUtility.ToJson(_runnerReport, true));
        }
        catch (Exception exception)
        {
            Debug.LogWarning(LogPrefix + " could not write runner smoke report: " + exception.Message);
        }
    }
}
