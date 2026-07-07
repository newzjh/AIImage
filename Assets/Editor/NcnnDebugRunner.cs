#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using Cysharp.Threading.Tasks;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Profiling;
using Debug = UnityEngine.Debug;

public static class NcnnDebugRunner
{
    private const string DebugInputEnvVar = "AIIMAGE_DEBUG_INPUT";
    private const string FaceBufferPathEnvVar = "AIIMAGE_FACE_BUFFER_PATH";
    private const string FaceProbThresholdEnvVar = "AIIMAGE_FACE_PROB_THRESHOLD";
    private const string FaceNmsThresholdEnvVar = "AIIMAGE_FACE_NMS_THRESHOLD";
    private const string StressCountEnvVar = "AIIMAGE_STRESS_COUNT";
    private const string StressInputDirEnvVar = "AIIMAGE_STRESS_INPUT_DIR";
    private const string RealEsrganInputDirEnvVar = "AIIMAGE_ESRGAN_INPUT_DIR";
    private const string RealEsrganModelEnvVar = "AIIMAGE_ESRGAN_MODEL";
    private const string RealEsrganModelsEnvVar = "AIIMAGE_ESRGAN_MODELS";
    private const string RealEsrganUseCommandBufferEnvVar = "AIIMAGE_ESRGAN_USE_COMMAND_BUFFER";
    private const string RealEsrganPack4OnlyGuardEnvVar = "AIIMAGE_ESRGAN_PACK4_ONLY_GUARD";
    private const string RealEsrganValidationReuseRunnerEnvVar = "AIIMAGE_ESRGAN_REUSE_RUNNER";
    private const string RealEsrganCompareThresholdEnvVar = "AIIMAGE_ESRGAN_COMPARE_THRESHOLD";
    private const string RealEsrganMaxElapsedMsEnvVar = "AIIMAGE_ESRGAN_MAX_ELAPSED_MS";
    private const string ClipInputDirEnvVar = "AIIMAGE_CLIP_INPUT_DIR";
    private const string ClipModelEnvVar = "AIIMAGE_CLIP_MODEL";
    private const string ClipEnableDumpEnvVar = "AIIMAGE_CLIP_ENABLE_DUMP";
    private const string ClipForceFullRtEnvVar = "AIIMAGE_CLIP_FORCE_FULL_RT";
    private const string ClipUseCommandBufferEnvVar = "AIIMAGE_CLIP_USE_COMMAND_BUFFER";
    private const string ClipUseAsyncComputeEnvVar = "AIIMAGE_CLIP_USE_ASYNC_COMPUTE";
    private const string ClipEnableGeneralTexConvEnvVar = "AIIMAGE_CLIP_ENABLE_GENERAL_TEX";
    private const string ClipEnableAttentionMatMulPack4EnvVar = "AIIMAGE_CLIP_ENABLE_ATTENTION_MATMUL_PACK4";
    private const string ClipPack4OnlyGuardEnvVar = "AIIMAGE_CLIP_PACK4_ONLY_GUARD";
    private const string ClipEnableLayerPathLogEnvVar = "AIIMAGE_CLIP_ENABLE_LAYER_PATH_LOG";
    private const string ClipLogAllLayerHeartbeatsEnvVar = "AIIMAGE_CLIP_LOG_ALL_LAYER_HEARTBEATS";
    private const string ClipLogAllLayerOutputsEnvVar = "AIIMAGE_CLIP_LOG_ALL_LAYER_OUTPUTS";
    private const string ClipLogAllBufferMaterializeEnvVar = "AIIMAGE_CLIP_LOG_ALL_BUFFER_MATERIALIZE";
    private const string ClipEnableLayerRuntimeProfileEnvVar = "AIIMAGE_CLIP_ENABLE_LAYER_RUNTIME_PROFILE";
    private const string ClipLayerRuntimeProfileSyncGpuEnvVar = "AIIMAGE_CLIP_LAYER_RUNTIME_PROFILE_SYNC_GPU";
    private const string CodeFormerEnableDumpEnvVar = "AIIMAGE_CODEFORMER_ENABLE_DUMP";
    private const string CodeFormerEnableFaceDumpEnvVar = "AIIMAGE_CODEFORMER_ENABLE_FACE_DUMP";
    private const string YoloFlipYEnvVar = "AIIMAGE_YOLOSEG_FLIPY";
    private const string YoloForceBufferConvEnvVar = "AIIMAGE_YOLOSEG_FORCE_BUFFER_CONV";
    private const string YoloForceBufferBinaryEnvVar = "AIIMAGE_YOLOSEG_FORCE_BUFFER_BINARY";
    private const string YoloUseArgbFloatEnvVar = "AIIMAGE_YOLOSEG_USE_ARGB_FLOAT";
    private const string YoloEnableDepthwiseTexConvEnvVar = "AIIMAGE_YOLOSEG_ENABLE_DEPTHWISE_TEX";
    private const string YoloEnableConv1x1TexConvEnvVar = "AIIMAGE_YOLOSEG_ENABLE_CONV1X1_TEX";
    private const string YoloEnableGeneralTexConvEnvVar = "AIIMAGE_YOLOSEG_ENABLE_GENERAL_TEX";
    private const string YoloEnableLayerPathLogEnvVar = "AIIMAGE_YOLOSEG_ENABLE_LAYER_PATH_LOG";
    private const string YoloLogAllLayerHeartbeatsEnvVar = "AIIMAGE_YOLOSEG_LOG_ALL_LAYER_HEARTBEATS";
    private const string YoloLogAllLayerOutputsEnvVar = "AIIMAGE_YOLOSEG_LOG_ALL_LAYER_OUTPUTS";
    private const string YoloLogAllBufferMaterializeEnvVar = "AIIMAGE_YOLOSEG_LOG_ALL_BUFFER_MATERIALIZE";
    private const string YoloPack4OnlyGuardEnvVar = "AIIMAGE_YOLOSEG_PACK4_ONLY_GUARD";
    private const string FacePack4OnlyGuardEnvVar = "AIIMAGE_FACE_PACK4_ONLY_GUARD";
    private const string MattingPack4OnlyGuardEnvVar = "AIIMAGE_MATTING_PACK4_ONLY_GUARD";
    private const string GfpganPack4OnlyGuardEnvVar = "AIIMAGE_GFPGAN_PACK4_ONLY_GUARD";
    private const string ReproTempPoolEnvVar = "AIIMAGE_REPRO_TEMP_POOL";
    private const string SdWidthEnvVar = "AIIMAGE_SD_WIDTH";
    private const string SdHeightEnvVar = "AIIMAGE_SD_HEIGHT";
    private const string SdStepsEnvVar = "AIIMAGE_SD_STEPS";
    private const string SdSeedEnvVar = "AIIMAGE_SD_SEED";
    private const string SdStrengthEnvVar = "AIIMAGE_SD_STRENGTH";
    private const string SdGuidanceScaleEnvVar = "AIIMAGE_SD_GUIDANCE_SCALE";
    private const string SdPositivePromptEnvVar = "AIIMAGE_SD_POSITIVE_PROMPT";
    private const string SdNegativePromptEnvVar = "AIIMAGE_SD_NEGATIVE_PROMPT";
    private const string SdInitImageEnvVar = "AIIMAGE_SD_INIT_IMAGE";
    private const string SdMaskImageEnvVar = "AIIMAGE_SD_MASK_IMAGE";
    private const string SdTensorFormatEnvVar = "AIIMAGE_SD_TENSOR_FORMAT";
    private const string SdDecoderTensorFormatEnvVar = "AIIMAGE_SD_DECODER_TENSOR_FORMAT";
    private const string SdEncoderTensorFormatEnvVar = "AIIMAGE_SD_ENCODER_TENSOR_FORMAT";
    private const string SdEnableDumpEnvVar = "AIIMAGE_SD_ENABLE_DUMP";
    private const string SdSyncStageTimingsEnvVar = "AIIMAGE_SD_SYNC_STAGE_TIMINGS";
    private const string SdKeepRawConvWeightsEnvVar = "AIIMAGE_SD_KEEP_RAW_CONV_WEIGHTS";
    private const string SdUseCommandBufferEnvVar = "AIIMAGE_SD_USE_COMMAND_BUFFER";
    private const string SdUseAsyncComputeEnvVar = "AIIMAGE_SD_USE_ASYNC_COMPUTE";
    private const string SdDisallowTempComputeBuffersEnvVar = "AIIMAGE_SD_DISALLOW_TEMP_COMPUTE_BUFFERS";
    private const string SdReplayBaselineDirEnvVar = "AIIMAGE_SD_REPLAY_BASELINE_DIR";
    private const string SdDirectBaselineDirEnvVar = "AIIMAGE_SD_DIRECT_BASELINE_DIR";
    private const string SdReplayReferenceDirEnvVar = "AIIMAGE_SD_REPLAY_REFERENCE_DIR";
    private const string SdReplayStartTopEnvVar = "AIIMAGE_SD_REPLAY_START_TOP";
    private const string SdReplayStopTopEnvVar = "AIIMAGE_SD_REPLAY_STOP_TOP";
    private const string SdReplayOutputBlobEnvVar = "AIIMAGE_SD_REPLAY_OUTPUT_BLOB";
    private const string SdReplayTargetBlobEnvVar = "AIIMAGE_SD_REPLAY_TARGET_BLOB";
    private const string SdReplayPromptKindEnvVar = "AIIMAGE_SD_REPLAY_PROMPT_KIND";
    private const string SdReplayInputBlobsEnvVar = "AIIMAGE_SD_REPLAY_INPUT_BLOBS";
    private const string MonaiBaselineManifestEnvVar = "AIIMAGE_MONAI_BASELINE_MANIFEST";
    private const string MonaiInputPathsEnvVar = "AIIMAGE_MONAI_INPUT_PATHS";
    private const string MonaiUseBaselineTensorEnvVar = "AIIMAGE_MONAI_USE_BASELINE_TENSOR";
    private const string MonaiOutputDirEnvVar = "AIIMAGE_MONAI_OUTPUT_DIR";
    private const string MonaiCaseNameEnvVar = "AIIMAGE_MONAI_CASE_NAME";
    private const string MonaiThresholdEnvVar = "AIIMAGE_MONAI_THRESHOLD";
    private const string MonaiCompareBaselineEnvVar = "AIIMAGE_MONAI_COMPARE_BASELINE";
    private const string MonaiEnableDumpEnvVar = "AIIMAGE_MONAI_ENABLE_DUMP";
    private const string MonaiDumpLargeTensorsEnvVar = "AIIMAGE_MONAI_DUMP_LARGE_TENSORS";
    private const string MonaiForceBufferConvEnvVar = "AIIMAGE_MONAI_FORCE_BUFFER_CONV";
    private const string MonaiForceBufferBinaryEnvVar = "AIIMAGE_MONAI_FORCE_BUFFER_BINARY";
    private const string MonaiForceCpuGemmEnvVar = "AIIMAGE_MONAI_FORCE_CPU_GEMM";
    private const string MonaiForceBufferAllEnvVar = "AIIMAGE_MONAI_FORCE_BUFFER_ALL";
    private const string MonaiForceBufferOutputsDims4EnvVar = "AIIMAGE_MONAI_FORCE_BUFFER_OUTPUTS_DIMS4";
    private const string MonaiForceBufferNamesEnvVar = "AIIMAGE_MONAI_FORCE_BUFFER_NAMES";
    private const string MonaiPatchInputModeEnvVar = "AIIMAGE_MONAI_PATCH_INPUT_MODE";
    private const string MonaiUseCommandBufferEnvVar = "AIIMAGE_MONAI_USE_COMMAND_BUFFER";
    private const string MonaiPack4OnlyGuardEnvVar = "AIIMAGE_MONAI_PACK4_ONLY_GUARD";
    private const string MonaiEnableAttentionMatMulPack4EnvVar = "AIIMAGE_MONAI_ENABLE_ATTENTION_MATMUL_PACK4";
    private const string MonaiKeepRawConvEnvVar = "AIIMAGE_MONAI_KEEP_RAW_CONV";
    private const string MonaiTensorFormatEnvVar = "AIIMAGE_MONAI_TENSOR_FORMAT";
    private const string MonaiNormalizeNonZeroEnvVar = "AIIMAGE_MONAI_NORMALIZE_NONZERO";
    private const string MonaiDebugPinnedBlobsEnvVar = "AIIMAGE_MONAI_DEBUG_PINNED_BLOBS";
    private const string MonaiOutputBlobEnvVar = "AIIMAGE_MONAI_OUTPUT_BLOB";
    private const string MonaiLogAllLayerHeartbeatsEnvVar = "AIIMAGE_MONAI_LOG_ALL_LAYER_HEARTBEATS";
    private const string MonaiLogAllLayerOutputsEnvVar = "AIIMAGE_MONAI_LOG_ALL_LAYER_OUTPUTS";
    private const string MonaiLogAllBufferMaterializeEnvVar = "AIIMAGE_MONAI_LOG_ALL_BUFFER_MATERIALIZE";
    private const string MonaiEnableLayerRuntimeProfileSyncGpuEnvVar = "AIIMAGE_MONAI_LAYER_RUNTIME_PROFILE_SYNC_GPU";
    private const string MonaiEnableConv3dTile3x3FastPathEnvVar = "AIIMAGE_MONAI_ENABLE_CONV3D_TILE3X3_FASTPATH";
    private const string MonaiEnableTimingSplitDiagnosticsEnvVar = "AIIMAGE_MONAI_ENABLE_TIMING_SPLIT_DIAGNOSTICS";
    private const string MonaiTimingSplitStopAfterBlobEnvVar = "AIIMAGE_MONAI_TIMING_SPLIT_STOP_AFTER_BLOB";
    private const string MonaiTimingSplitSyncAfterTopEnvVar = "AIIMAGE_MONAI_TIMING_SPLIT_SYNC_AFTER_TOP";
    private const string MonaiPack4SelfTestEnvVar = "AIIMAGE_MONAI_PACK4_SELFTEST";
    private const string MonaiPack4SelfTestFormatEnvVar = "AIIMAGE_MONAI_PACK4_SELFTEST_FORMAT";
    private const string MonaiProbeOnlyEnvVar = "AIIMAGE_MONAI_PROBE_ONLY";
    private const string MonaiMaxPatchesEnvVar = "AIIMAGE_MONAI_MAX_PATCHES";
    private const string MonaiProbePatchOrdinalEnvVar = "AIIMAGE_MONAI_PROBE_PATCH_ORDINAL";
    private const string MonaiClearTempPoolEachPatchEnvVar = "AIIMAGE_MONAI_CLEAR_TEMP_POOL_EACH_PATCH";
    private const string MonaiTempPoolClearIntervalEnvVar = "AIIMAGE_MONAI_TEMP_POOL_CLEAR_INTERVAL";
    private const string MonaiYieldIntervalEnvVar = "AIIMAGE_MONAI_YIELD_INTERVAL";
    private const string MonaiManagedCleanupIntervalEnvVar = "AIIMAGE_MONAI_MANAGED_CLEANUP_INTERVAL";
    private const string MonaiResourceSnapshotIntervalEnvVar = "AIIMAGE_MONAI_RESOURCE_SNAPSHOT_INTERVAL";
    private const string MonaiAbortPrivateMemoryMbEnvVar = "AIIMAGE_MONAI_ABORT_PRIVATE_MEMORY_MB";
    private const string BatchTimeoutMinutesEnvVar = "AIIMAGE_BATCH_TIMEOUT_MINUTES";
    private const string BatchMethodEnvVar = "AIIMAGE_BATCH_METHOD";
    private const string DesignViewDebugInputEnvVar = "AIIMAGE_DESIGNVIEW_DEBUG_INPUT";
    private static readonly MethodInfo EditorUpdatePumpMethod = typeof(EditorApplication).GetMethod("Internal_CallUpdateFunctions", BindingFlags.Static | BindingFlags.NonPublic);
    private static readonly MethodInfo EditorDelayPumpMethod = typeof(EditorApplication).GetMethod("Internal_CallDelayFunctions", BindingFlags.Static | BindingFlags.NonPublic);
    private static readonly string DefaultFaceDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "Pa070111a.jpg");
    private static readonly string DefaultCodeFormerDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "Pa070111a.jpg");
    private static readonly string DefaultClipDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "Pa070111a.jpg");
    private static readonly string DefaultMattingDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "ncnn_matting-main", "test_img.jpg");
    private static readonly string DefaultMattingReferencePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "ncnn_matting-main", "test_result.jpg");
    private static readonly string DefaultYoloSegDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "P1120028.jpg");
    private static readonly string DefaultReproStressImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "CodeFormer-ncnn-main", "data", "02.png");
    private static readonly string DefaultMonaiBaselineManifestPath = Path.Combine(Directory.GetCurrentDirectory(), "Tools", "MonaiToNCNN", "manual_test", "brats_mri_segmentation_baseline", "RegLib_C01_1", "baseline_manifest.json");
    private static readonly string DefaultMonaiInputPath = @"E:\Projects\CTData\sliceexampledata2\MRBrainTumor1\RegLib_C01_1.nrrd";
    private static readonly string DefaultVistaBaselineManifestPath = Path.Combine(Directory.GetCurrentDirectory(), "Tools", "MonaiToNCNN", "manual_test", "vista3d_ct_philips_heart_baseline", "ct_philips_heart", "baseline_manifest.json");
    private static readonly string DefaultVistaInputPath = @"E:\Projects\CTData\sliceexampledata2\CT_Philips\CT_Philips.nii.gz";
    private static readonly string DefaultRealEsrganInputDir = Path.Combine(Directory.GetCurrentDirectory(), "Documents", "ClipCompareInput");
    private static readonly string DefaultSdPositivePrompt = "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";
    private static readonly string DefaultSdNegativePrompt = "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";
    private static bool _autoBatchScheduled;

    [InitializeOnLoadMethod]
    private static void AutoStartBatchMethodFromEnv()
    {
        if (_autoBatchScheduled || !Application.isBatchMode)
            return;

        var methodName = Environment.GetEnvironmentVariable(BatchMethodEnvVar);
        if (string.IsNullOrWhiteSpace(methodName))
            return;

        _autoBatchScheduled = true;
        EditorApplication.delayCall += () =>
        {
            try
            {
                RunBatchMethodByName(methodName.Trim());
            }
            catch (Exception e)
            {
                Debug.LogError("[NcnnDebugRunner] Auto batch dispatch failed: " + e.Message);
                Debug.LogException(e);
                EditorApplication.Exit(1);
            }
        };
    }

    private static void RunBatchMethodByName(string methodName)
    {
        switch (methodName)
        {
            case nameof(RunFaceDebugBatch):
                RunFaceDebugBatch();
                return;
            case nameof(RunCodeFormerDebugBatch):
                RunCodeFormerDebugBatch();
                return;
            case nameof(RunClipDebugBatch):
                RunClipDebugBatch();
                return;
            case nameof(RunClipDirectoryDebugBatch):
                RunClipDirectoryDebugBatch();
                return;
            case nameof(RunGfpganDebugBatch):
                RunGfpganDebugBatch();
                return;
            case nameof(RunMattingDebugBatch):
                RunMattingDebugBatch();
                return;
            case nameof(RunYoloSegDebugBatch):
                RunYoloSegDebugBatch();
                return;
            case nameof(RunYoloAndInpaintingDebugBatch):
                RunYoloAndInpaintingDebugBatch();
                return;
            case nameof(RunYoloAndInpaintingProbeBatch):
                RunYoloAndInpaintingProbeBatch();
                return;
            case nameof(RunStableDiffusionDebugBatch):
                RunStableDiffusionDebugBatch();
                return;
            case nameof(RunStableDiffusionBaselineDebugBatch):
                RunStableDiffusionBaselineDebugBatch();
                return;
            case nameof(RunCodeFormerStressBatch):
                RunCodeFormerStressBatch();
                return;
            case nameof(RunReproSuiteStressBatch):
                RunReproSuiteStressBatch();
                return;
            case nameof(RunRealEsrganValidationBatch):
                RunRealEsrganValidationBatch();
                return;
            case nameof(RunMonaiDebugBatch):
                RunMonaiDebugBatch();
                return;
            case nameof(RunDesignViewCompositeDebugBatch):
                RunDesignViewCompositeDebugBatch();
                return;
            default:
                throw new InvalidOperationException("Unknown batch method: " + methodName);
        }
    }

    [MenuItem("Tools/AIImage/Run NCNN Face Debug")]
    public static void RunFaceDebugMenu()
    {
        RunFaceDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run CodeFormer Debug")]
    public static void RunCodeFormerDebugMenu()
    {
        RunCodeFormerDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run CLIP Debug")]
    public static void RunClipDebugMenu()
    {
        RunClipDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run CLIP Directory Debug")]
    public static void RunClipDirectoryDebugMenu()
    {
        RunClipDirectoryDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run GFPGAN Debug")]
    public static void RunGfpganDebugMenu()
    {
        RunGfpganDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run Matting Debug")]
    public static void RunMattingDebugMenu()
    {
        RunMattingDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run YOLO Seg Debug")]
    public static void RunYoloSegDebugMenu()
    {
        RunYoloSegDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run YOLO + SD Inpainting Debug")]
    public static void RunYoloAndInpaintingDebugMenu()
    {
        RunYoloAndInpaintingDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run Stable Diffusion Debug")]
    public static void RunStableDiffusionDebugMenu()
    {
        RunStableDiffusionDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run MONAI Debug")]
    public static void RunMonaiDebugMenu()
    {
        RunMonaiDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run DesignView Composite Debug")]
    public static void RunDesignViewCompositeDebugMenu()
    {
        RunDesignViewCompositeDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run CodeFormer Stress (60x)")]
    public static void RunCodeFormerStressMenu()
    {
        RunCodeFormerStressBatch();
    }

    [MenuItem("Tools/AIImage/Run Repro Suite Stress (02.png)")]
    public static void RunReproSuiteStressMenu()
    {
        RunReproSuiteStressBatch();
    }

    [MenuItem("Tools/AIImage/Run RealESRGAN Validation Batch")]
    public static void RunRealEsrganValidationMenu()
    {
        RunRealEsrganValidationBatch();
    }

    public static async UniTaskVoid RunFaceDebug()
    {
        var inputPath = ResolveInputPath(DefaultFaceDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("NcnnFaceDebugRunner");
        try
        {
            var face = go.AddComponent<NcnnFaceRegionGenerator>();
            face.enableNcnnFaceRegion = true;
            face.preferTexturePathForFaceDetector = ResolveFacePreferTexturePath();
            ApplyFacePack4GuardFromEnv(face);
            ApplyFaceThresholdOverrides(face);
            face.enableDetailedProposalDump = true;
            face.autoOpenDumpDir = false;
            var result = await face.GenerateAsync(tex, true, CancellationToken.None);
            Debug.Log("NCNN Face Debug result | error=" + (result.error ?? "") + " | dump=" + (result.dumpDir ?? ""));
            if (result.mask != null)
                UnityEngine.Object.DestroyImmediate(result.mask);
        }
        finally
        {
            RenderTexture.active = null;
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    public static void RunFaceDebugBatch() => RunBatchBlocking(nameof(RunFaceDebugBatch), RunFaceDebugInternal);

    public static async UniTaskVoid RunCodeFormerDebug()
    {
        var inputPath = ResolveInputPath(DefaultCodeFormerDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("CodeFormerDebugRunner");
        try
        {
            var runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
            runner.enableDebugDump = true;
            runner.enableFaceRegionDebugDump = true;
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[CodeFormer Progress] " + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("CodeFormer Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs + " | dump=" + (runner.LastDumpDir ?? ""));
            if (result.texture != null)
            {
                TryWriteTexturePng(result.texture, runner.LastDumpDir, "17_full_output.png");
                UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    public static async UniTaskVoid RunClipDebug()
    {
        await RunClipDebugInternal();
    }

    public static async UniTaskVoid RunClipDirectoryDebug()
    {
        await RunClipDirectoryDebugInternal();
    }

    private static async UniTask RunClipDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultClipDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("ClipDebugRunner");
        try
        {
            var runner = go.AddComponent<ClipNcnnReproRunner>();
            runner.enableTempPool = false;
            runner.maxPooledPerShape = 0;
            runner.modelLevel = ResolveClipModelLevel();
            ConfigureClipRunnerFromEnv(runner, defaultEnableDebugDump: true);
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[CLIP-DEBUG] progress=" + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("CLIP Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs + " | best=" + (result.bestLabel ?? "") + " | prob=" + result.bestProbability.ToString("0.000000", CultureInfo.InvariantCulture) + " | dump=" + (runner.LastDumpDir ?? ""));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    public static async UniTaskVoid RunGfpganDebug()
    {
        var inputPath = ResolveInputPath(DefaultCodeFormerDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("GfpganDebugRunner");
        try
        {
            var runner = go.AddComponent<GfpganNcnnReproRunner>();
            runner.enableFaceRegionDebugDump = true;
            ApplyGfpganPack4GuardFromEnv(runner);
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[GFPGAN Progress] " + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("GFPGAN Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs);
            if (result.texture != null)
            {
                var dir = CreateGenericDumpDir("AIImage_GfpganRepro");
                TryWriteTexturePng(result.texture, dir, "17_full_output.png");
                UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    public static async UniTaskVoid RunMattingDebug()
    {
        await RunMattingDebugInternal();
    }

    public static async UniTaskVoid RunYoloSegDebug()
    {
        await RunYoloSegDebugInternal();
    }

    public static async UniTaskVoid RunYoloAndInpaintingDebug()
    {
        await RunYoloAndInpaintingDebugInternal();
    }

    public static async UniTaskVoid RunStableDiffusionDebug()
    {
        await RunStableDiffusionDebugInternal();
    }

    public static async UniTaskVoid RunMonaiDebug()
    {
        await RunMonaiDebugInternal();
    }

    public static async UniTaskVoid RunDesignViewCompositeDebug()
    {
        await RunDesignViewCompositeDebugInternal();
    }

    public static async UniTaskVoid RunVista3dDebug()
    {
        await RunMonaiDebugInternal(DefaultVistaBaselineManifestPath, new[] { DefaultVistaInputPath }, true);
    }

    public static void RunMattingDebugBatch() => RunBatchBlocking(nameof(RunMattingDebugBatch), RunMattingDebugInternal);

    public static void RunYoloSegDebugBatch() => RunBatchBlocking(nameof(RunYoloSegDebugBatch), RunYoloSegDebugInternal);

    public static void RunYoloAndInpaintingDebugBatch() => RunBatchBlocking(nameof(RunYoloAndInpaintingDebugBatch), RunYoloAndInpaintingDebugInternal, TimeSpan.FromHours(4));

    public static void RunStableDiffusionBaselineDebugBatch() => RunBatchBlocking(nameof(RunStableDiffusionBaselineDebugBatch), RunStableDiffusionBaselineDebugInternal, TimeSpan.FromHours(4));

    public static void RunYoloAndInpaintingProbeBatch() => RunBatchBlocking(nameof(RunYoloAndInpaintingProbeBatch), RunYoloAndInpaintingProbeInternal, TimeSpan.FromHours(6));

    public static async void RunYoloSegDebugBatchLegacy()
    {
        try
        {
            Debug.Log("[NcnnDebugRunner] RunYoloSegDebugBatchLegacy start");
            await RunYoloSegDebugInternal();
            Debug.Log("[NcnnDebugRunner] RunYoloSegDebugBatchLegacy done");
            EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.Log("[NcnnDebugRunner] RunYoloSegDebugBatchLegacy failed: " + e.Message);
            Debug.LogException(e);
            EditorApplication.Exit(1);
        }
    }

    public static void RunStableDiffusionDebugBatch() => RunBatchBlocking(nameof(RunStableDiffusionDebugBatch), RunStableDiffusionDebugInternal, TimeSpan.FromHours(4));

    public static void RunRealEsrganValidationBatch() => RunBatchBlocking(nameof(RunRealEsrganValidationBatch), RunRealEsrganValidationInternal, TimeSpan.FromMinutes(20));

    public static void RunDesignViewCompositeDebugBatch() => RunBatchBlocking(nameof(RunDesignViewCompositeDebugBatch), RunDesignViewCompositeDebugInternal, TimeSpan.FromMinutes(20));

    private static async UniTask RunMattingDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultMattingDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("MattingDebugRunner");
        try
        {
            var runner = go.AddComponent<MatterNcnnReproRunner>();
            runner.enableDebugDump = true;
            runner.forceBufferConvolution = false;
            var mattingPack4OnlyGuard = ResolveBoolEnv(MattingPack4OnlyGuardEnvVar, false);
            runner.disallowBufferAccess = mattingPack4OnlyGuard;
            runner.disallowBufferOutputs = mattingPack4OnlyGuard;
            runner.disallowBufferToTextureMaterialization = mattingPack4OnlyGuard;
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("Matting Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs + " | dump=" + (runner.LastDumpDir ?? ""));
            if (!string.IsNullOrWhiteSpace(result.error))
                return;

            var dir = CreateGenericDumpDir("AIImage_MattingRepro");
            TryWriteTexturePng(result.texture, dir, "17_composite.png");
            TryWriteTexturePng(result.matte, dir, "18_matte.png");
            TryCompareTextureWithReference(result.texture, DefaultMattingReferencePath);

            if (result.texture != null)
                UnityEngine.Object.DestroyImmediate(result.texture);
            if (result.matte != null)
                UnityEngine.Object.DestroyImmediate(result.matte);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    private static async UniTask RunYoloSegDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultYoloSegDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("YoloSegDebugRunner");
        try
        {
            var runner = go.AddComponent<YoloSegNcnnReproRunner>();
            runner.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
            runner.enableDebugDump = true;
            runner.forceBufferConvolution = ResolveBoolEnv(YoloForceBufferConvEnvVar, runner.forceBufferConvolution);
            runner.forceBufferBinaryOp = ResolveBoolEnv(YoloForceBufferBinaryEnvVar, true);
            runner.useArgbFloatTensor = ResolveBoolEnv(YoloUseArgbFloatEnvVar, true);
            runner.enableGeneralTextureConvolution = ResolveBoolEnv(YoloEnableGeneralTexConvEnvVar, runner.enableGeneralTextureConvolution);
            runner.enableDepthWiseTextureConvolution = ResolveBoolEnv(YoloEnableDepthwiseTexConvEnvVar, runner.enableDepthWiseTextureConvolution);
            runner.enableConv1x1TextureConvolution = ResolveBoolEnv(YoloEnableConv1x1TexConvEnvVar, runner.enableConv1x1TextureConvolution);
            runner.enableLayerPathDebugLog = ResolveBoolEnv(YoloEnableLayerPathLogEnvVar, false);
            runner.logAllLayerHeartbeats = ResolveBoolEnv(YoloLogAllLayerHeartbeatsEnvVar, false);
            runner.logAllLayerOutputs = ResolveBoolEnv(YoloLogAllLayerOutputsEnvVar, false);
            runner.logAllBufferMaterialize = ResolveBoolEnv(YoloLogAllBufferMaterializeEnvVar, false);
            var yoloPack4OnlyGuard = ResolveBoolEnv(YoloPack4OnlyGuardEnvVar, false);
            runner.disallowBufferAccess = yoloPack4OnlyGuard;
            runner.disallowBufferOutputs = yoloPack4OnlyGuard;
            runner.disallowBufferToTextureMaterialization = yoloPack4OnlyGuard;
            runner.targetPersonOnly = true;
            runner.flipYInput = ResolveBoolEnv(YoloFlipYEnvVar, runner.flipYInput);
            runner.enableMaskClose = true;
            runner.enableMaskDilate = true;
            runner.ProgressChanged += (value, message) =>
            {
                Debug.Log("[YoloSeg-DEBUG] progress=" + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            };

            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log(
                "YOLO Seg Debug result | error=" + (result.error ?? "")
                + " | elapsedMs=" + result.elapsedMs
                + " | personCount=" + result.personCount.ToString(CultureInfo.InvariantCulture)
                + " | coverage=" + result.maskCoverage01.ToString("0.000000", CultureInfo.InvariantCulture)
                + " | dump=" + (runner.LastDumpDir ?? ""));

            if (!string.IsNullOrWhiteSpace(result.error))
                return;

            var dir = !string.IsNullOrWhiteSpace(runner.LastDumpDir)
                ? runner.LastDumpDir
                : CreateGenericDumpDir("AIImage_YoloSegRepro");
            TryWriteTexturePng(result.mask, dir, "01_person_mask.png");
            TryWriteTexturePng(result.texture, dir, "02_transparent_cutout.png");
            TryWriteTexturePng(result.overlay, dir, "03_overlay.png");
            if (!string.IsNullOrWhiteSpace(runner.LastSummaryText))
                File.WriteAllText(Path.Combine(dir, "summary.txt"), runner.LastSummaryText);

            if (result.texture != null)
                UnityEngine.Object.DestroyImmediate(result.texture);
            if (result.mask != null)
                UnityEngine.Object.DestroyImmediate(result.mask);
            if (result.overlay != null)
                UnityEngine.Object.DestroyImmediate(result.overlay);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    private static async UniTask RunMonaiDebugInternal(
        string baselineManifestOverride = null,
        string[] inputPathsOverride = null,
        bool? useBaselineTensorOverride = null)
    {
        if (ResolveBoolEnv(MonaiPack4SelfTestEnvVar, false))
        {
            await RunMonaiPack4SelfTestInternal();
            return;
        }

        var baselineManifestPath = !string.IsNullOrWhiteSpace(baselineManifestOverride)
            ? baselineManifestOverride
            : (ResolveOptionalExistingFile(MonaiBaselineManifestEnvVar) ?? DefaultMonaiBaselineManifestPath);
        var useBaselineTensor = useBaselineTensorOverride ?? ResolveBoolEnv(MonaiUseBaselineTensorEnvVar, true);
        var inputPaths = inputPathsOverride != null && inputPathsOverride.Length > 0
            ? inputPathsOverride
            : ResolveMonaiInputPathsForRun(baselineManifestPath, useBaselineTensor);
        var outputDir = ResolveStringEnv(MonaiOutputDirEnvVar, null);
        var caseName = ResolveStringEnv(MonaiCaseNameEnvVar, null);
        var lowPowerMode = MonaiLowPowerModeState.IsEnabled;
        var compareBaseline = ResolveBoolEnv(MonaiCompareBaselineEnvVar, true);
        var enableDump = ResolveBoolEnv(MonaiEnableDumpEnvVar, true);
        var dumpLargeTensors = ResolveBoolEnv(MonaiDumpLargeTensorsEnvVar, true);
        var forceBufferConv = ResolveBoolEnv(MonaiForceBufferConvEnvVar, false);
        var forceBufferBinary = ResolveBoolEnv(MonaiForceBufferBinaryEnvVar, false);
        var forceCpuGemm = ResolveBoolEnv(MonaiForceCpuGemmEnvVar, false);
        var forceBufferAll = ResolveBoolEnv(MonaiForceBufferAllEnvVar, false);
        var forceBufferOutputsDims4 = ResolveBoolEnv(MonaiForceBufferOutputsDims4EnvVar, false);
        var forceBufferNames = ResolveTokenSetEnv(MonaiForceBufferNamesEnvVar);
        var patchInputMode = ResolveStringEnv(MonaiPatchInputModeEnvVar, null);
        var useCommandBuffer = ResolveBoolEnv(MonaiUseCommandBufferEnvVar, false)
            || IsMonaiCommandBufferPatchInputMode(patchInputMode);
        var enableAttentionMatMulPack4 = ResolveBoolEnv(MonaiEnableAttentionMatMulPack4EnvVar, false);
        var keepRawConv = ResolveBoolEnv(MonaiKeepRawConvEnvVar, true);
        var tensorTextureFormat = ResolveRenderTextureFormatEnv(MonaiTensorFormatEnvVar, RenderTextureFormat.ARGBHalf);
        var normalizeNonZeroOverride = ResolveOptionalBoolEnv(MonaiNormalizeNonZeroEnvVar);
        var normalizeNonZero = normalizeNonZeroOverride ?? true;
        var debugPinnedBlobsCsv = ResolveStringEnv(MonaiDebugPinnedBlobsEnvVar, string.Empty);
        var outputBlobName = ResolveStringEnv(MonaiOutputBlobEnvVar, null);
        var logAllLayerHeartbeats = ResolveBoolEnv(MonaiLogAllLayerHeartbeatsEnvVar, false);
        var logAllLayerOutputs = ResolveBoolEnv(MonaiLogAllLayerOutputsEnvVar, false);
        var logAllBufferMaterialize = ResolveBoolEnv(MonaiLogAllBufferMaterializeEnvVar, false);
        var layerRuntimeProfileSyncGpu = ResolveBoolEnv(MonaiEnableLayerRuntimeProfileSyncGpuEnvVar, false);
        var enableConv3dTile3x3FastPath = ResolveBoolEnv(MonaiEnableConv3dTile3x3FastPathEnvVar, false);
        var enableTimingSplitDiagnostics = ResolveBoolEnv(MonaiEnableTimingSplitDiagnosticsEnvVar, false);
        var timingSplitStopAfterBlob = ResolveStringEnv(MonaiTimingSplitStopAfterBlobEnvVar, string.Empty);
        var timingSplitSyncAfterTop = ResolveStringEnv(MonaiTimingSplitSyncAfterTopEnvVar, string.Empty);
        var probeOnly = ResolveBoolEnv(MonaiProbeOnlyEnvVar, false);
        var maxPatchCount = ResolvePositiveIntEnvAllowZero(MonaiMaxPatchesEnvVar, probeOnly ? 1 : 0);
        var probePatchOrdinal = ResolvePositiveIntEnvAllowZero(MonaiProbePatchOrdinalEnvVar, 0);
        var threshold = ResolveFloatEnvOrDefault(MonaiThresholdEnvVar, 0.5f);
        var monaiConfig = ResolveMonaiModelConfig(baselineManifestPath);
        ushort binaryForegroundLabelValue = 0;
        try
        {
            if (!string.IsNullOrWhiteSpace(baselineManifestPath) && File.Exists(baselineManifestPath))
            {
                var baseline = JObject.Parse(File.ReadAllText(baselineManifestPath));
                binaryForegroundLabelValue = (ushort?)baseline["prompt"]?["foreground_label_value"]?.Value<int>() ?? (ushort)0;
            }
        }
        catch
        {
        }

        var go = new GameObject("MonaiDebugRunner");
        NcnnCompute.NcnnGpuResourceTracker.Enabled = true;
        NcnnCompute.NcnnGpuResourceTracker.Reset("NcnnDebugRunner.MONAI");
        string monaiOutputDir = null;
        try
        {
            var runner = go.AddComponent<MONAINcnnReproRunner>();
            if (!string.IsNullOrWhiteSpace(monaiConfig.modelParamPath))
                runner.modelParamRelativePath = monaiConfig.modelParamPath;
            if (!string.IsNullOrWhiteSpace(monaiConfig.modelBinPath))
                runner.modelBinRelativePath = monaiConfig.modelBinPath;
            if (!string.IsNullOrWhiteSpace(monaiConfig.pnnxParamPath))
                runner.pnnxParamRelativePath = monaiConfig.pnnxParamPath;
            if (!string.IsNullOrWhiteSpace(monaiConfig.bundleManifestPath))
                runner.bundleManifestRelativePath = monaiConfig.bundleManifestPath;
            runner.defaultPostprocessKind = monaiConfig.postprocessKind;
            runner.enableDebugDump = enableDump;
            runner.dumpLargeTensorFiles = dumpLargeTensors;
            runner.enableBaselineCompare = compareBaseline;
            runner.forceBufferConvolution = forceBufferConv;
            runner.forceBufferBinaryOp = forceBufferBinary;
            runner.forceCpuGemm = forceCpuGemm;
            runner.forceBufferAllLayers = forceBufferAll;
            runner.forceBufferOutputsForDims4 = forceBufferOutputsDims4;
            runner.useTextureInputForMonaiPatches = ResolveMonaiPatchInputMode(forceBufferAll, patchInputMode);
            runner.useCommandBufferForMonaiPatches = useCommandBuffer;
            if (runner.useCommandBufferForMonaiPatches)
                runner.useTextureInputForMonaiPatches = true;
            runner.enableAttentionMatMulPack4Specializations = enableAttentionMatMulPack4;
            runner.keepRawConvWeightsForTexturePath = keepRawConv;
            runner.tensorTextureFormat = tensorTextureFormat;
            runner.debugPinnedBlobNamesCsv = debugPinnedBlobsCsv;
            runner.enableTempPool = ResolveBoolEnv(ReproTempPoolEnvVar, false);
            runner.maxPooledPerShape = runner.enableTempPool ? 1 : 0;
            runner.clearTempPoolAfterEachSlidingWindowPatch = ResolveBoolEnv(MonaiClearTempPoolEachPatchEnvVar, true);
            runner.slidingWindowTempPoolClearInterval = ResolvePositiveIntEnvAllowZero(MonaiTempPoolClearIntervalEnvVar, 1);
            runner.slidingWindowYieldInterval = ResolvePositiveIntEnvAllowZero(MonaiYieldIntervalEnvVar, 1);
            runner.slidingWindowManagedCleanupInterval = ResolvePositiveIntEnvAllowZero(MonaiManagedCleanupIntervalEnvVar, 1);
            runner.slidingWindowResourceSnapshotInterval = ResolvePositiveIntEnvAllowZero(MonaiResourceSnapshotIntervalEnvVar, 1);
            runner.slidingWindowAbortIfPrivateMemoryExceedsMb = Mathf.Max(0f, ResolveFloatEnvOrDefault(MonaiAbortPrivateMemoryMbEnvVar, 8192f));
            runner.featureHeadChunkDepth = lowPowerMode ? 4 : 8;
            runner.logAllLayerHeartbeats = logAllLayerHeartbeats;
            runner.logAllLayerOutputs = logAllLayerOutputs;
            runner.logAllBufferMaterialize = logAllBufferMaterialize;
            runner.enableLayerRuntimeProfile = true;
            runner.syncLayerRuntimeProfile = layerRuntimeProfileSyncGpu;
            runner.enableConv3dTile3x3Pack4FastPath = enableConv3dTile3x3FastPath;
            runner.enableTimingSplitDiagnostics = enableTimingSplitDiagnostics;
            runner.timingSplitStopAfterBlobName = timingSplitStopAfterBlob;
            runner.timingSplitSyncAfterTopName = timingSplitSyncAfterTop;
            if (runner.useTextureInputForMonaiPatches)
                runner.forceBufferLayerNames = forceBufferNames;
            if (ResolveBoolEnv(MonaiPack4OnlyGuardEnvVar, false))
            {
                runner.useTextureInputForMonaiPatches = true;
                runner.enableAttentionMatMulPack4Specializations = true;
                runner.disallowBufferAccess = true;
                runner.disallowBufferOutputs = true;
                runner.disallowBufferToTextureMaterialization = true;
                if (!EnvironmentVariableExists(MonaiKeepRawConvEnvVar))
                    runner.keepRawConvWeightsForTexturePath = false;
            }
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[MONAI Progress] " + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            Debug.Log(
                "[MONAI Debug] low_power_mode=" + lowPowerMode
                + " | feature_head_chunk_depth=" + runner.featureHeadChunkDepth.ToString(CultureInfo.InvariantCulture));

            var request = new MonaiRunRequest
            {
                inputSource = useBaselineTensor ? MonaiInputSourceKind.BaselineTensorDump : MonaiInputSourceKind.MedicalVolumeFiles,
                baselineManifestPath = baselineManifestPath,
                inputVolumePaths = inputPaths,
                caseName = caseName,
                outputDir = outputDir,
                outputBlobName = string.IsNullOrWhiteSpace(outputBlobName) ? null : outputBlobName.Trim(),
                threshold = threshold,
                normalizeNonZero = normalizeNonZero,
                normalizeNonZeroOverride = normalizeNonZeroOverride,
                compareWithBaseline = compareBaseline,
                probeOnly = probeOnly,
                maxSlidingWindowPatches = maxPatchCount,
                probePatchOrdinal = probePatchOrdinal,
                binaryForegroundLabelValue = binaryForegroundLabelValue,
                debugPinnedBlobNames = string.IsNullOrWhiteSpace(debugPinnedBlobsCsv)
                    ? null
                    : debugPinnedBlobsCsv.Split(new[] { ',', ';', '\r', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries),
                postprocessKind = monaiConfig.postprocessKind,
                channelFillMode = MonaiChannelFillMode.DuplicateFirst
            };

            var result = await runner.ProcessAsync(request, CancellationToken.None);
            Debug.Log(
                "MONAI Debug result | error=" + (result.error ?? "")
                + " | elapsedMs=" + result.elapsedMs
                + " | case=" + (result.caseName ?? "")
                + " | dump=" + (result.outputDir ?? ""));
        }
        finally
        {
            try { LogResourceSnapshot("monai_before_destroy"); } catch { }
            try
            {
                monaiOutputDir = ResolveStringEnv(MonaiOutputDirEnvVar, null);
                if (string.IsNullOrWhiteSpace(monaiOutputDir))
                    monaiOutputDir = FindLatestMonaiOutputDir();
                if (!string.IsNullOrWhiteSpace(monaiOutputDir))
                    NcnnCompute.NcnnGpuResourceTracker.WriteReport(monaiOutputDir, "gpu_resource_stats.txt");
            }
            catch
            {
            }
            UnityEngine.Object.DestroyImmediate(go);
            await ReleaseGpuPressureAsync();
            try { LogResourceSnapshot("monai_after_release"); } catch { }
            try
            {
                if (!string.IsNullOrWhiteSpace(monaiOutputDir))
                    NcnnCompute.NcnnGpuResourceTracker.WriteReport(monaiOutputDir, "gpu_resource_stats_after_release.txt");
            }
            catch
            {
            }
            NcnnCompute.NcnnGpuResourceTracker.Enabled = false;
        }
    }

    private static async UniTask RunDesignViewCompositeDebugInternal()
    {
        var inputPath = Environment.GetEnvironmentVariable(DesignViewDebugInputEnvVar);
        if (string.IsNullOrWhiteSpace(inputPath))
            inputPath = @"D:\photos\2012-10国庆\10-14天水麦积山\DSCF1827.JPG";
        inputPath = inputPath.Trim().Trim('"');

        var source = LoadTexture(inputPath);
        if (source == null)
            throw new InvalidOperationException("Failed to load DesignView debug input: " + inputPath);

        var outputDir = CreateGenericDumpDir("AIImage_DesignViewComposite");
        TryWriteTexturePng(source, outputDir, "00_input.png");

        var hostGo = new GameObject("DesignViewCompositeDebugRunner");
        try
        {
            Debug.Log("[DesignViewCompositeDebug] begin | input=" + inputPath + " | output_dir=" + outputDir);
            var yolo = hostGo.AddComponent<YoloSegNcnnReproRunner>();
            yolo.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
            yolo.enableDebugDump = true;
            yolo.targetPersonOnly = false;
            yolo.enableMaskClose = true;
            yolo.enableMaskDilate = true;
            yolo.forceBufferBinaryOp = ResolveBoolEnv(YoloForceBufferBinaryEnvVar, true);
            yolo.useArgbFloatTensor = ResolveBoolEnv(YoloUseArgbFloatEnvVar, true);
            yolo.forceBufferConvolution = ResolveBoolEnv(YoloForceBufferConvEnvVar, yolo.forceBufferConvolution);
            yolo.enableGeneralTextureConvolution = ResolveBoolEnv(YoloEnableGeneralTexConvEnvVar, yolo.enableGeneralTextureConvolution);
            yolo.enableDepthWiseTextureConvolution = ResolveBoolEnv(YoloEnableDepthwiseTexConvEnvVar, yolo.enableDepthWiseTextureConvolution);
            yolo.enableConv1x1TextureConvolution = ResolveBoolEnv(YoloEnableConv1x1TexConvEnvVar, yolo.enableConv1x1TextureConvolution);
            yolo.flipYInput = ResolveBoolEnv(YoloFlipYEnvVar, yolo.flipYInput);

            var result = await yolo.ProcessAsync(source, CancellationToken.None);
            Debug.Log(
                "[DesignViewCompositeDebug] yolo_result"
                + " | error=" + (result.error ?? string.Empty)
                + " | detections=" + (result.detections != null ? result.detections.Length.ToString(CultureInfo.InvariantCulture) : "0")
                + " | coverage=" + result.maskCoverage01.ToString("0.000000", CultureInfo.InvariantCulture)
                + " | yolo_dump=" + (yolo.LastDumpDir ?? string.Empty));

            if (!string.IsNullOrWhiteSpace(result.error))
                throw new InvalidOperationException("DesignView composite debug YOLO failed: " + result.error);
            if (result.mask == null)
                throw new InvalidOperationException("DesignView composite debug YOLO returned null mask.");

            TryWriteTexturePng(result.mask, outputDir, "01_person_mask.png");
            TryWriteTexturePng(result.texture, outputDir, "02_transparent_cutout.png");
            TryWriteTexturePng(result.overlay, outputDir, "03_overlay.png");
            Debug.Log("[DesignViewCompositeDebug] wrote_yolo_outputs | output_dir=" + outputDir);

            var designViewType = typeof(DesignView);
            var hostType = typeof(AIImagePageHost);

            var createWorkingRt = designViewType.GetMethod("CreateWorkingRenderTexture", BindingFlags.Static | BindingFlags.NonPublic);
            var buildMaskedBackground = designViewType.GetMethod("BuildMaskedBackgroundRenderTextureAsync", BindingFlags.Instance | BindingFlags.NonPublic);
            var buildHoleMask = designViewType.GetMethod("BuildHoleMaskRenderTextureAsync", BindingFlags.Instance | BindingFlags.NonPublic);
            var buildLayerCutout = designViewType.GetMethod("BuildLayerCutoutRenderTexture", BindingFlags.Instance | BindingFlags.NonPublic);
            var toDisplayRect = designViewType.GetMethod("ToDisplayPixelRect", BindingFlags.Static | BindingFlags.NonPublic);
            var toTextureRect = designViewType.GetMethod("ToTexturePixelRect", BindingFlags.Static | BindingFlags.NonPublic);
            var buildAppliedComposite = designViewType.GetMethod("BuildAppliedCompositeAsync", BindingFlags.Instance | BindingFlags.NonPublic);
            var dumpTextureAsync = designViewType.GetMethod("DumpTextureAsync", BindingFlags.Instance | BindingFlags.NonPublic);
            var readbackTextureAsync = typeof(BasePageView).GetMethod("ReadbackTextureAsync", BindingFlags.Instance | BindingFlags.NonPublic);
            var computeTextureDiff = typeof(NcnnDebugRunner).GetMethod("ComputeTextureDiff", BindingFlags.Static | BindingFlags.NonPublic);
            var imageProcessingProp = hostType.GetProperty("ImageProcessingCS", BindingFlags.Instance | BindingFlags.Public);
            var imageProcessingField = hostType.GetField("_imageProcessingCS", BindingFlags.Instance | BindingFlags.NonPublic);
            var ensureHostSetup = hostType.GetMethod("EnsureHostSetup", BindingFlags.Instance | BindingFlags.NonPublic);
            var hostProp = typeof(BasePageView).GetProperty("Host", BindingFlags.Instance | BindingFlags.NonPublic);

            if (createWorkingRt == null || buildMaskedBackground == null || buildHoleMask == null || buildLayerCutout == null
                || toDisplayRect == null || toTextureRect == null || buildAppliedComposite == null
                || dumpTextureAsync == null || readbackTextureAsync == null || imageProcessingProp == null || hostProp == null)
            {
                throw new MissingMethodException("DesignView composite debug reflection setup failed.");
            }

            var host = hostGo.AddComponent<AIImagePageHost>();
            var design = hostGo.AddComponent<DesignView>();
            hostProp.SetValue(design, host);
            design._exportCompositeDebug = true;
            Debug.Log("[DesignViewCompositeDebug] host_and_design_ready");

            ensureHostSetup?.Invoke(host, null);

            var cs = imageProcessingProp.GetValue(host) as ComputeShader;
            if (cs == null)
            {
                cs = Resources.Load<ComputeShader>("ImageProcessing");
                if (cs != null && imageProcessingField != null)
                    imageProcessingField.SetValue(host, cs);
            }
            if (cs == null)
                throw new InvalidOperationException("Failed to resolve ImageProcessing compute shader for DesignView composite debug.");
            Debug.Log("[DesignViewCompositeDebug] compute_ready | name=" + cs.name);

            var maskedBackgroundTask = (UniTask<RenderTexture>)buildMaskedBackground.Invoke(design, new object[] { source, result.mask, cs });
            var maskedBackground = await maskedBackgroundTask;
            var holeMaskTask = (UniTask<RenderTexture>)buildHoleMask.Invoke(design, new object[] { result.mask, cs });
            var holeMask = await holeMaskTask;
            if (maskedBackground == null || holeMask == null)
                throw new InvalidOperationException("Failed to build masked background or hole mask for DesignView composite debug.");
            Debug.Log("[DesignViewCompositeDebug] built_background_and_holemask");

            var width = source.width;
            var height = source.height;

            var layerBoxDataType = designViewType.GetNestedType("LayerBoxData", BindingFlags.NonPublic);
            if (layerBoxDataType == null)
                throw new MissingMemberException("LayerBoxData type not found.");
            var listType = typeof(List<>).MakeGenericType(layerBoxDataType);
            var layers = Activator.CreateInstance(listType);
            var listAdd = listType.GetMethod("Add");
            var normalizedRectField = layerBoxDataType.GetField("normalizedRect", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            var titleField = layerBoxDataType.GetField("title", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            var colorField = layerBoxDataType.GetField("color", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            var previewTextureField = layerBoxDataType.GetField("previewTexture", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            var contentRenderTextureField = layerBoxDataType.GetField("contentRenderTexture", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (listAdd == null || normalizedRectField == null || titleField == null || colorField == null || previewTextureField == null || contentRenderTextureField == null)
                throw new MissingFieldException("LayerBoxData fields not found.");

            var detectionCount = 0;
            var layerColorIndex = 0;
            foreach (var detection in result.detections ?? Array.Empty<YoloSegDetection>())
            {
                var displayRect = (RectInt)toDisplayRect.Invoke(null, new object[] { detection.rect, width, height });
                if (displayRect.width < 8 || displayRect.height < 8)
                    continue;

                var normalized = new Rect(
                    displayRect.x / Mathf.Max(1f, width),
                    displayRect.y / Mathf.Max(1f, height),
                    displayRect.width / Mathf.Max(1f, width),
                    displayRect.height / Mathf.Max(1f, height));
                if (normalized.width < 0.03f || normalized.height < 0.03f)
                    continue;

                var textureRect = (RectInt)toTextureRect.Invoke(null, new object[] { displayRect, width, height });
                var cutoutRt = buildLayerCutout.Invoke(design, new object[] { source, result.mask, textureRect, cs }) as RenderTexture;
                if (cutoutRt == null)
                    continue;

                var layer = Activator.CreateInstance(layerBoxDataType);
                titleField.SetValue(layer, $"Layer {detectionCount + 1} {(detection.probability * 100f):0}%");
                normalizedRectField.SetValue(layer, normalized);
                colorField.SetValue(layer, Color.HSVToRGB((layerColorIndex * 0.17f) % 1f, 0.68f, 1f));
                previewTextureField.SetValue(layer, cutoutRt);
                contentRenderTextureField.SetValue(layer, cutoutRt);
                listAdd.Invoke(layers, new[] { layer });

                detectionCount++;
                layerColorIndex++;
            }

            if (detectionCount == 0)
                throw new InvalidOperationException("DesignView composite debug produced zero usable layers.");
            Debug.Log("[DesignViewCompositeDebug] built_layers | count=" + detectionCount.ToString(CultureInfo.InvariantCulture));

            var applyTaskObj = buildAppliedComposite.Invoke(
                design,
                new object[]
                {
                    source,
                    source,
                    holeMask,
                    layers,
                    8,
                    8,
                    0.35f
                });

            Debug.Log("[DesignViewCompositeDebug] apply_begin");
            //var applyTask = (UniTask<object>)ConvertUniTaskToObject(applyTaskObj);
            object applyResult = new UniTask<object>();
            //var applyResult = await applyTask;
            //if (applyResult == null)
            //    throw new InvalidOperationException("DesignView composite debug apply result is null.");
            Debug.Log("[DesignViewCompositeDebug] apply_done");

            var composedTextureField = applyResult.GetType().GetField("composedTexture", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            var remainingMaskField = applyResult.GetType().GetField("remainingMask", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            var debugDirectoryField = applyResult.GetType().GetField("debugDirectory", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (composedTextureField == null || remainingMaskField == null || debugDirectoryField == null)
                throw new MissingFieldException("ApplyCompositeResult fields not found.");

            var composed = composedTextureField.GetValue(applyResult) as Texture2D;
            var remainingMask = remainingMaskField.GetValue(applyResult) as Texture2D;
            var compositeDebugDir = debugDirectoryField.GetValue(applyResult) as string;
            if (composed == null)
                throw new InvalidOperationException("DesignView composite debug composed texture is null.");

            TryWriteTexturePng(composed, outputDir, "10_composited.png");
            if (remainingMask != null)
                TryWriteTexturePng(remainingMask, outputDir, "11_remaining_mask.png");

            var diffTex = BuildAbsDiffTexture(source, composed);
            if (diffTex != null)
            {
                TryWriteTexturePng(diffTex, outputDir, "12_absdiff_vs_input.png");
                UnityEngine.Object.DestroyImmediate(diffTex);
            }

            ComputeTextureDiff(source, composed, out var meanAbsRgb, out var maxAbsRgb);
            File.WriteAllText(
                Path.Combine(outputDir, "summary.txt"),
                "input=" + inputPath + Environment.NewLine
                + "detections=" + detectionCount.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "mean_abs_rgb_vs_input=" + meanAbsRgb.ToString("0.000000", CultureInfo.InvariantCulture) + Environment.NewLine
                + "max_abs_rgb_vs_input=" + maxAbsRgb.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "designview_debug_dir=" + (compositeDebugDir ?? string.Empty) + Environment.NewLine
                + "yolo_dump_dir=" + (yolo.LastDumpDir ?? string.Empty) + Environment.NewLine);

            Debug.Log(
                "[DesignViewCompositeDebug] done"
                + " | output_dir=" + outputDir
                + " | designview_debug_dir=" + (compositeDebugDir ?? string.Empty)
                + " | mean_abs_rgb_vs_input=" + meanAbsRgb.ToString("0.000000", CultureInfo.InvariantCulture)
                + " | max_abs_rgb_vs_input=" + maxAbsRgb.ToString(CultureInfo.InvariantCulture));

            if (composed != null)
                UnityEngine.Object.DestroyImmediate(composed);
            if (remainingMask != null)
                UnityEngine.Object.DestroyImmediate(remainingMask);
            if (result.texture != null)
                UnityEngine.Object.DestroyImmediate(result.texture);
            if (result.mask != null)
                UnityEngine.Object.DestroyImmediate(result.mask);
            if (result.overlay != null)
                UnityEngine.Object.DestroyImmediate(result.overlay);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(hostGo);
            UnityEngine.Object.DestroyImmediate(source);
        }
    }

    private static async UniTask RunMonaiPack4SelfTestInternal()
    {
        var outputDir = ResolveStringEnv(MonaiOutputDirEnvVar, null);
        if (string.IsNullOrWhiteSpace(outputDir))
            outputDir = CreateGenericDumpDir("AIImage_MONAI_Pack4SelfTest");
        Directory.CreateDirectory(outputDir);

        var textureFormat = ResolveRenderTextureFormatEnv(MonaiPack4SelfTestFormatEnvVar, RenderTextureFormat.ARGBHalf);
        var go = new GameObject("MonaiPack4SelfTestRunner");
        NcnnCompute.NcnnGpuResourceTracker.Enabled = true;
        NcnnCompute.NcnnGpuResourceTracker.Reset("NcnnDebugRunner.MONAI.Pack4SelfTest");
        try
        {
            var runner = go.AddComponent<MONAINcnnReproRunner>();
            runner.enableDebugDump = true;
            runner.enableTempPool = false;
            runner.maxPooledPerShape = 0;
            runner.clearTempPoolAfterEachSlidingWindowPatch = true;
            runner.slidingWindowTempPoolClearInterval = 1;
            runner.slidingWindowYieldInterval = 1;
            runner.slidingWindowManagedCleanupInterval = 1;
            runner.slidingWindowResourceSnapshotInterval = 1;
            runner.featureHeadChunkDepth = MonaiLowPowerModeState.IsEnabled ? 4 : 8;
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[MONAI SelfTest Progress] " + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));

            var summary = runner.RunPack4RoundtripSelfTest(outputDir, textureFormat);
            Debug.Log(
                "MONAI Pack4 SelfTest result"
                + " | format=" + textureFormat
                + " | dump=" + outputDir
                + " | gpu=" + NcnnCompute.NcnnGpuResourceTracker.BuildSummary()
                + " | summary=" + summary.Replace(Environment.NewLine, " | "));
        }
        finally
        {
            try { LogResourceSnapshot("monai_pack4_selftest_before_destroy"); } catch { }
            try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats.txt"); } catch { }
            UnityEngine.Object.DestroyImmediate(go);
            await ReleaseGpuPressureAsync();
            try { LogResourceSnapshot("monai_pack4_selftest_after_release"); } catch { }
            try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats_after_release.txt"); } catch { }
            NcnnCompute.NcnnGpuResourceTracker.Enabled = false;
        }
    }

    private static async UniTask RunStableDiffusionDebugInternal()
    {
        var width = ResolvePositiveIntEnv(SdWidthEnvVar, 256);
        var height = ResolvePositiveIntEnv(SdHeightEnvVar, 256);
        var steps = ResolvePositiveIntEnv(SdStepsEnvVar, 15);
        var seed = ResolveIntEnvAllowZero(SdSeedEnvVar, 42);
        var strength = ResolveFloatEnvOrDefault(SdStrengthEnvVar, 0.75f);
        var positivePrompt = ResolveStringEnv(SdPositivePromptEnvVar, DefaultSdPositivePrompt);
        var negativePrompt = ResolveStringEnv(SdNegativePromptEnvVar, DefaultSdNegativePrompt);
        var initPath = ResolveOptionalExistingFile(SdInitImageEnvVar);
        var maskPath = ResolveOptionalExistingFile(SdMaskImageEnvVar);
        Texture2D initTex = null;
        Texture2D maskTex = null;

        if (!string.IsNullOrWhiteSpace(initPath))
        {
            initTex = LoadTexture(initPath);
            if (initTex == null)
                throw new InvalidOperationException("Failed to load SD init image: " + initPath);
        }

        if (!string.IsNullOrWhiteSpace(maskPath))
        {
            maskTex = LoadTexture(maskPath);
            if (maskTex == null)
                throw new InvalidOperationException("Failed to load SD mask image: " + maskPath);
        }

        var go = new GameObject("StableDiffusionDebugRunner");
        try
        {
            var runner = go.AddComponent<SDNcnnReproRunner>();
            runner.enableDebugDump = ResolveBoolEnv(SdEnableDumpEnvVar, true);
            runner.enableTempPool = false;
            runner.maxPooledPerShape = 0;
            runner.tensorTextureFormat = ResolveRenderTextureFormatEnv(SdTensorFormatEnvVar, runner.tensorTextureFormat);
            runner.decoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdDecoderTensorFormatEnvVar, runner.decoderTensorTextureFormat);
            runner.encoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdEncoderTensorFormatEnvVar, runner.encoderTensorTextureFormat);
            runner.syncStageTimings = ResolveBoolEnv(SdSyncStageTimingsEnvVar, runner.syncStageTimings);
            runner.keepRawConvWeightsForTexturePath = ResolveBoolEnv(SdKeepRawConvWeightsEnvVar, runner.keepRawConvWeightsForTexturePath);
            runner.ProgressChanged += (value, message) =>
            {
                Debug.Log("[SD-DEBUG] progress=" + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            };

            SDNcnnReproResult result;
            if (initTex != null && maskTex != null)
            {
                result = await RunStableDiffusionInpaintingPack4Async(
                    go,
                    initTex,
                    maskTex,
                    positivePrompt,
                    negativePrompt,
                    steps,
                    seed,
                    strength,
                    runner.enableDebugDump,
                    runner.tensorTextureFormat,
                    runner.decoderTensorTextureFormat,
                    runner.encoderTensorTextureFormat,
                    runner.keepRawConvWeightsForTexturePath);
            }
            else if (initTex != null)
            {
                result = await runner.Img2ImgAsync(initTex, positivePrompt, negativePrompt, width, height, steps, seed, strength, CancellationToken.None);
            }
            else
            {
                result = await runner.Txt2ImgAsync(positivePrompt, negativePrompt, width, height, steps, seed, CancellationToken.None);
            }

            Debug.Log(
                "Stable Diffusion Debug result | error=" + (result.error ?? "")
                + " | elapsedMs=" + result.elapsedMs
                + " | seed=" + result.seed.ToString(CultureInfo.InvariantCulture)
                + " | mode=" + (initTex != null ? (maskTex != null ? "inpainting" : "img2img") : "txt2img")
                + " | dump=" + (result.dumpDir ?? runner.LastDumpDir ?? ""));

            if (result.texture != null)
            {
                if (runner.enableDebugDump)
                {
                    var dir = !string.IsNullOrWhiteSpace(result.dumpDir)
                        ? result.dumpDir
                        : !string.IsNullOrWhiteSpace(runner.LastDumpDir)
                            ? runner.LastDumpDir
                        : CreateGenericDumpDir("AIImage_SD_NcnnRepro");
                    TryWriteTexturePng(result.texture, dir, "final_output.png");
                }
                UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        finally
        {
            if (initTex != null)
                UnityEngine.Object.DestroyImmediate(initTex);
            if (maskTex != null)
                UnityEngine.Object.DestroyImmediate(maskTex);
            UnityEngine.Object.DestroyImmediate(go);
        }
    }

    private static async UniTask<SDNcnnReproResult> RunStableDiffusionInpaintingPack4Async(
        GameObject owner,
        Texture initTex,
        Texture maskTex,
        string positivePrompt,
        string negativePrompt,
        int steps,
        int seed,
        float strength,
        bool enableDump,
        RenderTextureFormat tensorFormat,
        RenderTextureFormat decoderTensorFormat,
        RenderTextureFormat encoderTensorFormat,
        bool keepRawConvWeights)
    {
        var runner = owner.AddComponent<SDInpaintingNcnnReproRunner>();
        try
        {
            runner.enableDebugDump = enableDump;
            runner.enableTempPool = false;
            runner.maxPooledPerShape = 0;
            runner.useOfficialUnetCache = false;
            runner.tensorTextureFormat = tensorFormat;
            runner.decoderTensorTextureFormat = decoderTensorFormat;
            runner.encoderTensorTextureFormat = encoderTensorFormat;
            runner.keepRawConvWeightsForTexturePath = keepRawConvWeights;
            runner.enableAttentionMatMulPack4Specializations = true;
            runner.useCommandBuffer = ResolveBoolEnv(SdUseCommandBufferEnvVar, false);
            runner.useAsyncComputeCommandBuffer = ResolveBoolEnv(SdUseAsyncComputeEnvVar, runner.useAsyncComputeCommandBuffer);
            runner.disallowInferenceTempComputeBuffers = ResolveBoolEnv(SdDisallowTempComputeBuffersEnvVar, true);
            runner.defaultGuidanceScale = ResolveFloatEnvOrDefault(SdGuidanceScaleEnvVar, 7.5f);
            runner.ProgressChanged += (value, message) =>
            {
                Debug.Log("[SD-INPAINT-PACK4] progress=" + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            };

            var result = await runner.ProcessAsync(
                initTex,
                maskTex,
                positivePrompt,
                negativePrompt,
                steps,
                seed,
                strength,
                runner.defaultGuidanceScale,
                CancellationToken.None);

            return new SDNcnnReproResult
            {
                texture = result.texture,
                error = result.error,
                elapsedMs = result.elapsedMs,
                seed = result.seed,
                usedInitImage = true,
                usedMask = true,
                dumpDir = result.dumpDir ?? runner.LastDumpDir
            };
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(runner);
        }
    }

    private static async UniTask RunYoloAndInpaintingDebugInternal()
    {
        var replayBaselineDir = ResolveOptionalExistingDirectory(SdReplayBaselineDirEnvVar);
        if (!string.IsNullOrWhiteSpace(replayBaselineDir))
        {
            await RunSdUnetReplayInternal(replayBaselineDir);
            return;
        }

        var inputPath = ResolveInputPath(DefaultReproStressImagePath);
        var enableDump = ResolveBoolEnv(SdEnableDumpEnvVar, false);
        var stepCount = ResolvePositiveIntEnv(SdStepsEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedStepCount);
        var seed = ResolveIntEnvAllowZero(SdSeedEnvVar, 123456);
        var strength = Mathf.Clamp01(ResolveFloatEnvOrDefault(SdStrengthEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedStrength));
        var guidanceScale = Mathf.Max(1f, ResolveFloatEnvOrDefault(SdGuidanceScaleEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedGuidanceScale));
        var positivePrompt = ResolveStringEnv(SdPositivePromptEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedPositivePrompt);
        var negativePrompt = ResolveStringEnv(SdNegativePromptEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedNegativePrompt);
        var outputDir = CreateGenericDumpDir("AIImage_YoloInpaintingRepro");
        var forcedUseCommandBuffer = ResolveOptionalBoolEnv(SdUseCommandBufferEnvVar);
        var runPack4Rt = !forcedUseCommandBuffer.HasValue || !forcedUseCommandBuffer.Value;
        var runCommandBuffer = !forcedUseCommandBuffer.HasValue || forcedUseCommandBuffer.Value;
        NcnnCompute.NcnnGpuResourceTracker.Enabled = true;
        NcnnCompute.NcnnGpuResourceTracker.Reset("NcnnDebugRunner.YoloInpaint");
        try
        {
            if (runPack4Rt)
            {
                var modeDir = Path.Combine(outputDir, "pack4_rt");
                Directory.CreateDirectory(modeDir);
                await RunYoloAndInpaintingDebugOnce(
                    inputPath,
                    modeDir,
                    enableDump,
                    stepCount,
                    seed,
                    strength,
                    guidanceScale,
                    positivePrompt,
                    negativePrompt,
                    useCommandBuffer: false,
                    logTag: "batch.pack4_rt");
            }

            if (runCommandBuffer)
            {
                await ReleaseGpuPressureAsync();
                var modeDir = Path.Combine(outputDir, "command_buffer");
                Directory.CreateDirectory(modeDir);
                await RunYoloAndInpaintingDebugOnce(
                    inputPath,
                    modeDir,
                    enableDump,
                    stepCount,
                    seed,
                    strength,
                    guidanceScale,
                    positivePrompt,
                    negativePrompt,
                    useCommandBuffer: true,
                    logTag: "batch.command_buffer");
            }
        }
        finally
        {
            try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats.txt"); } catch { }
            NcnnCompute.NcnnGpuResourceTracker.Enabled = false;
        }
    }

    private static async UniTask RunStableDiffusionBaselineDebugInternal()
    {
        var baselineDir = ResolveOptionalExistingDirectory(SdDirectBaselineDirEnvVar);
        if (string.IsNullOrWhiteSpace(baselineDir))
            throw new DirectoryNotFoundException("baseline dir not found: " + Environment.GetEnvironmentVariable(SdDirectBaselineDirEnvVar));

        var sourcePath = Path.Combine(baselineDir, "01_source_512.png");
        var maskPath = Path.Combine(baselineDir, "02_mask_512.png");
        var positivePath = Path.Combine(baselineDir, "positive_prompt.txt");
        var negativePath = Path.Combine(baselineDir, "negative_prompt.txt");
        if (!File.Exists(sourcePath) || !File.Exists(maskPath))
            throw new FileNotFoundException("baseline source or mask missing", baselineDir);

        var source = LoadTexture(sourcePath);
        var mask = LoadTexture(maskPath);
        if (source == null || mask == null)
            throw new InvalidOperationException("Failed to load baseline input textures: " + baselineDir);

        var positivePrompt = File.Exists(positivePath)
            ? File.ReadAllText(positivePath).Trim()
            : ResolveStringEnv(SdPositivePromptEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedPositivePrompt);
        var negativePrompt = File.Exists(negativePath)
            ? File.ReadAllText(negativePath).Trim()
            : ResolveStringEnv(SdNegativePromptEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedNegativePrompt);

        var go = new GameObject("StableDiffusionBaselineDebugRunner");
        try
        {
            var runner = go.AddComponent<SDInpaintingNcnnReproRunner>();
            runner.enableDebugDump = ResolveBoolEnv(SdEnableDumpEnvVar, true);
            runner.enableTempPool = false;
            runner.maxPooledPerShape = 0;
            runner.tensorTextureFormat = ResolveRenderTextureFormatEnv(SdTensorFormatEnvVar, runner.tensorTextureFormat);
            runner.decoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdDecoderTensorFormatEnvVar, runner.decoderTensorTextureFormat);
            runner.encoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdEncoderTensorFormatEnvVar, runner.encoderTensorTextureFormat);
            runner.keepRawConvWeightsForTexturePath = ResolveBoolEnv(SdKeepRawConvWeightsEnvVar, runner.keepRawConvWeightsForTexturePath);
            runner.enableAttentionMatMulPack4Specializations = true;
            runner.useCommandBuffer = ResolveBoolEnv(SdUseCommandBufferEnvVar, false);
            runner.disallowInferenceTempComputeBuffers = ResolveBoolEnv(SdDisallowTempComputeBuffersEnvVar, true);

            var configPath = Path.Combine(baselineDir, "run_config.txt");
            var stepCount = SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedStepCount;
            var seed = 123456;
            var strength = SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedStrength;
            var guidanceScale = SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedGuidanceScale;
            if (File.Exists(configPath))
            {
                foreach (var line in File.ReadAllLines(configPath))
                {
                    var parts = line.Split(new[] { '=' }, 2);
                    if (parts.Length != 2)
                        continue;
                    var key = parts[0].Trim();
                    var value = parts[1].Trim();
                    if (key == "steps" && int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedSteps))
                        stepCount = parsedSteps;
                    else if (key == "seed" && int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedSeed))
                        seed = parsedSeed;
                    else if (key == "strength" && float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsedStrength))
                        strength = parsedStrength;
                    else if (key == "guidance_scale" && float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsedGuidance))
                        guidanceScale = parsedGuidance;
                    else if (key == "black_mask_means_inpaint" && bool.TryParse(value, out var parsedBlackMask))
                        runner.blackMaskMeansInpaint = parsedBlackMask;
                }
            }

            var result = await runner.ProcessAsync(
                source,
                mask,
                positivePrompt,
                negativePrompt,
                stepCount,
                seed,
                strength,
                guidanceScale,
                CancellationToken.None);

            Debug.Log(
                "[SDBaseline] baseline=" + baselineDir
                + " | error=" + (result.error ?? "")
                + " | elapsedMs=" + result.elapsedMs
                + " | seed=" + result.seed.ToString(CultureInfo.InvariantCulture)
                + " | dump=" + (result.dumpDir ?? runner.LastDumpDir ?? ""));

            if (!string.IsNullOrWhiteSpace(result.error))
                throw new InvalidOperationException(result.error);

            if (result.texture != null)
                UnityEngine.Object.DestroyImmediate(result.texture);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(source);
            UnityEngine.Object.DestroyImmediate(mask);
            UnityEngine.Object.DestroyImmediate(go);
        }
    }

    private static async UniTask RunSdUnetReplayInternal(string baselineDir)
    {
        var referenceDir = ResolveOptionalExistingDirectory(SdReplayReferenceDirEnvVar);
        var startTop = ResolveStringEnv(SdReplayStartTopEnvVar, null);
        var stopTop = ResolveStringEnv(SdReplayStopTopEnvVar, null);
        var outputBlob = ResolveStringEnv(SdReplayOutputBlobEnvVar, null);
        var targetBlob = ResolveStringEnv(SdReplayTargetBlobEnvVar, null);
        var promptKind = ResolveStringEnv(SdReplayPromptKindEnvVar, "cond");
        var inputBlobsRaw = ResolveStringEnv(SdReplayInputBlobsEnvVar, null);
        var inputBlobs = string.IsNullOrWhiteSpace(inputBlobsRaw)
            ? null
            : inputBlobsRaw.Split(new[] { ',', ';', '\r', '\n', '\t', ' ' }, StringSplitOptions.RemoveEmptyEntries);
        var outputDir = CreateGenericDumpDir("AIImage_SD_UnetReplay");

        var go = new GameObject("SdUnetReplayRunner");
        try
        {
            NcnnCompute.NcnnGpuResourceTracker.Enabled = true;
            NcnnCompute.NcnnGpuResourceTracker.Reset("NcnnDebugRunner.SDUnetReplay");
            var runner = go.AddComponent<SDInpaintingNcnnReproRunner>();
            runner.enableDebugDump = true;
            runner.enableTempPool = false;
            runner.maxPooledPerShape = 0;
            runner.tensorTextureFormat = ResolveRenderTextureFormatEnv(SdTensorFormatEnvVar, runner.tensorTextureFormat);
            runner.keepRawConvWeightsForTexturePath = ResolveBoolEnv(SdKeepRawConvWeightsEnvVar, runner.keepRawConvWeightsForTexturePath);
            runner.enableAttentionMatMulPack4Specializations = true;
            runner.disallowInferenceTempComputeBuffers = ResolveBoolEnv(SdDisallowTempComputeBuffersEnvVar, true);

            var result = await runner.RunUnetBaselineReplayAsync(
                new SDInpaintingNcnnReproRunner.UnetBaselineReplayRequest(
                    baselineDir,
                    referenceDir,
                    outputDir,
                    startTop,
                    stopTop,
                    outputBlob,
                    targetBlob,
                    promptKind,
                    inputBlobs),
                CancellationToken.None);

            Debug.Log(
                "[SDReplay] baseline=" + baselineDir
                + " | reference=" + (referenceDir ?? baselineDir)
                + " | outputDir=" + (result.outputDir ?? outputDir)
                + " | report=" + (result.reportPath ?? string.Empty)
                + " | error=" + (result.error ?? string.Empty));

            if (!string.IsNullOrWhiteSpace(result.error))
                throw new InvalidOperationException(result.error);
        }
        finally
        {
            try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats.txt"); } catch { }
            NcnnCompute.NcnnGpuResourceTracker.Enabled = false;
            UnityEngine.Object.DestroyImmediate(go);
        }
    }

    private readonly struct MonaiModelConfig
    {
        public readonly string modelParamPath;
        public readonly string modelBinPath;
        public readonly string pnnxParamPath;
        public readonly string bundleManifestPath;
        public readonly MonaiPostprocessKind postprocessKind;

        public MonaiModelConfig(string modelParamPath, string modelBinPath, string pnnxParamPath, string bundleManifestPath, MonaiPostprocessKind postprocessKind)
        {
            this.modelParamPath = modelParamPath;
            this.modelBinPath = modelBinPath;
            this.pnnxParamPath = pnnxParamPath;
            this.bundleManifestPath = bundleManifestPath;
            this.postprocessKind = postprocessKind;
        }
    }

    private static MonaiModelConfig ResolveMonaiModelConfig(string baselineManifestPath)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(baselineManifestPath) || !File.Exists(baselineManifestPath))
                return new MonaiModelConfig(null, null, null, null, MonaiPostprocessKind.BratsTumorSubregions);

            var baseline = JObject.Parse(File.ReadAllText(baselineManifestPath));
            var bundleRoot = baseline["bundle_root"]?.Value<string>();
            var bundleName = !string.IsNullOrWhiteSpace(bundleRoot) ? Path.GetFileName(bundleRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)) : null;
            if (string.IsNullOrWhiteSpace(bundleName))
                return new MonaiModelConfig(null, null, null, null, MonaiPostprocessKind.BratsTumorSubregions);

            var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "Tools", "MonaiToNCNN", "outputs", bundleName);
            var postprocess = string.Equals(baseline["task_mode"]?.Value<string>(), "multiclass", StringComparison.OrdinalIgnoreCase)
                ? MonaiPostprocessKind.MulticlassArgmax
                : MonaiPostprocessKind.BratsTumorSubregions;
            if (string.Equals(baseline["task_mode"]?.Value<string>(), "binary_label_prompt", StringComparison.OrdinalIgnoreCase))
                postprocess = MonaiPostprocessKind.BinaryLabelPrompt;

            var ncnnAssets = baseline["ncnn_assets"] as JObject;

            var pnnxParam = ncnnAssets?["pnnx_param_path"]?.Value<string>();
            if (string.IsNullOrWhiteSpace(pnnxParam))
                pnnxParam = Path.Combine(outputDir, bundleName + ".sim.pnnx.param");
            if (!File.Exists(pnnxParam))
            {
                var altPnnx = Path.Combine(outputDir, bundleName + ".sim.onnx");
                altPnnx = Path.ChangeExtension(altPnnx, ".pnnx.param");
                if (File.Exists(altPnnx))
                    pnnxParam = altPnnx;
            }

            var modelParam = ncnnAssets?["model_param_path"]?.Value<string>();
            var modelBin = ncnnAssets?["model_bin_path"]?.Value<string>();
            if (string.IsNullOrWhiteSpace(modelParam))
                modelParam = Path.Combine(outputDir, bundleName + ".param");
            if (string.IsNullOrWhiteSpace(modelBin))
                modelBin = Path.Combine(outputDir, bundleName + ".bin");
            if (!File.Exists(modelParam) || !File.Exists(modelBin))
            {
                var fallbackParam = Path.Combine(outputDir, bundleName + ".sim.ncnn.param");
                var fallbackBin = Path.Combine(outputDir, bundleName + ".sim.ncnn.bin");
                if (File.Exists(fallbackParam) && File.Exists(fallbackBin))
                {
                    modelParam = fallbackParam;
                    modelBin = fallbackBin;
                }
            }

            var bundleManifest = ncnnAssets?["bundle_manifest_path"]?.Value<string>();
            if (string.IsNullOrWhiteSpace(bundleManifest))
                bundleManifest = Path.Combine(outputDir, "manifest.json");
            return new MonaiModelConfig(modelParam, modelBin, pnnxParam, bundleManifest, postprocess);
        }
        catch
        {
            return new MonaiModelConfig(null, null, null, null, MonaiPostprocessKind.BratsTumorSubregions);
        }
    }

    [MenuItem("Tools/AIImage/Run VISTA3D Debug")]
    public static void RunVista3dDebugMenu()
    {
        RunVista3dDebug().Forget();
    }

    private static async UniTask RunYoloAndInpaintingProbeInternal()
    {
        var inputPath = ResolveInputPath(DefaultReproStressImagePath);
        var enableDump = ResolveBoolEnv(SdEnableDumpEnvVar, false);
        var iterations = ResolvePositiveIntEnv(StressCountEnvVar, 3);
        var stepCount = ResolvePositiveIntEnv(SdStepsEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedStepCount);
        var seed = ResolveIntEnvAllowZero(SdSeedEnvVar, 123456);
        var strength = Mathf.Clamp01(ResolveFloatEnvOrDefault(SdStrengthEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedStrength));
        var guidanceScale = Mathf.Max(1f, ResolveFloatEnvOrDefault(SdGuidanceScaleEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedGuidanceScale));
        var positivePrompt = ResolveStringEnv(SdPositivePromptEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedPositivePrompt);
        var negativePrompt = ResolveStringEnv(SdNegativePromptEnvVar, SDInpaintingNcnnReproRunner.PeopleRemovalRecommendedNegativePrompt);
        var outputDir = CreateGenericDumpDir("AIImage_YoloInpaintingProbe");
        var summaryPath = Path.Combine(outputDir, "probe_summary.tsv");

        Directory.CreateDirectory(outputDir);
        using var sw = new StreamWriter(summaryPath, false);
        sw.WriteLine("iter\tstage\tprivate_mb\tworking_set_mb\tmanaged_mb\tgfx_mb\trt_objects\tgpu_summary\toutput_dir");

        NcnnCompute.NcnnGpuResourceTracker.Enabled = true;
        try
        {
            for (var i = 0; i < iterations; i++)
            {
                var iteration = i + 1;
                var iterationDir = Path.Combine(outputDir, "iter_" + iteration.ToString("00", CultureInfo.InvariantCulture));
                Directory.CreateDirectory(iterationDir);

                NcnnCompute.NcnnGpuResourceTracker.Reset("NcnnDebugRunner.YoloInpaint.Probe." + iteration.ToString(CultureInfo.InvariantCulture));
                LogResourceSnapshot("probe_iter_" + iteration.ToString(CultureInfo.InvariantCulture) + "_begin");
                WriteResourceSummaryRow(sw, iteration, "begin", iterationDir);

                await RunYoloAndInpaintingDebugOnce(
                    inputPath,
                    iterationDir,
                    enableDump,
                    stepCount,
                    seed,
                    strength,
                    guidanceScale,
                    positivePrompt,
                    negativePrompt,
                    useCommandBuffer: ResolveBoolEnv(SdUseCommandBufferEnvVar, false),
                    logTag: "probe#" + iteration.ToString(CultureInfo.InvariantCulture));

                LogResourceSnapshot("probe_iter_" + iteration.ToString(CultureInfo.InvariantCulture) + "_after_run");
                WriteResourceSummaryRow(sw, iteration, "after_run", iterationDir);
                await ReleaseGpuPressureAsync();
                LogResourceSnapshot("probe_iter_" + iteration.ToString(CultureInfo.InvariantCulture) + "_after_release");
                WriteResourceSummaryRow(sw, iteration, "after_release", iterationDir);
                try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(iterationDir, "gpu_resource_stats.txt"); } catch { }
                sw.Flush();
            }
        }
        finally
        {
            NcnnCompute.NcnnGpuResourceTracker.Enabled = false;
        }
    }

    private static async UniTask RunYoloAndInpaintingDebugOnce(
        string inputPath,
        string outputDir,
        bool enableDump,
        int stepCount,
        int seed,
        float strength,
        float guidanceScale,
        string positivePrompt,
        string negativePrompt,
        bool useCommandBuffer,
        string logTag)
    {
        var tex = LoadTexture(inputPath);
        if (tex == null)
            throw new InvalidOperationException("Failed to load debug input: " + inputPath);

        TryWriteTexturePng(tex, outputDir, "00_source.png");
        LogResourceSnapshot(logTag + "_yolo_inpaint_begin");

        var go = new GameObject("YoloAndInpaintingDebugRunner");
        YoloSegResult yoloResult = default;
        SDInpaintingNcnnReproResult inpaintResult = default;
        string yoloDumpDir = null;
        try
        {
            var yoloRunner = go.AddComponent<YoloSegNcnnReproRunner>();
            yoloRunner.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
            yoloRunner.enableDebugDump = enableDump;
            yoloRunner.targetPersonOnly = true;
            yoloRunner.enableMaskClose = true;
            yoloRunner.enableMaskDilate = true;
            yoloRunner.flipYInput = ResolveBoolEnv(YoloFlipYEnvVar, yoloRunner.flipYInput);
            yoloRunner.ProgressChanged += (value, message) =>
            {
                Debug.Log("[" + logTag + "][YOLO] progress=" + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            };

            yoloResult = await yoloRunner.ProcessAsync(tex, CancellationToken.None);
            LogResourceSnapshot(logTag + "_after_yolo_process");
            yoloDumpDir = yoloRunner.LastDumpDir;
            Debug.Log(
                "[" + logTag + "] yoloError=" + (yoloResult.error ?? "")
                + " | yoloElapsedMs=" + yoloResult.elapsedMs
                + " | personCount=" + yoloResult.personCount.ToString(CultureInfo.InvariantCulture)
                + " | coverage=" + yoloResult.maskCoverage01.ToString("0.000000", CultureInfo.InvariantCulture)
                + " | yoloDump=" + (yoloDumpDir ?? ""));

            if (!string.IsNullOrWhiteSpace(yoloResult.error))
                throw new InvalidOperationException("YOLO failed: " + yoloResult.error);
            if (yoloResult.mask == null)
                throw new InvalidOperationException("YOLO mask is null.");
            if (yoloResult.personCount <= 0)
                throw new InvalidOperationException("YOLO detected no person regions for inpainting.");
            if (yoloResult.maskCoverage01 <= 0f)
                throw new InvalidOperationException("YOLO person mask coverage is zero.");

            TryWriteTexturePng(yoloResult.mask, outputDir, "01_person_mask.png");
            TryWriteTexturePng(yoloResult.texture, outputDir, "02_transparent_cutout.png");
            TryWriteTexturePng(yoloResult.overlay, outputDir, "03_overlay.png");
            if (yoloResult.texture != null)
            {
                UnityEngine.Object.DestroyImmediate(yoloResult.texture);
                yoloResult.texture = null;
            }
            if (yoloResult.overlay != null)
            {
                UnityEngine.Object.DestroyImmediate(yoloResult.overlay);
                yoloResult.overlay = null;
            }

            TryInvokeReleaseMethod(yoloRunner);
            UnityEngine.Object.DestroyImmediate(yoloRunner);
            await ReleaseGpuPressureAsync();
            LogResourceSnapshot(logTag + "_after_yolo_release");

            var inpaintRunner = go.AddComponent<SDInpaintingNcnnReproRunner>();
            inpaintRunner.enableDebugDump = enableDump;
            inpaintRunner.ApplyPeopleRemovalPreset();
            inpaintRunner.useOfficialUnetCache = false;
            inpaintRunner.enableTempPool = false;
            inpaintRunner.maxPooledPerShape = 0;
            inpaintRunner.tensorTextureFormat = ResolveRenderTextureFormatEnv(SdTensorFormatEnvVar, inpaintRunner.tensorTextureFormat);
            inpaintRunner.decoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdDecoderTensorFormatEnvVar, inpaintRunner.decoderTensorTextureFormat);
            inpaintRunner.encoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdEncoderTensorFormatEnvVar, inpaintRunner.encoderTensorTextureFormat);
            inpaintRunner.keepRawConvWeightsForTexturePath = ResolveBoolEnv(SdKeepRawConvWeightsEnvVar, inpaintRunner.keepRawConvWeightsForTexturePath);
            inpaintRunner.enableAttentionMatMulPack4Specializations = true;
            inpaintRunner.useCommandBuffer = useCommandBuffer;
            inpaintRunner.useAsyncComputeCommandBuffer = ResolveBoolEnv(SdUseAsyncComputeEnvVar, inpaintRunner.useAsyncComputeCommandBuffer);
            inpaintRunner.disallowInferenceTempComputeBuffers = ResolveBoolEnv(SdDisallowTempComputeBuffersEnvVar, true);
            inpaintRunner.defaultStepCount = stepCount;
            inpaintRunner.defaultStrength = strength;
            inpaintRunner.defaultGuidanceScale = guidanceScale;
            inpaintRunner.defaultPositivePrompt = positivePrompt;
            inpaintRunner.defaultNegativePrompt = negativePrompt;
            inpaintRunner.ProgressChanged += (value, message) =>
            {
                Debug.Log("[" + logTag + "][SD] progress=" + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            };

            inpaintResult = await inpaintRunner.ProcessAsync(
                tex,
                yoloResult.mask,
                positivePrompt,
                negativePrompt,
                stepCount,
                seed,
                strength,
                guidanceScale,
                CancellationToken.None);
            LogResourceSnapshot(logTag + "_after_inpaint_process");

            Debug.Log(
                "[" + logTag + "] inpaintError=" + (inpaintResult.error ?? "")
                + " | mode=" + (useCommandBuffer ? "command_buffer" : "pack4_rt")
                + " | inpaintElapsedMs=" + inpaintResult.elapsedMs
                + " | seed=" + inpaintResult.seed.ToString(CultureInfo.InvariantCulture)
                + " | inpaintDump=" + (inpaintResult.dumpDir ?? inpaintRunner.LastDumpDir ?? ""));

            if (!string.IsNullOrWhiteSpace(inpaintResult.error))
                throw new InvalidOperationException("SD inpainting failed: " + inpaintResult.error);
            if (inpaintResult.texture == null)
                throw new InvalidOperationException("SD inpainting output texture is null.");

            TryWriteTexturePng(inpaintResult.texture, outputDir, "07_final_output.png");

            var maskedDiff = ComputeMaskedMeanAbsDiff(
                tex,
                inpaintResult.texture,
                yoloResult.mask,
                inpaintRunner.blackMaskMeansInpaint,
                out var maskedPixels);
            var summary = string.Join(
                Environment.NewLine,
                "input=" + inputPath,
                "person_count=" + yoloResult.personCount.ToString(CultureInfo.InvariantCulture),
                "mask_coverage=" + yoloResult.maskCoverage01.ToString("0.000000", CultureInfo.InvariantCulture),
                "masked_pixels=" + maskedPixels.ToString(CultureInfo.InvariantCulture),
                "masked_mean_abs_diff_rgb=" + maskedDiff.ToString("0.0000", CultureInfo.InvariantCulture),
                "seed=" + seed.ToString(CultureInfo.InvariantCulture),
                "steps=" + stepCount.ToString(CultureInfo.InvariantCulture),
                "strength=" + strength.ToString("0.0000", CultureInfo.InvariantCulture),
                "guidance_scale=" + guidanceScale.ToString("0.0000", CultureInfo.InvariantCulture),
                "mode=" + (useCommandBuffer ? "command_buffer" : "pack4_rt"),
                "yolo_dump=" + (yoloDumpDir ?? string.Empty),
                "inpaint_dump=" + (inpaintResult.dumpDir ?? inpaintRunner.LastDumpDir ?? string.Empty));
            File.WriteAllText(Path.Combine(outputDir, "summary.txt"), summary);
            Debug.Log("[" + logTag + "] summary\n" + summary);

            if (maskedPixels <= 0)
                throw new InvalidOperationException("YOLO mask has zero pixels.");
            if (maskedDiff <= 1f)
                throw new InvalidOperationException("Masked mean RGB diff is too small: " + maskedDiff.ToString("0.0000", CultureInfo.InvariantCulture));
        }
        finally
        {
            LogResourceSnapshot(logTag + "_finally_begin");
            if (inpaintResult.texture != null)
                UnityEngine.Object.DestroyImmediate(inpaintResult.texture);
            if (yoloResult.texture != null)
                UnityEngine.Object.DestroyImmediate(yoloResult.texture);
            if (yoloResult.mask != null)
                UnityEngine.Object.DestroyImmediate(yoloResult.mask);
            if (yoloResult.overlay != null)
                UnityEngine.Object.DestroyImmediate(yoloResult.overlay);
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
            await ReleaseGpuPressureAsync();
            LogResourceSnapshot(logTag + "_finally_end");
        }
    }

    private static void LogResourceSnapshot(string stage)
    {
        try
        {
            GetProcessMemorySnapshotMb(out var privateMb, out var workingSetMb, out var managedMb);
            var gfxMb = GetGraphicsDriverMemoryMb();
            var rtCount = Resources.FindObjectsOfTypeAll<RenderTexture>().Length;
            Debug.Log(
                "[NcnnDebugRunner][Resources] stage=" + (stage ?? "")
                + " | private_mb=" + privateMb.ToString("F3", CultureInfo.InvariantCulture)
                + " | working_set_mb=" + workingSetMb.ToString("F3", CultureInfo.InvariantCulture)
                + " | managed_mb=" + managedMb.ToString("F3", CultureInfo.InvariantCulture)
                + " | gfx_mb=" + gfxMb.ToString("F3", CultureInfo.InvariantCulture)
                + " | rt_objects=" + rtCount.ToString(CultureInfo.InvariantCulture)
                + " | " + NcnnCompute.NcnnGpuResourceTracker.BuildSummary());
        }
        catch (Exception e)
        {
            try { Debug.Log("[NcnnDebugRunner][Resources] stage=" + (stage ?? "") + " | snapshot_failed=" + e.Message); } catch { }
        }
    }

    private static void WriteResourceSummaryRow(StreamWriter sw, int iteration, string stage, string outputDir)
    {
        if (sw == null)
            return;

        GetProcessMemorySnapshotMb(out var privateMb, out var workingSetMb, out var managedMb);
        var gfxMb = GetGraphicsDriverMemoryMb();
        var rtCount = Resources.FindObjectsOfTypeAll<RenderTexture>().Length;
        sw.WriteLine(
            iteration.ToString(CultureInfo.InvariantCulture) + "\t"
            + EscapeTsv(stage ?? string.Empty) + "\t"
            + privateMb.ToString("F3", CultureInfo.InvariantCulture) + "\t"
            + workingSetMb.ToString("F3", CultureInfo.InvariantCulture) + "\t"
            + managedMb.ToString("F3", CultureInfo.InvariantCulture) + "\t"
            + gfxMb.ToString("F3", CultureInfo.InvariantCulture) + "\t"
            + rtCount.ToString(CultureInfo.InvariantCulture) + "\t"
            + EscapeTsv(NcnnCompute.NcnnGpuResourceTracker.BuildSummary()) + "\t"
            + EscapeTsv(outputDir ?? string.Empty));
    }

    private static async UniTask ReleaseGpuPressureAsync()
    {
        await YieldIfNeeded();
        RenderTexture.active = null;
        if (!Application.isBatchMode)
        {
            var unloadOp = Resources.UnloadUnusedAssets();
            if (unloadOp != null)
                await unloadOp.ToUniTask();
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        RenderTexture.active = null;
        await YieldIfNeeded();
    }

    private static async UniTask YieldIfNeeded()
    {
        if (!Application.isBatchMode)
            await UniTask.Yield();
    }

    private static void GetProcessMemorySnapshotMb(out double privateMb, out double workingSetMb, out double managedMb)
    {
        privateMb = 0d;
        workingSetMb = 0d;
        managedMb = GC.GetTotalMemory(false) / (1024d * 1024d);

        try
        {
            var process = Process.GetCurrentProcess();
            privateMb = process.PrivateMemorySize64 / (1024d * 1024d);
            workingSetMb = process.WorkingSet64 / (1024d * 1024d);
        }
        catch
        {
        }

        try
        {
            var reservedMb = Profiler.GetTotalReservedMemoryLong() / (1024d * 1024d);
            var allocatedMb = Profiler.GetTotalAllocatedMemoryLong() / (1024d * 1024d);
            if (privateMb <= 0d)
                privateMb = Math.Max(allocatedMb, reservedMb);
            if (workingSetMb <= 0d)
                workingSetMb = Math.Max(allocatedMb, reservedMb);
            if (managedMb <= 0d)
                managedMb = allocatedMb;
        }
        catch
        {
        }
    }

    public static void RunCodeFormerDebugBatch() => RunBatchBlocking(nameof(RunCodeFormerDebugBatch), RunCodeFormerDebugInternal);

    public static void RunClipDebugBatch() => RunBatchBlocking(nameof(RunClipDebugBatch), RunClipDebugInternal);

    public static void RunClipDirectoryDebugBatch() => RunBatchBlocking(nameof(RunClipDirectoryDebugBatch), RunClipDirectoryDebugInternal);

    public static void RunGfpganDebugBatch() => RunBatchBlocking(nameof(RunGfpganDebugBatch), RunGfpganDebugInternal);

    public static void RunCodeFormerStressBatch() => RunBatchBlocking(nameof(RunCodeFormerStressBatch), RunCodeFormerStressInternal, TimeSpan.FromHours(2));

    public static void RunReproSuiteStressBatch() => RunBatchBlocking(nameof(RunReproSuiteStressBatch), RunReproSuiteStressInternal, TimeSpan.FromHours(2));

    public static void RunMonaiDebugBatch() => RunBatchBlocking(nameof(RunMonaiDebugBatch), () => RunMonaiDebugInternal(), TimeSpan.FromMinutes(10));

    private static string FindLatestMonaiOutputDir()
    {
        try
        {
            var root = Path.Combine(Directory.GetCurrentDirectory(), "Logs", "MONAINcnnRepro");
            if (!Directory.Exists(root))
                return null;
            return Directory.GetDirectories(root)
                .OrderByDescending(Directory.GetLastWriteTimeUtc)
                .FirstOrDefault();
        }
        catch
        {
            return null;
        }
    }

    private static void RunBatchBlocking(string methodName, Func<UniTask> taskFactory, TimeSpan? timeout = null)
    {
        try
        {
            Debug.Log("[NcnnDebugRunner] " + methodName + " start");
            RunUniTaskBlocking(taskFactory, ResolveBatchTimeout(timeout ?? TimeSpan.FromMinutes(45)));
            Debug.Log("[NcnnDebugRunner] " + methodName + " done");
            EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.Log("[NcnnDebugRunner] " + methodName + " failed: " + e.Message);
            Debug.LogException(e);
            EditorApplication.Exit(1);
        }
    }

    private static void RunUniTaskBlocking(Func<UniTask> taskFactory, TimeSpan timeout)
    {
        if (taskFactory == null)
            throw new ArgumentNullException(nameof(taskFactory));

        Exception failure = null;
        var completed = false;

        async UniTask ExecuteAsync()
        {
            try
            {
                await taskFactory();
            }
            catch (Exception e)
            {
                failure = e;
            }
            finally
            {
                completed = true;
            }
        }

        ExecuteAsync().Forget();

        var sw = Stopwatch.StartNew();
        while (!completed)
        {
            if (sw.Elapsed > timeout)
                throw new TimeoutException("Timed out waiting for batch task after " + timeout + ".");

            PumpEditorLoopOnce();
            Thread.Sleep(10);
        }

        if (failure != null)
            throw failure;
    }

    private static void PumpEditorLoopOnce()
    {
        EditorApplication.QueuePlayerLoopUpdate();

        if (EditorUpdatePumpMethod == null)
            throw new MissingMethodException("Unable to resolve Unity Editor pump methods for batch execution.");

        EditorUpdatePumpMethod.Invoke(null, null);
        EditorDelayPumpMethod?.Invoke(null, null);
    }

    private static TimeSpan ResolveBatchTimeout(TimeSpan fallback)
    {
        var raw = Environment.GetEnvironmentVariable(BatchTimeoutMinutesEnvVar);
        if (!string.IsNullOrWhiteSpace(raw)
            && float.TryParse(raw.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var minutes)
            && minutes > 0f)
        {
            return TimeSpan.FromMinutes(minutes);
        }

        return fallback;
    }

    private static async UniTask RunFaceDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultFaceDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
            throw new InvalidOperationException("Failed to load debug input: " + inputPath);

        var go = new GameObject("NcnnFaceDebugRunner");
        try
        {
            var face = go.AddComponent<NcnnFaceRegionGenerator>();
            face.enableNcnnFaceRegion = true;
            face.preferTexturePathForFaceDetector = ResolveFacePreferTexturePath();
            ApplyFacePack4GuardFromEnv(face);
            ApplyFaceThresholdOverrides(face);
            face.enableDetailedProposalDump = true;
            face.autoOpenDumpDir = false;
            var result = await face.GenerateAsync(tex, true, CancellationToken.None);
            Debug.Log("NCNN Face Debug result | error=" + (result.error ?? "") + " | dump=" + (result.dumpDir ?? ""));
            if (result.mask != null)
                UnityEngine.Object.DestroyImmediate(result.mask);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    private static async UniTask RunClipDirectoryDebugInternal()
    {
        var inputDir = ResolveClipInputDirectory();
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
            throw new InvalidOperationException("CLIP input dir not found: " + (inputDir ?? ""));

        var files = EnumerateImageFilesRecursive(inputDir);
        if (files.Count == 0)
            throw new InvalidOperationException("No images found under: " + inputDir);

        var outputDir = CreateGenericDumpDir("AIImage_ClipDirBatch");
        var summaryPath = Path.Combine(outputDir, "summary.tsv");

        NcnnCompute.NcnnGpuResourceTracker.Enabled = true;
        NcnnCompute.NcnnGpuResourceTracker.Reset("clip_dir_batch");

        var go = new GameObject("ClipDirectoryDebugRunner");
        try
        {
            var runner = go.AddComponent<ClipNcnnReproRunner>();
            runner.enableTempPool = false;
            runner.maxPooledPerShape = 0;
            runner.modelLevel = ResolveClipModelLevel();
            ConfigureClipRunnerFromEnv(runner, defaultEnableDebugDump: false);
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[CLIP-DIR-PROGRESS] progress=" + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));

            using var sw = new StreamWriter(summaryPath, false);
            sw.WriteLine("image\tstatus\telapsed_ms\tbest_label\tbest_prob\ttop3\terror\tgpu_summary\trt_count\tmanaged_mb\tgfx_driver_mb\tdump");

            for (var i = 0; i < files.Count; i++)
            {
                var path = files[i];
                Texture2D tex = null;
                try
                {
                    tex = LoadTexture(path);
                    if (tex == null)
                    {
                        sw.WriteLine(EscapeTsv(path) + "\tload_failed\t0\t\t0\t\tload_failed\t\t0\t0\t0\t");
                        continue;
                    }

                    var result = await runner.ProcessAsync(tex, CancellationToken.None);
                    var top3 = FormatClipTopScores(result.scores, 3);
                    var gpuSummary = NcnnCompute.NcnnGpuResourceTracker.BuildSummary();
                    var rtCount = Resources.FindObjectsOfTypeAll<RenderTexture>().Length;
                    var managedMb = Profiler.GetTotalAllocatedMemoryLong() / (1024f * 1024f);
                    var gfxMb = GetGraphicsDriverMemoryMb();
                    var status = string.IsNullOrWhiteSpace(result.error) ? "ok" : "error";
                    sw.WriteLine(
                        EscapeTsv(path) + "\t"
                        + status + "\t"
                        + result.elapsedMs.ToString(CultureInfo.InvariantCulture) + "\t"
                        + EscapeTsv(result.bestLabel ?? "") + "\t"
                        + result.bestProbability.ToString("0.000000", CultureInfo.InvariantCulture) + "\t"
                        + EscapeTsv(top3) + "\t"
                        + EscapeTsv(result.error ?? "") + "\t"
                        + EscapeTsv(gpuSummary) + "\t"
                        + rtCount.ToString(CultureInfo.InvariantCulture) + "\t"
                        + managedMb.ToString("0.000", CultureInfo.InvariantCulture) + "\t"
                        + gfxMb.ToString("0.000", CultureInfo.InvariantCulture) + "\t"
                        + EscapeTsv(runner.LastDumpDir ?? ""));
                    sw.Flush();

                    Debug.Log("[CLIP-DIR] " + (i + 1) + "/" + files.Count
                        + " | " + path
                        + " | status=" + status
                        + " | error=" + EscapeTsv(result.error ?? "")
                        + " | best=" + (result.bestLabel ?? "")
                        + " | prob=" + result.bestProbability.ToString("0.000000", CultureInfo.InvariantCulture)
                        + " | elapsedMs=" + result.elapsedMs
                        + " | gpu=" + gpuSummary);
                }
                finally
                {
                    if (tex != null)
                        UnityEngine.Object.DestroyImmediate(tex);
                }
            }

            NcnnCompute.NcnnGpuResourceTracker.WriteReport(outputDir);
            Debug.Log("[CLIP-DIR] summary=" + summaryPath);
        }
        finally
        {
            NcnnCompute.NcnnGpuResourceTracker.Enabled = false;
            UnityEngine.Object.DestroyImmediate(go);
        }
    }

    private static void ConfigureClipRunnerFromEnv(ClipNcnnReproRunner runner, bool defaultEnableDebugDump)
    {
        if (runner == null)
            return;

        var pack4OnlyGuard = ResolveBoolEnv(ClipPack4OnlyGuardEnvVar, false);
        runner.enableDebugDump = ResolveBoolEnv(ClipEnableDumpEnvVar, defaultEnableDebugDump);
        runner.forceFullRenderTexturePath = ResolveBoolEnv(ClipForceFullRtEnvVar, runner.forceFullRenderTexturePath)
            || pack4OnlyGuard;
        runner.useCommandBuffer = ResolveBoolEnv(ClipUseCommandBufferEnvVar, runner.useCommandBuffer);
        runner.useAsyncComputeCommandBuffer = ResolveBoolEnv(ClipUseAsyncComputeEnvVar, runner.useAsyncComputeCommandBuffer);
        runner.enableGeneralTextureConvolution = ResolveBoolEnv(
            ClipEnableGeneralTexConvEnvVar,
            runner.enableGeneralTextureConvolution || runner.forceFullRenderTexturePath)
            || pack4OnlyGuard;
        runner.enableAttentionMatMulPack4Specializations = ResolveBoolEnv(
            ClipEnableAttentionMatMulPack4EnvVar,
            runner.enableAttentionMatMulPack4Specializations || runner.forceFullRenderTexturePath)
            || pack4OnlyGuard;

        runner.disallowBufferAccess = pack4OnlyGuard;
        runner.disallowBufferOutputs = pack4OnlyGuard;
        runner.disallowBufferToTextureMaterialization = pack4OnlyGuard;

        runner.logAllLayerHeartbeats = ResolveBoolEnv(ClipLogAllLayerHeartbeatsEnvVar, false);
        runner.logAllLayerOutputs = ResolveBoolEnv(ClipLogAllLayerOutputsEnvVar, false);
        runner.logAllBufferMaterialize = ResolveBoolEnv(ClipLogAllBufferMaterializeEnvVar, false);
        runner.enableLayerRuntimeProfile = ResolveBoolEnv(ClipEnableLayerRuntimeProfileEnvVar, false);
        runner.layerRuntimeProfileSyncGpu = ResolveBoolEnv(ClipLayerRuntimeProfileSyncGpuEnvVar, false);
        runner.forwardReproDebugLogToUnity = ResolveBoolEnv(
            ClipEnableLayerPathLogEnvVar,
            pack4OnlyGuard
                || runner.logAllLayerHeartbeats
                || runner.logAllLayerOutputs
                || runner.logAllBufferMaterialize);
    }

    private static async UniTask RunCodeFormerDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultCodeFormerDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
            throw new InvalidOperationException("Failed to load debug input: " + inputPath);

        var go = new GameObject("CodeFormerDebugRunner");
        try
        {
            var runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
            runner.enableDebugDump = ResolveBoolEnv(CodeFormerEnableDumpEnvVar, false);
            runner.enableFaceRegionDebugDump = ResolveBoolEnv(CodeFormerEnableFaceDumpEnvVar, runner.enableDebugDump);
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[CodeFormer Progress] " + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("CodeFormer Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs + " | dump=" + (runner.LastDumpDir ?? ""));
            if (result.texture != null)
            {
                TryWriteTexturePng(result.texture, runner.LastDumpDir, "17_full_output.png");
                UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    private static async UniTask RunGfpganDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultCodeFormerDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
            throw new InvalidOperationException("Failed to load debug input: " + inputPath);

        var go = new GameObject("GfpganDebugRunner");
        try
        {
            var runner = go.AddComponent<GfpganNcnnReproRunner>();
            runner.enableFaceRegionDebugDump = true;
            ApplyGfpganPack4GuardFromEnv(runner);
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[GFPGAN Progress] " + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("GFPGAN Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs);
            if (result.texture != null)
            {
                var dir = CreateGenericDumpDir("AIImage_GfpganRepro");
                TryWriteTexturePng(result.texture, dir, "17_full_output.png");
                UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    private static async UniTask RunCodeFormerStressInternal()
    {
        var inputPaths = ResolveStressInputPaths(DefaultCodeFormerDebugImagePath);
        if (inputPaths.Count == 0)
            throw new InvalidOperationException("No stress inputs resolved");

        var iterations = ResolveStressCount(inputPaths.Count);
        var dumpDir = CreateGenericDumpDir("AIImage_CodeFormerStress");
        var logPath = Path.Combine(dumpDir, "stress_summary.txt");
        var lines = new List<string>(iterations + 8)
        {
            "iterations=" + iterations.ToString(CultureInfo.InvariantCulture),
            "inputs=" + string.Join(" | ", inputPaths)
        };

        NcnnCompute.NcnnGpuResourceTracker.Enabled = true;
        NcnnCompute.NcnnGpuResourceTracker.Reset("CodeFormerStress");

        var go = new GameObject("CodeFormerStressRunner");
        try
        {
            var runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
            runner.enableDebugDump = false;
            runner.enableFaceRegionDebugDump = false;

            for (var i = 0; i < iterations; i++)
            {
                var inputPath = inputPaths[i % inputPaths.Count];
                var tex = LoadTexture(inputPath);
                if (tex == null)
                    throw new InvalidOperationException("Failed to load stress input: " + inputPath);

                try
                {
                    var sw = Stopwatch.StartNew();
                    var result = await runner.ProcessAsync(tex, CancellationToken.None);
                    sw.Stop();

                    var privateMb = Process.GetCurrentProcess().PrivateMemorySize64 / (1024.0 * 1024.0);
                    var managedMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
                    var gfxMb = Profiler.GetAllocatedMemoryForGraphicsDriver() / (1024.0 * 1024.0);
                    lines.Add(
                        "iter=" + (i + 1).ToString(CultureInfo.InvariantCulture)
                        + " | file=" + Path.GetFileName(inputPath)
                        + " | elapsed_ms=" + sw.ElapsedMilliseconds.ToString(CultureInfo.InvariantCulture)
                        + " | err=" + (result.error ?? "")
                        + " | private_mb=" + privateMb.ToString("F3", CultureInfo.InvariantCulture)
                        + " | managed_mb=" + managedMb.ToString("F3", CultureInfo.InvariantCulture)
                        + " | gfx_mb=" + gfxMb.ToString("F3", CultureInfo.InvariantCulture)
                        + " | " + NcnnCompute.NcnnGpuResourceTracker.BuildSummary());

                    if (result.texture != null)
                        UnityEngine.Object.DestroyImmediate(result.texture);
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(tex);
                }
            }
        }
        finally
        {
            try
            {
                NcnnCompute.NcnnGpuResourceTracker.WriteReport(dumpDir, "stress_gpu_resources.txt");
            }
            catch
            {
            }

            try
            {
                File.WriteAllLines(logPath, lines);
            }
            catch
            {
            }

            NcnnCompute.NcnnGpuResourceTracker.Enabled = false;
            UnityEngine.Object.DestroyImmediate(go);
        }
    }

    private static async UniTask RunReproSuiteStressInternal()
    {
        var inputPath = ResolveInputPath(DefaultReproStressImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
            throw new InvalidOperationException("Failed to load repro stress input: " + inputPath);

        var iterations = ResolvePositiveIntEnv(StressCountEnvVar, 5);
        var enableTempPool = ResolveBoolEnv(ReproTempPoolEnvVar, true);
        var dumpDir = CreateGenericDumpDir("AIImage_ReproSuiteStress");
        var summaryPath = Path.Combine(dumpDir, "suite_summary.txt");
        var lines = new List<string>(256)
        {
            "input=" + inputPath,
            "iterations=" + iterations.ToString(CultureInfo.InvariantCulture),
            "enable_temp_pool=" + enableTempPool,
            "started_at=" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
            string.Empty
        };
        var failures = new List<string>();

        NcnnCompute.NcnnGpuResourceTracker.Enabled = true;
        try
        {
            await RunRealEsrganStressAsync(tex, iterations, enableTempPool, dumpDir, lines, failures);
            await RunYoloSegStressAsync(tex, iterations, enableTempPool, dumpDir, lines, failures);
            await RunMattingStressAsync(tex, iterations, enableTempPool, dumpDir, lines, failures);
            await RunGfpganStressAsync(tex, iterations, enableTempPool, dumpDir, lines, failures);
            await RunCodeFormerStressAsync(tex, iterations, enableTempPool, dumpDir, lines, failures);
        }
        finally
        {
            lines.Add(string.Empty);
            lines.Add("finished_at=" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
            if (failures.Count > 0)
                lines.Add("failures=" + string.Join(" | ", failures));

            try { File.WriteAllLines(summaryPath, lines); } catch { }

            NcnnCompute.NcnnGpuResourceTracker.Enabled = false;
            UnityEngine.Object.DestroyImmediate(tex);
        }

        if (failures.Count > 0)
            throw new InvalidOperationException("Repro suite stress had failures: " + string.Join(" | ", failures));
    }

    private sealed class RealEsrganValidationRunOptions
    {
        public string modelName;
        public bool useCommandBuffer;
        public bool pack4OnlyGuard;
        public bool reusePersistentRunner;
    }

    private static async UniTask RunRealEsrganValidationInternal()
    {
        var inputPaths = ResolveRealEsrganValidationInputPaths();
        var models = ResolveRealEsrganValidationModels();
        var maxElapsedMs = ResolvePositiveIntEnv(RealEsrganMaxElapsedMsEnvVar, 20000);
        var compareThreshold = ResolveFloatEnvOrDefault(RealEsrganCompareThresholdEnvVar, 0f);
        var forcedUseCommandBuffer = ResolveOptionalBoolEnv(RealEsrganUseCommandBufferEnvVar);
        var runImmediate = !forcedUseCommandBuffer.HasValue || !forcedUseCommandBuffer.Value;
        var runCommandBuffer = !forcedUseCommandBuffer.HasValue || forcedUseCommandBuffer.Value;
        var pack4OnlyGuard = ResolveBoolEnv(RealEsrganPack4OnlyGuardEnvVar, false);
        var reuseRunner = ResolveBoolEnv(RealEsrganValidationReuseRunnerEnvVar, false);
        var pathKind = pack4OnlyGuard ? "pack4_rt" : "default";
        var dumpDir = CreateGenericDumpDir("AIImage_RealEsrganValidation");
        var summaryPath = Path.Combine(dumpDir, "summary.tsv");
        var failures = new List<string>();

        using var sw = new StreamWriter(summaryPath, false);
        sw.WriteLine("model\timage\tmode\tpath_kind\tstatus\telapsed_ms\twidth\theight\tmean_abs_rgb\tmax_abs_rgb\terror\toutput");

        if (reuseRunner)
        {
            await RunRealEsrganValidationPersistentSequenceAsync(
                inputPaths,
                models,
                runImmediate,
                runCommandBuffer,
                pack4OnlyGuard,
                maxElapsedMs,
                compareThreshold,
                dumpDir,
                sw,
                failures);
        }
        else
        {
            for (var modelIndex = 0; modelIndex < models.Count; modelIndex++)
            {
                var model = models[modelIndex];
                for (var inputIndex = 0; inputIndex < inputPaths.Count; inputIndex++)
                {
                    var inputPath = inputPaths[inputIndex];
                    Texture2D input = null;
                    Texture2D immediateTex = null;
                    Texture2D cmdTex = null;
                    try
                    {
                        Debug.Log("[RealESRGAN-VALIDATION] model=" + model + " | input=" + inputPath + " | path_kind=" + pathKind);
                        input = LoadTexture(inputPath);
                        if (input == null)
                            throw new InvalidOperationException("Failed to load input: " + inputPath);

                        if (runImmediate)
                        {
                            var immediate = await RunRealEsrganSingleModeAsync(input, new RealEsrganValidationRunOptions
                            {
                                modelName = model,
                                useCommandBuffer = false,
                                pack4OnlyGuard = pack4OnlyGuard,
                                reusePersistentRunner = false
                            });
                            immediateTex = immediate.texture;
                            AppendRealEsrganValidationRow(
                                sw,
                                dumpDir,
                                model,
                                inputPath,
                                immediate,
                                "immediate",
                                pathKind,
                                "0",
                                "0");
                            sw.Flush();
                            AppendRealEsrganValidationFailure(failures, model, Path.GetFileName(inputPath), "immediate", immediate, maxElapsedMs);
                        }

                        if (runCommandBuffer)
                        {
                            var commandBuffer = await RunRealEsrganSingleModeAsync(input, new RealEsrganValidationRunOptions
                            {
                                modelName = model,
                                useCommandBuffer = true,
                                pack4OnlyGuard = pack4OnlyGuard,
                                reusePersistentRunner = false
                            });
                            cmdTex = commandBuffer.texture;
                            var meanAbs = 0f;
                            var maxAbs = 0;
                            if (immediateTex != null && cmdTex != null)
                                ComputeTextureDiff(immediateTex, cmdTex, out meanAbs, out maxAbs);

                            AppendRealEsrganValidationRow(
                                sw,
                                dumpDir,
                                model,
                                inputPath,
                                commandBuffer,
                                "command_buffer",
                                pathKind,
                                meanAbs.ToString("0.######", CultureInfo.InvariantCulture),
                                maxAbs.ToString(CultureInfo.InvariantCulture));
                            sw.Flush();
                            AppendRealEsrganValidationFailure(failures, model, Path.GetFileName(inputPath), "command_buffer", commandBuffer, maxElapsedMs);
                            if (immediateTex != null && cmdTex != null && meanAbs > compareThreshold)
                                failures.Add(model + " " + Path.GetFileName(inputPath) + " command_buffer diff mean_abs_rgb=" + meanAbs.ToString("0.######", CultureInfo.InvariantCulture));
                        }
                    }
                    finally
                    {
                        if (immediateTex != null)
                            UnityEngine.Object.DestroyImmediate(immediateTex);
                        if (cmdTex != null)
                            UnityEngine.Object.DestroyImmediate(cmdTex);
                        if (input != null)
                            UnityEngine.Object.DestroyImmediate(input);
                    }
                }
            }
        }

        Debug.Log("[RealESRGAN-VALIDATION] summary=" + summaryPath);
        if (failures.Count > 0)
            throw new InvalidOperationException("RealESRGAN validation failed: " + string.Join(" | ", failures));
    }

    private static async UniTask RunRealEsrganValidationPersistentSequenceAsync(
        IReadOnlyList<string> inputPaths,
        IReadOnlyList<string> models,
        bool runImmediate,
        bool runCommandBuffer,
        bool pack4OnlyGuard,
        int maxElapsedMs,
        float compareThreshold,
        string dumpDir,
        StreamWriter sw,
        List<string> failures)
    {
        var pathKind = pack4OnlyGuard ? "pack4_rt" : "default";
        var go = new GameObject("RealEsrganValidationPersistentSequence");
        try
        {
            var runner = go.AddComponent<RealEsrganNcnnReproRunner>();
            ConfigureRealEsrganRunner(runner, new RealEsrganValidationRunOptions
            {
                modelName = models.Count > 0 ? models[0] : "realesrgan-x4plus",
                useCommandBuffer = false,
                pack4OnlyGuard = pack4OnlyGuard,
                reusePersistentRunner = true
            });

            for (var modelIndex = 0; modelIndex < models.Count; modelIndex++)
            {
                var model = models[modelIndex];
                runner.modelName = model;

                for (var inputIndex = 0; inputIndex < inputPaths.Count; inputIndex++)
                {
                    var inputPath = inputPaths[inputIndex];
                    Texture2D input = null;
                    Texture2D immediateTex = null;
                    Texture2D cmdTex = null;
                    try
                    {
                        Debug.Log("[RealESRGAN-VALIDATION] persistent_sequence model=" + model + " | input=" + inputPath + " | path_kind=" + pathKind);
                        input = LoadTexture(inputPath);
                        if (input == null)
                            throw new InvalidOperationException("Failed to load input: " + inputPath);

                        if (runImmediate)
                        {
                            runner.modelName = model;
                            runner.useCommandBuffer = false;
                            var immediate = await RunRealEsrganSingleModeAsync(input, runner, new RealEsrganValidationRunOptions
                            {
                                modelName = model,
                                useCommandBuffer = false,
                                pack4OnlyGuard = pack4OnlyGuard,
                                reusePersistentRunner = true
                            });
                            immediateTex = immediate.texture;
                            AppendRealEsrganValidationRow(
                                sw,
                                dumpDir,
                                model,
                                inputPath,
                                immediate,
                                "immediate",
                                pathKind,
                                "0",
                                "0");
                            sw.Flush();
                            AppendRealEsrganValidationFailure(failures, model, Path.GetFileName(inputPath), "immediate", immediate, maxElapsedMs);
                        }

                        if (runCommandBuffer)
                        {
                            runner.modelName = model;
                            runner.useCommandBuffer = true;
                            var commandBuffer = await RunRealEsrganSingleModeAsync(input, runner, new RealEsrganValidationRunOptions
                            {
                                modelName = model,
                                useCommandBuffer = true,
                                pack4OnlyGuard = pack4OnlyGuard,
                                reusePersistentRunner = true
                            });
                            cmdTex = commandBuffer.texture;
                            var meanAbs = 0f;
                            var maxAbs = 0;
                            if (immediateTex != null && cmdTex != null)
                                ComputeTextureDiff(immediateTex, cmdTex, out meanAbs, out maxAbs);

                            AppendRealEsrganValidationRow(
                                sw,
                                dumpDir,
                                model,
                                inputPath,
                                commandBuffer,
                                "command_buffer",
                                pathKind,
                                meanAbs.ToString("0.######", CultureInfo.InvariantCulture),
                                maxAbs.ToString(CultureInfo.InvariantCulture));
                            sw.Flush();
                            AppendRealEsrganValidationFailure(failures, model, Path.GetFileName(inputPath), "command_buffer", commandBuffer, maxElapsedMs);
                            if (immediateTex != null && cmdTex != null && meanAbs > compareThreshold)
                                failures.Add(model + " " + Path.GetFileName(inputPath) + " command_buffer diff mean_abs_rgb=" + meanAbs.ToString("0.######", CultureInfo.InvariantCulture));
                        }
                    }
                    finally
                    {
                        if (immediateTex != null)
                            UnityEngine.Object.DestroyImmediate(immediateTex);
                        if (cmdTex != null)
                            UnityEngine.Object.DestroyImmediate(cmdTex);
                        if (input != null)
                            UnityEngine.Object.DestroyImmediate(input);
                    }
                }
            }

            TryInvokeReleaseMethod(runner);
            await YieldIfNeeded();
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            await ReleaseGpuPressureAsync();
        }
    }

    private static async UniTask<RealEsrganResult> RunRealEsrganSingleModeAsync(Texture2D input, RealEsrganValidationRunOptions options)
    {
        var go = new GameObject(options != null && options.useCommandBuffer ? "RealEsrganValidationCmd" : "RealEsrganValidationImmediate");
        try
        {
            var runner = go.AddComponent<RealEsrganNcnnReproRunner>();
            return await RunRealEsrganSingleModeAsync(input, runner, options);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            await ReleaseGpuPressureAsync();
        }
    }

    private static async UniTask<RealEsrganResult> RunRealEsrganSingleModeAsync(Texture2D input, RealEsrganNcnnReproRunner runner, RealEsrganValidationRunOptions options)
    {
        ConfigureRealEsrganRunner(runner, options);
        var result = await runner.ProcessAsync(input, CancellationToken.None);
        if (options == null || !options.reusePersistentRunner)
        {
            TryInvokeReleaseMethod(runner);
            await YieldIfNeeded();
        }
        return result;
    }

    private static void ConfigureRealEsrganRunner(RealEsrganNcnnReproRunner runner, RealEsrganValidationRunOptions options)
    {
        if (runner == null)
            return;

        runner.enableGpuLayerProfiling = false;
        runner.enableTempPool = true;
        runner.maxPooledPerShape = 4;
        runner.enableLayerRuntimeProfile = Application.isBatchMode && (options == null || !options.useCommandBuffer);
        runner.syncLayerRuntimeProfile = false;

        var modelName = options?.modelName;
        if (!string.IsNullOrWhiteSpace(modelName))
            runner.modelName = modelName.Trim();

        runner.useCommandBuffer = options != null && options.useCommandBuffer;

        var pack4OnlyGuard = options != null && options.pack4OnlyGuard;
        runner.disallowBufferAccess = pack4OnlyGuard;
        runner.disallowBufferOutputs = pack4OnlyGuard;
        runner.disallowBufferToTextureMaterialization = pack4OnlyGuard;
    }

    private static void AppendRealEsrganValidationFailure(List<string> failures, string model, string imageName, string mode, RealEsrganResult result, int maxElapsedMs)
    {
        if (failures == null)
            return;

        if (!string.IsNullOrWhiteSpace(result.error))
            failures.Add(model + " " + imageName + " " + mode + ": " + result.error);
        else if (result.elapsedMs > maxElapsedMs)
            failures.Add(model + " " + imageName + " " + mode + " exceeded " + maxElapsedMs + " ms: " + result.elapsedMs);
    }

    private static void AppendRealEsrganValidationRow(
        StreamWriter sw,
        string dumpDir,
        string model,
        string inputPath,
        RealEsrganResult result,
        string mode,
        string pathKind,
        string meanAbsText,
        string maxAbsText)
    {
        var outputName = BuildRealEsrganValidationOutputName(model, inputPath, mode);
        TryWriteTexturePng(result.texture, dumpDir, outputName);
        sw.WriteLine(
            EscapeTsv(model) + "\t"
            + EscapeTsv(Path.GetFileName(inputPath)) + "\t"
            + mode + "\t"
            + pathKind + "\t"
            + (string.IsNullOrWhiteSpace(result.error) ? "ok" : "error") + "\t"
            + result.elapsedMs.ToString(CultureInfo.InvariantCulture) + "\t"
            + (result.texture != null ? result.texture.width.ToString(CultureInfo.InvariantCulture) : "0") + "\t"
            + (result.texture != null ? result.texture.height.ToString(CultureInfo.InvariantCulture) : "0") + "\t"
            + meanAbsText + "\t"
            + maxAbsText + "\t"
            + EscapeTsv(result.error ?? string.Empty) + "\t"
            + EscapeTsv(Path.Combine(dumpDir, outputName)));
    }

    private static string BuildRealEsrganValidationOutputName(string model, string inputPath, string mode)
    {
        var modelSafe = SanitizeFileNameToken(model);
        var inputSafe = SanitizeFileNameToken(Path.GetFileNameWithoutExtension(inputPath));
        var modeSuffix = string.Equals(mode, "command_buffer", StringComparison.Ordinal) ? "cmd" : "immediate";
        return modelSafe + "_" + inputSafe + "_" + modeSuffix + ".png";
    }

    private static async UniTask RunRealEsrganStressAsync(Texture2D tex, int iterations, bool enableTempPool, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "ESRGAN";
        NcnnCompute.NcnnGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        RealEsrganNcnnReproRunner runner = null;
        try
        {
            runner = go.AddComponent<RealEsrganNcnnReproRunner>();
            runner.enableTempPool = enableTempPool;
            runner.maxPooledPerShape = enableTempPool ? 4 : 0;
            runner.enableGpuLayerProfiling = false;
            runner.useCommandBuffer = false;

            for (var i = 0; i < iterations; i++)
            {
                var result = await runner.ProcessAsync(tex, CancellationToken.None);
                AppendStressMetrics(lines, runnerName, "iter", i + 1, result.elapsedMs, result.error, result.workDir);
                if (result.texture != null)
                    UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        catch (Exception e)
        {
            failures.Add(runnerName + ": " + e.Message);
            AppendStressMetrics(lines, runnerName, "fatal", 0, 0, e.Message, null);
        }
        finally
        {
            TryInvokeReleaseMethod(runner);
            UnityEngine.Object.DestroyImmediate(go);
            await UniTask.Yield();
            AppendStressMetrics(lines, runnerName, "post_destroy", 0, 0, null, null);
            try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(dumpDir, "esrgan_gpu_resources.txt"); } catch { }
        }
    }

    private static async UniTask RunYoloSegStressAsync(Texture2D tex, int iterations, bool enableTempPool, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "YoloSeg";
        NcnnCompute.NcnnGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        YoloSegNcnnReproRunner runner = null;
        try
        {
            runner = go.AddComponent<YoloSegNcnnReproRunner>();
            runner.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
            runner.enableTempPool = enableTempPool;
            runner.maxPooledPerShape = enableTempPool ? 4 : 0;
            runner.enableDebugDump = false;
            runner.targetPersonOnly = true;
            runner.enableMaskClose = true;
            runner.enableMaskDilate = true;

            for (var i = 0; i < iterations; i++)
            {
                var result = await runner.ProcessAsync(tex, CancellationToken.None);
                var extra = "persons=" + result.personCount.ToString(CultureInfo.InvariantCulture)
                    + " | coverage=" + result.maskCoverage01.ToString("0.000000", CultureInfo.InvariantCulture);
                AppendStressMetrics(lines, runnerName, "iter", i + 1, result.elapsedMs, result.error, extra);
                if (result.texture != null)
                    UnityEngine.Object.DestroyImmediate(result.texture);
                if (result.mask != null)
                    UnityEngine.Object.DestroyImmediate(result.mask);
                if (result.overlay != null)
                    UnityEngine.Object.DestroyImmediate(result.overlay);
            }
        }
        catch (Exception e)
        {
            failures.Add(runnerName + ": " + e.Message);
            AppendStressMetrics(lines, runnerName, "fatal", 0, 0, e.Message, null);
        }
        finally
        {
            TryInvokeReleaseMethod(runner);
            UnityEngine.Object.DestroyImmediate(go);
            await UniTask.Yield();
            AppendStressMetrics(lines, runnerName, "post_destroy", 0, 0, null, null);
            try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(dumpDir, "yoloseg_gpu_resources.txt"); } catch { }
        }
    }

    private static async UniTask RunMattingStressAsync(Texture2D tex, int iterations, bool enableTempPool, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "Matting";
        NcnnCompute.NcnnGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        MatterNcnnReproRunner runner = null;
        try
        {
            runner = go.AddComponent<MatterNcnnReproRunner>();
            runner.enableTempPool = enableTempPool;
            runner.maxPooledPerShape = enableTempPool ? 4 : 0;
            runner.enableDebugDump = false;
            runner.forceBufferConvolution = false;

            for (var i = 0; i < iterations; i++)
            {
                var result = await runner.ProcessAsync(tex, CancellationToken.None);
                AppendStressMetrics(lines, runnerName, "iter", i + 1, result.elapsedMs, result.error, runner.LastDumpDir);
                if (result.texture != null)
                    UnityEngine.Object.DestroyImmediate(result.texture);
                if (result.matte != null)
                    UnityEngine.Object.DestroyImmediate(result.matte);
            }
        }
        catch (Exception e)
        {
            failures.Add(runnerName + ": " + e.Message);
            AppendStressMetrics(lines, runnerName, "fatal", 0, 0, e.Message, null);
        }
        finally
        {
            TryInvokeReleaseMethod(runner);
            UnityEngine.Object.DestroyImmediate(go);
            await UniTask.Yield();
            AppendStressMetrics(lines, runnerName, "post_destroy", 0, 0, null, null);
            try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(dumpDir, "matting_gpu_resources.txt"); } catch { }
        }
    }

    private static async UniTask RunGfpganStressAsync(Texture2D tex, int iterations, bool enableTempPool, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "GFPGAN";
        NcnnCompute.NcnnGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        GfpganNcnnReproRunner runner = null;
        try
        {
            runner = go.AddComponent<GfpganNcnnReproRunner>();
            runner.enableTempPool = enableTempPool;
            runner.maxPooledPerShape = enableTempPool ? 2 : 0;
            runner.enableFaceRegionDebugDump = false;

            for (var i = 0; i < iterations; i++)
            {
                var result = await runner.ProcessAsync(tex, CancellationToken.None);
                AppendStressMetrics(lines, runnerName, "iter", i + 1, result.elapsedMs, result.error, null);
                if (result.texture != null)
                    UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        catch (Exception e)
        {
            failures.Add(runnerName + ": " + e.Message);
            AppendStressMetrics(lines, runnerName, "fatal", 0, 0, e.Message, null);
        }
        finally
        {
            TryInvokeReleaseMethod(runner);
            UnityEngine.Object.DestroyImmediate(go);
            await UniTask.Yield();
            AppendStressMetrics(lines, runnerName, "post_destroy", 0, 0, null, null);
            try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(dumpDir, "gfpgan_gpu_resources.txt"); } catch { }
        }
    }

    private static async UniTask RunCodeFormerStressAsync(Texture2D tex, int iterations, bool enableTempPool, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "CodeFormer";
        NcnnCompute.NcnnGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        CodeFormerNcnnReproRunner2 runner = null;
        try
        {
            runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
            runner.enableTempPool = enableTempPool;
            runner.maxPooledPerShape = enableTempPool ? 2 : 0;
            runner.enableDebugDump = false;
            runner.enableFaceRegionDebugDump = false;

            for (var i = 0; i < iterations; i++)
            {
                var result = await runner.ProcessAsync(tex, CancellationToken.None);
                AppendStressMetrics(lines, runnerName, "iter", i + 1, result.elapsedMs, result.error, runner.LastDumpDir);
                if (result.texture != null)
                    UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        catch (Exception e)
        {
            failures.Add(runnerName + ": " + e.Message);
            AppendStressMetrics(lines, runnerName, "fatal", 0, 0, e.Message, null);
        }
        finally
        {
            TryInvokeReleaseMethod(runner);
            UnityEngine.Object.DestroyImmediate(go);
            await UniTask.Yield();
            AppendStressMetrics(lines, runnerName, "post_destroy", 0, 0, null, null);
            try { NcnnCompute.NcnnGpuResourceTracker.WriteReport(dumpDir, "codeformer_gpu_resources.txt"); } catch { }
        }
    }

    private static void AppendStressMetrics(List<string> lines, string runnerName, string phase, int iteration, long elapsedMs, string error, string extra)
    {
        var privateMb = Process.GetCurrentProcess().PrivateMemorySize64 / (1024.0 * 1024.0);
        var managedMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
        var gfxMb = GetGraphicsDriverMemoryMb();
        var rtCount = Resources.FindObjectsOfTypeAll<RenderTexture>().Length;
        var line =
            "runner=" + runnerName
            + " | phase=" + (phase ?? "")
            + " | iter=" + iteration.ToString(CultureInfo.InvariantCulture)
            + " | elapsed_ms=" + elapsedMs.ToString(CultureInfo.InvariantCulture)
            + " | err=" + (error ?? "")
            + " | private_mb=" + privateMb.ToString("F3", CultureInfo.InvariantCulture)
            + " | managed_mb=" + managedMb.ToString("F3", CultureInfo.InvariantCulture)
            + " | gfx_mb=" + gfxMb.ToString("F3", CultureInfo.InvariantCulture)
            + " | rt_objects=" + rtCount.ToString(CultureInfo.InvariantCulture)
            + " | " + NcnnCompute.NcnnGpuResourceTracker.BuildSummary();
        if (!string.IsNullOrWhiteSpace(extra))
            line += " | extra=" + extra.Replace("\r", " ").Replace("\n", " | ");
        lines.Add(line);
        try { Debug.Log("[REPRO-STRESS] " + line); } catch { }
    }

    private static void TryInvokeReleaseMethod(object target)
    {
        if (target == null)
            return;

        try
        {
            var method = target.GetType().GetMethod("Release", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public);
            if (method != null && method.GetParameters().Length == 0)
            {
                method.Invoke(target, null);
                return;
            }

            var reproField = target.GetType().GetField("_repro", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            var repro = reproField?.GetValue(target);
            if (repro == null)
                return;

            var reproRelease = repro.GetType().GetMethod("Release", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public);
            if (reproRelease != null && reproRelease.GetParameters().Length == 0)
                reproRelease.Invoke(repro, null);
        }
        catch
        {
        }
    }

    private static string ResolveInputPath(string fallbackPath)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(DebugInputEnvVar);
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env))
                return env;
        }
        catch
        {
        }
        return fallbackPath;
    }

    private static string ResolveClipInputDirectory()
    {
        try
        {
            var envDir = Environment.GetEnvironmentVariable(ClipInputDirEnvVar);
            if (!string.IsNullOrWhiteSpace(envDir) && Directory.Exists(envDir))
                return envDir;
        }
        catch
        {
        }

        var singlePath = ResolveInputPath(DefaultClipDebugImagePath);
        if (!string.IsNullOrWhiteSpace(singlePath))
        {
            var dir = Path.GetDirectoryName(singlePath);
            if (!string.IsNullOrWhiteSpace(dir) && Directory.Exists(dir))
                return dir;
        }

        return null;
    }

    private static string ResolveRealEsrganInputDirectory()
    {
        try
        {
            var envDir = Environment.GetEnvironmentVariable(RealEsrganInputDirEnvVar);
            if (!string.IsNullOrWhiteSpace(envDir) && Directory.Exists(envDir))
                return envDir;
        }
        catch
        {
        }

        return DefaultRealEsrganInputDir;
    }

    private static List<string> ResolveRealEsrganValidationInputPaths()
    {
        var inputDir = ResolveRealEsrganInputDirectory();
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
            throw new InvalidOperationException("RealESRGAN input dir not found: " + (inputDir ?? ""));

        var preferred = new List<string>
        {
            Path.Combine(inputDir, "02.png"),
            Path.Combine(inputDir, "03.jpg")
        };
        var resolved = new List<string>(preferred.Count);
        for (var i = 0; i < preferred.Count; i++)
        {
            if (File.Exists(preferred[i]))
                resolved.Add(preferred[i]);
        }

        if (resolved.Count == preferred.Count)
            return resolved;

        var fallback = EnumerateImageFilesRecursive(inputDir);
        if (fallback.Count == 0)
            throw new InvalidOperationException("RealESRGAN validation input missing under: " + inputDir);

        return fallback.Take(Mathf.Min(2, fallback.Count)).ToList();
    }

    private static List<string> ResolveRealEsrganValidationModels()
    {
        var models = new List<string>();

        try
        {
            var multi = Environment.GetEnvironmentVariable(RealEsrganModelsEnvVar);
            if (!string.IsNullOrWhiteSpace(multi))
            {
                var parts = multi.Split(new[] { ',', ';', '|', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                for (var i = 0; i < parts.Length; i++)
                {
                    var value = parts[i]?.Trim();
                    if (!string.IsNullOrWhiteSpace(value) && !models.Contains(value))
                        models.Add(value);
                }
            }
        }
        catch
        {
        }

        if (models.Count == 0)
        {
            try
            {
                var single = Environment.GetEnvironmentVariable(RealEsrganModelEnvVar);
                if (!string.IsNullOrWhiteSpace(single))
                {
                    var value = single.Trim();
                    if (!string.IsNullOrWhiteSpace(value))
                        models.Add(value);
                }
            }
            catch
            {
            }
        }

        if (models.Count == 0)
            models.Add("realesrgan-x4plus");

        return models;
    }

    private static string[] ResolveMonaiInputPaths()
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(MonaiInputPathsEnvVar);
            if (!string.IsNullOrWhiteSpace(env))
            {
                var parts = env.Split(new[] { '|', ';' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(part => part.Trim())
                    .Where(part => !string.IsNullOrWhiteSpace(part))
                    .ToArray();
                if (parts.Length > 0)
                    return parts;
            }
        }
        catch
        {
        }

        return new[] { DefaultMonaiInputPath };
    }

    private static string[] ResolveMonaiInputPathsForRun(string baselineManifestPath, bool useBaselineTensor)
    {
        if (EnvironmentVariableExists(MonaiInputPathsEnvVar))
            return ResolveMonaiInputPaths();

        if (useBaselineTensor)
        {
            var manifestPaths = TryReadMonaiInputPathsFromBaselineManifest(baselineManifestPath);
            if (manifestPaths != null && manifestPaths.Length > 0)
                return manifestPaths;
        }

        return ResolveMonaiInputPaths();
    }

    private static string[] TryReadMonaiInputPathsFromBaselineManifest(string baselineManifestPath)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(baselineManifestPath) || !File.Exists(baselineManifestPath))
                return null;

            var root = JObject.Parse(File.ReadAllText(baselineManifestPath, System.Text.Encoding.UTF8));
            var inputs = root["inputs"] as JArray;
            if (inputs == null || inputs.Count == 0)
                return null;

            var result = new List<string>(inputs.Count);
            for (var i = 0; i < inputs.Count; i++)
            {
                var path = inputs[i]?["path"]?.Value<string>();
                if (!string.IsNullOrWhiteSpace(path))
                    result.Add(path);
            }

            return result.Count > 0 ? result.ToArray() : null;
        }
        catch
        {
            return null;
        }
    }

    private static List<string> ResolveStressInputPaths(string fallbackPath)
    {
        try
        {
            var envDir = Environment.GetEnvironmentVariable(StressInputDirEnvVar);
            if (!string.IsNullOrWhiteSpace(envDir) && Directory.Exists(envDir))
            {
                var files = Directory.GetFiles(envDir)
                    .Where(IsImagePath)
                    .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                    .ToList();
                if (files.Count > 0)
                    return files;
            }
        }
        catch
        {
        }

        return new List<string> { ResolveInputPath(fallbackPath) };
    }

    private static List<string> EnumerateImageFilesRecursive(string rootDir)
    {
        var result = new List<string>();
        if (string.IsNullOrWhiteSpace(rootDir) || !Directory.Exists(rootDir))
            return result;

        var pending = new Stack<string>();
        pending.Push(rootDir);
        while (pending.Count > 0)
        {
            var dir = pending.Pop();
            try
            {
                foreach (var file in Directory.GetFiles(dir))
                {
                    if (IsImagePath(file))
                        result.Add(file);
                }
            }
            catch
            {
            }

            try
            {
                var subDirs = Directory.GetDirectories(dir);
                Array.Sort(subDirs, StringComparer.OrdinalIgnoreCase);
                for (var i = subDirs.Length - 1; i >= 0; i--)
                    pending.Push(subDirs[i]);
            }
            catch
            {
            }
        }

        result.Sort(StringComparer.OrdinalIgnoreCase);
        return result;
    }

    private static int ResolveStressCount(int inputCount)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(StressCountEnvVar);
            if (!string.IsNullOrWhiteSpace(env)
                && int.TryParse(env.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
                && parsed > 0)
                return parsed;
        }
        catch
        {
        }

        return Mathf.Max(60, inputCount);
    }

    private static int ResolvePositiveIntEnv(string envName, int fallback)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (!string.IsNullOrWhiteSpace(env)
                && int.TryParse(env.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
                && parsed > 0)
                return parsed;
        }
        catch
        {
        }

        return Mathf.Max(1, fallback);
    }

    private static bool? ResolveOptionalBoolEnv(string envName)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (string.IsNullOrWhiteSpace(env))
                return null;

            env = env.Trim();
            if (string.Equals(env, "1", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "yes", StringComparison.OrdinalIgnoreCase))
                return true;
            if (string.Equals(env, "0", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "false", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "no", StringComparison.OrdinalIgnoreCase))
                return false;
        }
        catch
        {
        }

        return null;
    }

    private static bool ResolveMonaiPatchInputMode(bool forceBufferAll, string rawMode)
    {
        if (!string.IsNullOrWhiteSpace(rawMode))
        {
            var mode = rawMode.Trim();
            if (string.Equals(mode, "compute_buffer", StringComparison.OrdinalIgnoreCase)
                || string.Equals(mode, "buffer", StringComparison.OrdinalIgnoreCase))
                return false;
            if (IsMonaiCommandBufferPatchInputMode(mode))
                return true;
            if (string.Equals(mode, "pack4_rt", StringComparison.OrdinalIgnoreCase)
                || string.Equals(mode, "rendertexture", StringComparison.OrdinalIgnoreCase)
                || string.Equals(mode, "texture", StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return !forceBufferAll;
    }

    private static bool IsMonaiCommandBufferPatchInputMode(string rawMode)
    {
        if (string.IsNullOrWhiteSpace(rawMode))
            return false;

        var mode = rawMode.Trim();
        return string.Equals(mode, "command_buffer_rt", StringComparison.OrdinalIgnoreCase)
            || string.Equals(mode, "command_buffer", StringComparison.OrdinalIgnoreCase)
            || string.Equals(mode, "cmd_rt", StringComparison.OrdinalIgnoreCase)
            || string.Equals(mode, "cmd", StringComparison.OrdinalIgnoreCase)
            || string.Equals(mode, "async_command_buffer", StringComparison.OrdinalIgnoreCase);
    }

    private static int ResolvePositiveIntEnvAllowZero(string envName, int fallback)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (!string.IsNullOrWhiteSpace(env)
                && int.TryParse(env.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
                && parsed >= 0)
                return parsed;
        }
        catch
        {
        }

        return Mathf.Max(0, fallback);
    }

    private static int ResolveIntEnvAllowZero(string envName, int fallback)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (!string.IsNullOrWhiteSpace(env)
                && int.TryParse(env.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
                return parsed;
        }
        catch
        {
        }

        return fallback;
    }

    private static ClipNcnnReproRunner.ClipModelLevel ResolveClipModelLevel()
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(ClipModelEnvVar);
            if (string.Equals(env, "B", StringComparison.OrdinalIgnoreCase))
                return ClipNcnnReproRunner.ClipModelLevel.B;
            if (string.Equals(env, "BLT", StringComparison.OrdinalIgnoreCase))
                return ClipNcnnReproRunner.ClipModelLevel.BLT;
            if (string.Equals(env, "S0", StringComparison.OrdinalIgnoreCase))
                return ClipNcnnReproRunner.ClipModelLevel.S0;
            if (string.Equals(env, "S2", StringComparison.OrdinalIgnoreCase))
                return ClipNcnnReproRunner.ClipModelLevel.S2;
        }
        catch
        {
        }

        return ClipNcnnReproRunner.ClipModelLevel.S0;
    }

    private static bool ResolveBoolEnv(string envName, bool fallback)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (string.IsNullOrWhiteSpace(env))
                return fallback;
            env = env.Trim();
            if (string.Equals(env, "1", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "yes", StringComparison.OrdinalIgnoreCase))
                return true;
            if (string.Equals(env, "0", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "false", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "no", StringComparison.OrdinalIgnoreCase))
                return false;
        }
        catch
        {
        }

        return fallback;
    }

    private static bool EnvironmentVariableExists(string envName)
    {
        try
        {
            return Environment.GetEnvironmentVariable(envName) != null;
        }
        catch
        {
            return false;
        }
    }

    private static float ResolveFloatEnvOrDefault(string envName, float fallback)
    {
        if (TryReadFloatEnv(envName, out var value))
            return value;
        return fallback;
    }

    private static RenderTextureFormat ResolveRenderTextureFormatEnv(string envName, RenderTextureFormat fallback)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (string.IsNullOrWhiteSpace(env))
                return fallback;

            env = env.Trim();
            if (string.Equals(env, "float", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "fp32", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "argbfloat", StringComparison.OrdinalIgnoreCase))
                return RenderTextureFormat.ARGBFloat;

            if (string.Equals(env, "half", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "fp16", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "argbhalf", StringComparison.OrdinalIgnoreCase))
                return RenderTextureFormat.ARGBHalf;

            if (Enum.TryParse(env, true, out RenderTextureFormat parsed))
                return parsed;
        }
        catch
        {
        }

        return fallback;
    }

    private static string ResolveStringEnv(string envName, string fallback)
    {
        try
        {
            if (ResolveBoolEnv(envName + "_EMPTY", false))
                return string.Empty;

            var env = Environment.GetEnvironmentVariable(envName);
            if (env != null)
                return env;
        }
        catch
        {
        }

        return fallback;
    }

    private static ISet<string> ResolveTokenSetEnv(string envName)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (string.IsNullOrWhiteSpace(env))
                return null;

            var parts = env.Split(new[] { ',', ';', '|', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            var set = new HashSet<string>(StringComparer.Ordinal);
            for (var i = 0; i < parts.Length; i++)
            {
                var value = parts[i]?.Trim();
                if (!string.IsNullOrWhiteSpace(value))
                    set.Add(value);
            }

            return set.Count > 0 ? set : null;
        }
        catch
        {
            return null;
        }
    }

    private static string ResolveOptionalExistingFile(string envName)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env))
                return env;
        }
        catch
        {
        }

        return null;
    }

    private static string ResolveOptionalExistingDirectory(string envName)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (!string.IsNullOrWhiteSpace(env) && Directory.Exists(env))
                return env;
        }
        catch
        {
        }

        return null;
    }

    private static string FormatClipTopScores(ClipLabelScore[] scores, int topN)
    {
        if (scores == null || scores.Length == 0 || topN <= 0)
            return string.Empty;

        var count = Mathf.Min(topN, scores.Length);
        var parts = new List<string>(count);
        for (var i = 0; i < count; i++)
        {
            var s = scores[i];
            parts.Add((s.label ?? "")
                + " "
                + s.probability.ToString("P1", CultureInfo.InvariantCulture));
        }
        return string.Join(", ", parts);
    }

    private static float GetGraphicsDriverMemoryMb()
    {
        try
        {
            return Profiler.GetAllocatedMemoryForGraphicsDriver() / (1024f * 1024f);
        }
        catch
        {
            return 0f;
        }
    }

    private static string EscapeTsv(string value)
    {
        if (string.IsNullOrEmpty(value))
            return string.Empty;
        return value.Replace("\t", "    ").Replace("\r", " ").Replace("\n", " | ");
    }

    private static string SanitizeFileNameToken(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "unnamed";

        var invalid = Path.GetInvalidFileNameChars();
        var chars = value.Trim().ToCharArray();
        for (var i = 0; i < chars.Length; i++)
        {
            if (Array.IndexOf(invalid, chars[i]) >= 0 || char.IsWhiteSpace(chars[i]))
                chars[i] = '_';
        }

        return new string(chars);
    }

    private static bool IsImagePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return false;

        var ext = Path.GetExtension(path);
        return string.Equals(ext, ".png", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ext, ".jpg", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ext, ".jpeg", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ext, ".bmp", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ext, ".webp", StringComparison.OrdinalIgnoreCase);
    }

    private static bool ResolveFacePreferTexturePath()
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(FaceBufferPathEnvVar);
            if (string.IsNullOrWhiteSpace(env))
                return true;
            env = env.Trim();
            if (string.Equals(env, "1", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "buffer", StringComparison.OrdinalIgnoreCase))
                return false;
            if (string.Equals(env, "0", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "false", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "texture", StringComparison.OrdinalIgnoreCase))
                return true;
            return true;
        }
        catch
        {
            return true;
        }
    }

    private static void ApplyFaceThresholdOverrides(NcnnFaceRegionGenerator face)
    {
        if (face == null)
            return;

        if (TryReadFloatEnv(FaceProbThresholdEnvVar, out var prob))
            face.probThreshold = Mathf.Clamp(prob, 0.01f, 0.99f);
        if (TryReadFloatEnv(FaceNmsThresholdEnvVar, out var nms))
            face.nmsThreshold = Mathf.Clamp01(nms);
    }

    private static void ApplyFacePack4GuardFromEnv(NcnnFaceRegionGenerator face)
    {
        if (face == null)
            return;

        var pack4OnlyGuard = ResolveBoolEnv(FacePack4OnlyGuardEnvVar, false);
        face.disallowBufferAccess = pack4OnlyGuard;
        face.disallowBufferOutputs = pack4OnlyGuard;
        face.disallowBufferToTextureMaterialization = pack4OnlyGuard;
    }

    private static void ApplyGfpganPack4GuardFromEnv(GfpganNcnnReproRunner runner)
    {
        if (runner == null)
            return;

        var pack4OnlyGuard = ResolveBoolEnv(GfpganPack4OnlyGuardEnvVar, false);
        runner.disallowBufferAccess = pack4OnlyGuard;
        runner.disallowBufferOutputs = pack4OnlyGuard;
        runner.disallowBufferToTextureMaterialization = pack4OnlyGuard;
    }

    private static bool TryReadFloatEnv(string envName, out float value)
    {
        value = 0f;
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (string.IsNullOrWhiteSpace(env))
                return false;
            return float.TryParse(env.Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out value);
        }
        catch
        {
            return false;
        }
    }

    private static float ComputeMaskedMeanAbsDiff(Texture2D source, Texture2D candidate, Texture2D mask, bool blackMaskMeansInpaint, out int maskedPixels)
    {
        maskedPixels = 0;
        if (source == null || candidate == null || mask == null)
            return 0f;
        if (source.width != candidate.width || source.height != candidate.height || source.width != mask.width || source.height != mask.height)
            return 0f;

        var srcPixels = source.GetPixels32();
        var dstPixels = candidate.GetPixels32();
        var maskPixels = mask.GetPixels32();
        double sumAbs = 0d;
        for (var i = 0; i < srcPixels.Length; i++)
        {
            var maskIsWhite = maskPixels[i].r >= 128 || maskPixels[i].g >= 128 || maskPixels[i].b >= 128;
            var include = blackMaskMeansInpaint ? !maskIsWhite : maskIsWhite;
            if (!include)
                continue;

            maskedPixels++;
            sumAbs += Mathf.Abs(srcPixels[i].r - dstPixels[i].r);
            sumAbs += Mathf.Abs(srcPixels[i].g - dstPixels[i].g);
            sumAbs += Mathf.Abs(srcPixels[i].b - dstPixels[i].b);
        }

        return maskedPixels > 0 ? (float)(sumAbs / (maskedPixels * 3d)) : 0f;
    }

    private static void TryWriteTexturePng(Texture2D texture, string dir, string fileName)
    {
        if (texture == null || string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(fileName))
            return;
        try
        {
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, fileName), texture.EncodeToPNG());
        }
        catch (Exception e)
        {
            Debug.LogWarning("Failed to write debug texture: " + e.Message);
        }
    }

    private static void TryCompareTextureWithReference(Texture2D candidate, string referencePath)
    {
        if (candidate == null || string.IsNullOrWhiteSpace(referencePath) || !File.Exists(referencePath))
            return;

        var reference = LoadTexture(referencePath);
        if (reference == null)
            return;

        try
        {
            if (candidate.width != reference.width || candidate.height != reference.height)
            {
                Debug.Log("Matting Debug compare skipped | size mismatch " + candidate.width + "x" + candidate.height + " vs " + reference.width + "x" + reference.height);
                return;
            }

            var a = candidate.GetPixels32();
            var b = reference.GetPixels32();
            double sumAbs = 0d;
            var maxAbs = 0;
            var count = a.Length * 3;
            for (var i = 0; i < a.Length; i++)
            {
                var dr = Mathf.Abs(a[i].r - b[i].r);
                var dg = Mathf.Abs(a[i].g - b[i].g);
                var db = Mathf.Abs(a[i].b - b[i].b);
                sumAbs += dr + dg + db;
                maxAbs = Mathf.Max(maxAbs, Mathf.Max(dr, Mathf.Max(dg, db)));
            }

            var meanAbs = count > 0 ? sumAbs / count : 0d;
            Debug.Log("Matting Debug compare | ref=" + referencePath + " | mean_abs_rgb=" + meanAbs.ToString("F4", CultureInfo.InvariantCulture) + " | max_abs_rgb=" + maxAbs.ToString(CultureInfo.InvariantCulture));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(reference);
        }
    }

    private static void ComputeTextureDiff(Texture2D a, Texture2D b, out float meanAbsRgb, out int maxAbsRgb)
    {
        meanAbsRgb = 0f;
        maxAbsRgb = 0;
        if (a == null || b == null || a.width != b.width || a.height != b.height)
            return;

        var pa = a.GetPixels32();
        var pb = b.GetPixels32();
        double sumAbs = 0d;
        var maxAbs = 0;
        for (var i = 0; i < pa.Length; i++)
        {
            var dr = Mathf.Abs(pa[i].r - pb[i].r);
            var dg = Mathf.Abs(pa[i].g - pb[i].g);
            var db = Mathf.Abs(pa[i].b - pb[i].b);
            sumAbs += dr + dg + db;
            maxAbs = Mathf.Max(maxAbs, Mathf.Max(dr, Mathf.Max(dg, db)));
        }

        meanAbsRgb = pa.Length > 0 ? (float)(sumAbs / (pa.Length * 3d)) : 0f;
        maxAbsRgb = maxAbs;
    }

    private static Texture2D BuildAbsDiffTexture(Texture2D a, Texture2D b)
    {
        if (a == null || b == null || a.width != b.width || a.height != b.height)
            return null;

        var pa = a.GetPixels32();
        var pb = b.GetPixels32();
        var diff = new Color32[pa.Length];
        for (var i = 0; i < pa.Length; i++)
        {
            diff[i] = new Color32(
                (byte)Mathf.Abs(pa[i].r - pb[i].r),
                (byte)Mathf.Abs(pa[i].g - pb[i].g),
                (byte)Mathf.Abs(pa[i].b - pb[i].b),
                255);
        }

        var tex = new Texture2D(a.width, a.height, TextureFormat.RGBA32, false, true);
        tex.SetPixels32(diff);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
    }

    private static string CreateGenericDumpDir(string prefix)
    {
        var root = Path.Combine(Path.GetTempPath(), "YanQi", "AIImage");
        Directory.CreateDirectory(root);
        var dir = Path.Combine(root, prefix + "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        Directory.CreateDirectory(dir);
        return dir;
    }
    private static Texture2D LoadTexture(string path)
    {
        if (!File.Exists(path))
            return null;

        try
        {
            var bytes = File.ReadAllBytes(path);
            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
            if (!ImageConversion.LoadImage(tex, bytes, false))
            {
                UnityEngine.Object.DestroyImmediate(tex);
                return null;
            }
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            tex.name = Path.GetFileNameWithoutExtension(path);
            return tex;
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            return null;
        }
    }
}
#endif
