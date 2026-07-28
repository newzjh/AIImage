#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using Aexis.Samples.Async;
using AIImage.Qwen35;
using Aexis.Samples.Json.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Profiling;
using Aexis.Ncnn;
using Debug = UnityEngine.Debug;
using Aexis.Execution;

public static class NcnnDebugRunner
{
    [Serializable]
    private sealed class D1RuntimeBenchmarkReport
    {
        public string schemaVersion = "aiimage.inference.runtime-benchmark/v1";
        public string runner;
        public string modelId;
        public string manifestPath;
        public string activationDtype;
        public string weightDtype;
        public string quantizationVersion;
        public string calibrationVersion;
        public string[] quantizedOperators;
        public bool strictTexturePlan;
        public string status;
        public string error;
        public long elapsedMs;
        public long peakTemporaryTextureBytes;
        public long peakTextureBytes;
        public long peakBufferBytes;
        public long peakTotalBytes;
        public string taskMetricName;
        public string taskMetricValue;
        public string taskMetricDetail;
        public string debugDumpPath;
        public string graphicsDevice;
    }

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
    private const string ClipPrecisionEnvVar = "AIIMAGE_CLIP_PRECISION";
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
    private const string ClipVerifyCommandBufferParityEnvVar = "AIIMAGE_CLIP_VERIFY_COMMAND_BUFFER_PARITY";
    private const string ClipCommandBufferParityToleranceEnvVar = "AIIMAGE_CLIP_COMMAND_BUFFER_PARITY_TOLERANCE";
    private const string ClipCommandBufferParityProbeBlobEnvVar = "AIIMAGE_CLIP_COMMAND_BUFFER_PARITY_PROBE_BLOB";
    private const string CodeFormerEnableDumpEnvVar = "AIIMAGE_CODEFORMER_ENABLE_DUMP";
    private const string CodeFormerEnableFaceDumpEnvVar = "AIIMAGE_CODEFORMER_ENABLE_FACE_DUMP";
    private const string CodeFormerPrecisionEnvVar = "AIIMAGE_CODEFORMER_PRECISION";
    private const string YoloFlipYEnvVar = "AIIMAGE_YOLOSEG_FLIPY";
    private const string YoloForceBufferConvEnvVar = "AIIMAGE_YOLOSEG_FORCE_BUFFER_CONV";
    private const string YoloForceBufferBinaryEnvVar = "AIIMAGE_YOLOSEG_FORCE_BUFFER_BINARY";
    private const string YoloUseArgbFloatEnvVar = "AIIMAGE_YOLOSEG_USE_ARGB_FLOAT";
    private const string YoloPrecisionEnvVar = "AIIMAGE_YOLOSEG_PRECISION";
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
    private const string MattingUseCommandBufferEnvVar = "AIIMAGE_MATTING_USE_COMMAND_BUFFER";
    private const string MattingUseAsyncComputeEnvVar = "AIIMAGE_MATTING_USE_ASYNC_COMPUTE";
    private const string MattingPrecisionEnvVar = "AIIMAGE_MATTING_PRECISION";
    private const string MattingFp32ReferenceEnvVar = "AIIMAGE_MATTING_FP32_REFERENCE";
    private const string MattingReferenceLabelEnvVar = "AIIMAGE_MATTING_REFERENCE_LABEL";
    private const string GfpganPack4OnlyGuardEnvVar = "AIIMAGE_GFPGAN_PACK4_ONLY_GUARD";
    private const string GfpganPrecisionEnvVar = "AIIMAGE_GFPGAN_PRECISION";
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
    private const string SdInpaintingPrecisionEnvVar = "AIIMAGE_SD_INPAINTING_PRECISION";
    private const string SdTensorFormatEnvVar = "AIIMAGE_SD_TENSOR_FORMAT";
    private const string SdDecoderTensorFormatEnvVar = "AIIMAGE_SD_DECODER_TENSOR_FORMAT";
    private const string SdEncoderTensorFormatEnvVar = "AIIMAGE_SD_ENCODER_TENSOR_FORMAT";
    private const string SdEnableDumpEnvVar = "AIIMAGE_SD_ENABLE_DUMP";
    private const string SdSyncStageTimingsEnvVar = "AIIMAGE_SD_SYNC_STAGE_TIMINGS";
    private const string SdKeepRawConvWeightsEnvVar = "AIIMAGE_SD_KEEP_RAW_CONV_WEIGHTS";
    private const string SdUseCommandBufferEnvVar = "AIIMAGE_SD_USE_COMMAND_BUFFER";
    private const string SdUseAsyncComputeEnvVar = "AIIMAGE_SD_USE_ASYNC_COMPUTE";
    private const string SdDisallowTempComputeBuffersEnvVar = "AIIMAGE_SD_DISALLOW_TEMP_COMPUTE_BUFFERS";
    private const string SdPack4OnlyGuardEnvVar = "AIIMAGE_SD_PACK4_ONLY_GUARD";
    private const string InpaintBackendEnvVar = "AIIMAGE_INPAINT_BACKEND";
    private const string DeepFillV2BackendEnvVar = "AIIMAGE_DEEPFILLV2_BACKEND";
    private const string DeepFillV2UseArgbFloatEnvVar = "AIIMAGE_DEEPFILLV2_USE_ARGB_FLOAT";
    private const string DeepFillV2FlipYInputEnvVar = "AIIMAGE_DEEPFILLV2_FLIPY_INPUT";
    private const string DeepFillV2FlipYOutputEnvVar = "AIIMAGE_DEEPFILLV2_FLIPY_OUTPUT";
    private const string DeepFillV2EnableLayerPathLogEnvVar = "AIIMAGE_DEEPFILLV2_ENABLE_LAYER_PATH_LOG";
    private const string DeepFillV2OnnxPathEnvVar = "AIIMAGE_DEEPFILLV2_ONNX_PATH";
    private const string DeepFillV2ParamPathEnvVar = "AIIMAGE_DEEPFILLV2_PARAM_PATH";
    private const string DeepFillV2BinPathEnvVar = "AIIMAGE_DEEPFILLV2_BIN_PATH";
    private const string DeepFillV2DebugTensorBlobEnvVar = "AIIMAGE_DEEPFILLV2_DEBUG_TENSOR_BLOB";
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
    private static readonly string DefaultFaceDebugImagePath = AexisApplicationExamplePaths.SampleTextureAbsolutePath("facedetail2.png");
    private static readonly string DefaultCodeFormerDebugImagePath = AexisApplicationExamplePaths.SampleTextureAbsolutePath("facedetail2.png");
    private static readonly string DefaultClipDebugImagePath = AexisApplicationExamplePaths.SampleTextureAbsolutePath("facedetail2.png");
    private static readonly string DefaultMattingDebugImagePath = AexisApplicationExamplePaths.SampleTextureAbsolutePath("facedetail3.jpg");
    private static readonly string DefaultMattingReferencePath = null;
    private static readonly string DefaultYoloSegDebugImagePath = AexisApplicationExamplePaths.SampleTextureAbsolutePath("facedetail2.png");
    private static readonly string DefaultReproStressImagePath = AexisApplicationExamplePaths.SampleTextureAbsolutePath("facedetail2.png");
    private static readonly string DefaultDeepFillV2DebugImagePath = AexisApplicationExamplePaths.SampleTextureAbsolutePath("facedetail3.jpg");
    private static readonly string DefaultDeepFillV2Case1ImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "deepfillv2", "deepfillv2-pytorch-master", "examples", "inpaint", "case1.png");
    private static readonly string DefaultDeepFillV2Case1MaskedPath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "deepfillv2", "deepfillv2-pytorch-master", "examples", "inpaint", "case1_masked.png");
    private static readonly string DefaultDeepFillV2Case1OutputPath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "deepfillv2", "deepfillv2-pytorch-master", "examples", "inpaint", "case1_out.png");
    private const double DeepFillV2Case1MaxFullMae = 1.25;
    private const double DeepFillV2Case1MaxMaskedMae = 14.0;
    private const int DeepFillV2Case1MaxAbs = 160;
    private static readonly string DefaultMonaiBaselineManifestPath = Path.Combine(Directory.GetCurrentDirectory(), "Tools", "MonaiToNCNN", "manual_test", "brats_mri_segmentation_baseline", "RegLib_C01_1", "baseline_manifest.json");
    private static readonly string DefaultMonaiInputPath = @"E:\Projects\CTData\sliceexampledata2\MRBrainTumor1\RegLib_C01_1.nrrd";
    private static readonly string DefaultVistaBaselineManifestPath = Path.Combine(Directory.GetCurrentDirectory(), "Tools", "MonaiToNCNN", "manual_test", "vista3d_ct_philips_heart_baseline", "ct_philips_heart", "baseline_manifest.json");
    private static readonly string DefaultVistaInputPath = @"E:\Projects\CTData\sliceexampledata2\CT_Philips\CT_Philips.nii.gz";
    private static readonly string DefaultRealEsrganInputDir = AexisApplicationExamplePaths.SampleTextureDirectory;
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
            case nameof(RunCodeFormerEncoderPack4RegressionBatch):
                RunCodeFormerEncoderPack4RegressionBatch();
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
            case nameof(RunSdInpaintingSampleMaskBatch):
                RunSdInpaintingSampleMaskBatch();
                return;
            case nameof(RunYoloAndDeepFillV2DebugBatch):
                RunYoloAndDeepFillV2DebugBatch();
                return;
            case nameof(RunDeepFillV2Case1DebugBatch):
                RunDeepFillV2Case1DebugBatch();
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
            case nameof(RunAexisStrictTextureValidationBatch):
                RunAexisStrictTextureValidationBatch();
                return;
            case nameof(RunMonaiDebugBatch):
                RunMonaiDebugBatch();
                return;
            case nameof(RunQwen35ContractBatch):
                RunQwen35ContractBatch();
                return;
            case nameof(RunQwen35NetworkLoadBatch):
                RunQwen35NetworkLoadBatch();
                return;
            case nameof(RunQwen35EmbedProbeBatch):
                RunQwen35EmbedProbeBatch();
                return;
            case nameof(RunQwen35TokenizerContractBatch):
                RunQwen35TokenizerContractBatch();
                return;
            case nameof(RunQwen35TextGenerationBatch):
                RunQwen35TextGenerationBatch();
                return;
            case nameof(RunQwen35MultimodalGenerationBatch):
                RunQwen35MultimodalGenerationBatch();
                return;
            case nameof(RunQwen35AsyncMultimodalGenerationBatch):
                RunQwen35AsyncMultimodalGenerationBatch();
                return;
            case nameof(RunQwen35DecoderPrefixProbeBatch):
                RunQwen35DecoderPrefixProbeBatch();
                return;
            case nameof(RunQwen35VisionPatchProbeBatch):
                RunQwen35VisionPatchProbeBatch();
                return;
            case nameof(RunQwen35VisionPositionProbeBatch):
                RunQwen35VisionPositionProbeBatch();
                return;
            case nameof(RunQwen35VisionPatchAtlasProbeBatch):
                RunQwen35VisionPatchAtlasProbeBatch();
                return;
            case nameof(RunQwen35VisionEncoderPrefixProbeBatch):
                RunQwen35VisionEncoderPrefixProbeBatch();
                return;
            case nameof(RunQwen35FullCheckpointAuditBatch):
                RunQwen35FullCheckpointAuditBatch();
                return;
            case nameof(RunDesignViewCompositeDebugBatch):
                RunDesignViewCompositeDebugBatch();
                return;
            default:
                throw new InvalidOperationException("Unknown batch method: " + methodName);
        }
    }

    [MenuItem("Aexis/Examples/Debug/Run NCNN Face Debug")]
    public static void RunFaceDebugMenu()
    {
        RunFaceDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run CodeFormer Debug")]
    public static void RunCodeFormerDebugMenu()
    {
        RunCodeFormerDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run CLIP Debug")]
    public static void RunClipDebugMenu()
    {
        RunClipDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run CLIP Directory Debug")]
    public static void RunClipDirectoryDebugMenu()
    {
        RunClipDirectoryDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run GFPGAN Debug")]
    public static void RunGfpganDebugMenu()
    {
        RunGfpganDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run Matting Debug")]
    public static void RunMattingDebugMenu()
    {
        RunMattingDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run YOLO Seg Debug")]
    public static void RunYoloSegDebugMenu()
    {
        RunYoloSegDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run YOLO + SD Inpainting Debug")]
    public static void RunYoloAndInpaintingDebugMenu()
    {
        RunYoloAndInpaintingDebug().Forget();
    }

    private static string ResolveQwen35ModelDirectory(string projectRoot, string fallbackModelDirectory)
    {
        var configured = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(configured))
            return Qwen35ModelDirectoryResolver.Resolve(configured, fallbackModelDirectory);
        if (Aexis.Samples.AexisSampleStreamingAssets.TryResolveDirectoryPath("QWEN35", out var streamingAssetsDirectory))
        {
            var candidate = Qwen35ModelDirectoryResolver.Resolve(streamingAssetsDirectory, fallbackModelDirectory);
            if (File.Exists(Path.Combine(candidate, "model.json")))
                return candidate;
        }
        return Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "_models", fallbackModelDirectory);
    }

    private static string ResolveQwen35ImagePath(string projectRoot)
    {
        var configured = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_IMAGE");
        if (!string.IsNullOrWhiteSpace(configured))
            return Path.GetFullPath(configured);

        var baselineImage = Path.Combine(projectRoot, "ref", "ncnn_llm-main", "test.jpg");
        if (File.Exists(baselineImage))
            return baselineImage;

        return Path.Combine(projectRoot, "ref", "03.jpg");
    }

    private static JArray ResolveQwen35ExpectedTokenIds(string modelDirectory)
    {
        var configured = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_EXPECTED_TOKEN_IDS");
        var expected = new JArray();
        if (!string.IsNullOrWhiteSpace(configured))
        {
            foreach (var value in configured.Split(','))
            {
                if (!int.TryParse(value.Trim(), out var tokenId))
                {
                    throw new ArgumentException(
                        "AIIMAGE_QWEN35_EXPECTED_TOKEN_IDS contains a non-integer token: " + value);
                }
                expected.Add(tokenId);
            }
            if (expected.Count == 0)
                throw new ArgumentException("AIIMAGE_QWEN35_EXPECTED_TOKEN_IDS did not contain any tokens.");
            return expected;
        }

        // Q4 quality must be verified against a caller-provided Q8 baseline.
        // Do not bake a previous corrupted token into the default acceptance path.
        return expected;
    }

    [MenuItem("Aexis/Examples/Debug/Run YOLO + DeepFillV2 Debug")]
    public static void RunYoloAndDeepFillV2DebugMenu()
    {
        RunYoloAndDeepFillV2DebugBatch();
    }

    [MenuItem("Aexis/Examples/Debug/Run Stable Diffusion Debug")]
    public static void RunStableDiffusionDebugMenu()
    {
        RunStableDiffusionDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run MONAI Debug")]
    public static void RunMonaiDebugMenu()
    {
        RunMonaiDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run DesignView Composite Debug")]
    public static void RunDesignViewCompositeDebugMenu()
    {
        RunDesignViewCompositeDebug().Forget();
    }

    [MenuItem("Aexis/Examples/Debug/Run CodeFormer Stress (60x)")]
    public static void RunCodeFormerStressMenu()
    {
        RunCodeFormerStressBatch();
    }

    [MenuItem("Aexis/Examples/Debug/Run Repro Suite Stress (02.png)")]
    public static void RunReproSuiteStressMenu()
    {
        RunReproSuiteStressBatch();
    }

    [MenuItem("Aexis/Examples/Debug/Run RealESRGAN Validation Batch")]
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
            Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
            Aexis.Execution.AexisGpuResourceTracker.Reset("D1.Clip");
            var runner = go.AddComponent<ClipNcnnReproRunner>();
            runner.modelLevel = ResolveClipModelLevel();
            ConfigureClipRunnerFromEnv(runner, defaultEnableDebugDump: true);
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[CLIP-DEBUG] progress=" + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            WriteD1RuntimeBenchmark(
                "clip",
                result.elapsedMs,
                result.error,
                runner.LastDumpDir,
                "top1_probability",
                result.bestProbability.ToString("0.000000", CultureInfo.InvariantCulture),
                result.bestLabel ?? string.Empty);
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
            var configuredPrecision = ResolveStringEnv(GfpganPrecisionEnvVar, "Auto");
            if (!Enum.TryParse(configuredPrecision, true, out Aexis.Execution.AexisPrecisionMode precisionMode))
                throw new InvalidOperationException("Invalid " + GfpganPrecisionEnvVar + ": " + configuredPrecision);
            runner.precisionMode = precisionMode;
            ApplyGfpganPack4GuardFromEnv(runner);
            runner.ProgressChanged += (value, message) =>
                Debug.Log("[GFPGAN Progress] " + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            if (!string.IsNullOrWhiteSpace(result.error))
                throw new InvalidOperationException("GFPGAN failed: " + result.error);
            if (result.texture == null)
                throw new InvalidOperationException("GFPGAN returned no output texture.");
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

    public static void RunSdInpaintingSampleMaskBatch() => RunBatchBlocking(nameof(RunSdInpaintingSampleMaskBatch), RunSdInpaintingSampleMaskInternal, TimeSpan.FromHours(4));

    public static void RunYoloAndDeepFillV2DebugBatch() => RunBatchBlocking(nameof(RunYoloAndDeepFillV2DebugBatch), RunYoloAndDeepFillV2DebugInternal, TimeSpan.FromMinutes(45));

    public static void RunDeepFillV2Case1DebugBatch() => RunBatchBlocking(nameof(RunDeepFillV2Case1DebugBatch), RunDeepFillV2Case1DebugInternal, TimeSpan.FromMinutes(45));

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

    private static async UniTask RunSdInpaintingSampleMaskInternal()
    {
        var inputPath = ResolveInputPath(DefaultReproStressImagePath);
        var source = LoadTexture(inputPath);
        if (source == null)
            throw new InvalidOperationException("Failed to load SD Inpainting sample input: " + inputPath);

        var mask = BuildSampleDeepFillMask(source);
        var go = new GameObject("SDInpaintingSampleMaskRunner");
        try
        {
            var runner = go.AddComponent<SDInpaintingNcnnReproRunner>();
            runner.enableDebugDump = ResolveBoolEnv(SdEnableDumpEnvVar, false);
            var result = await runner.ProcessAsync(source, mask, CancellationToken.None);
            if (!string.IsNullOrWhiteSpace(result.error))
                throw new InvalidOperationException("SD Inpainting failed: " + result.error);
            if (result.texture == null)
                throw new InvalidOperationException("SD Inpainting returned no output texture.");
            Debug.Log("SD Inpainting sample-mask result | elapsedMs=" + result.elapsedMs + " | dump=" + (result.dumpDir ?? string.Empty));
            UnityEngine.Object.DestroyImmediate(result.texture);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(mask);
            UnityEngine.Object.DestroyImmediate(source);
        }
    }

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
            Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
            Aexis.Execution.AexisGpuResourceTracker.Reset("D1.Matting");
            var runner = go.AddComponent<MatterNcnnReproRunner>();
            runner.enableDebugDump = true;
            var configuredPrecision = ResolveStringEnv(MattingPrecisionEnvVar, "Auto");
            if (!Enum.TryParse(configuredPrecision, true, out Aexis.Execution.AexisPrecisionMode precisionMode))
                throw new InvalidOperationException("Invalid " + MattingPrecisionEnvVar + ": " + configuredPrecision);
            runner.precisionMode = precisionMode;
            runner.forceBufferConvolution = false;
            // The released runner records the production Pack4 plan through a
            // CommandBuffer. The immediate path remains an explicit diagnostic
            // choice, never the sample's accidental fallback.
            runner.useCommandBuffer = ResolveBoolEnv(MattingUseCommandBufferEnvVar, true);
            runner.useAsyncComputeCommandBuffer = runner.useCommandBuffer
                && ResolveBoolEnv(MattingUseAsyncComputeEnvVar, false);
            var mattingPack4OnlyGuard = ResolveBoolEnv(MattingPack4OnlyGuardEnvVar, true);
            runner.disallowBufferAccess = mattingPack4OnlyGuard;
            runner.disallowBufferOutputs = mattingPack4OnlyGuard;
            runner.disallowBufferToTextureMaterialization = mattingPack4OnlyGuard;
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            var matteMetricDetail = result.matte != null
                ? result.matte.width.ToString(CultureInfo.InvariantCulture) + "x" + result.matte.height.ToString(CultureInfo.InvariantCulture)
                : string.Empty;
            matteMetricDetail += " | path=" + (runner.useCommandBuffer
                ? (runner.useAsyncComputeCommandBuffer ? "command_buffer_async" : "command_buffer")
                : "pack4_rt");
            var fp32MatteReference = ResolveStringEnv(MattingFp32ReferenceEnvVar, null);
            var matteReferenceLabel = ResolveStringEnv(MattingReferenceLabelEnvVar, "fp32");
            if (result.matte != null && !string.IsNullOrWhiteSpace(fp32MatteReference) && File.Exists(fp32MatteReference))
            {
                var referenceMatte = LoadTexture(fp32MatteReference);
                if (referenceMatte != null)
                {
                    try
                    {
                        ComputeTextureDiff(result.matte, referenceMatte, out var meanAbsU8, out var maxAbsU8);
                        var foregroundIou = ComputeMatteForegroundIou(result.matte, referenceMatte);
                        matteMetricDetail += " | " + matteReferenceLabel + "_mean_abs_u8=" + meanAbsU8.ToString("0.000000", CultureInfo.InvariantCulture)
                            + " | " + matteReferenceLabel + "_max_abs_u8=" + maxAbsU8.ToString(CultureInfo.InvariantCulture)
                            + " | " + matteReferenceLabel + "_foreground_iou_128=" + foregroundIou.ToString("0.000000", CultureInfo.InvariantCulture);
                    }
                    finally
                    {
                        UnityEngine.Object.DestroyImmediate(referenceMatte);
                    }
                }
            }
            WriteD1RuntimeBenchmark(
                "matting",
                result.elapsedMs,
                result.error,
                runner.LastDumpDir,
                "matte_mean_alpha",
                ComputeMatteMeanAlpha(result.matte).ToString("0.000000", CultureInfo.InvariantCulture),
                matteMetricDetail);
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
            var configuredPrecision = ResolveStringEnv(YoloPrecisionEnvVar, "Auto");
            if (!Enum.TryParse(configuredPrecision, true, out Aexis.Execution.AexisPrecisionMode precisionMode))
                throw new InvalidOperationException("Invalid " + YoloPrecisionEnvVar + ": " + configuredPrecision);
            runner.precisionMode = precisionMode;
            runner.forceBufferConvolution = ResolveBoolEnv(YoloForceBufferConvEnvVar, runner.forceBufferConvolution);
            // Production batches must exercise the Pack4 texture path. A forced
            // buffer branch is diagnostic-only and now requires DebugOracle.
            runner.forceBufferBinaryOp = ResolveBoolEnv(YoloForceBufferBinaryEnvVar, false);
            runner.useArgbFloatTensor = ResolveBoolEnv(YoloUseArgbFloatEnvVar, true);
            runner.enableGeneralTextureConvolution = ResolveBoolEnv(YoloEnableGeneralTexConvEnvVar, runner.enableGeneralTextureConvolution);
            runner.enableDepthWiseTextureConvolution = ResolveBoolEnv(YoloEnableDepthwiseTexConvEnvVar, runner.enableDepthWiseTextureConvolution);
            runner.enableConv1x1TextureConvolution = ResolveBoolEnv(YoloEnableConv1x1TexConvEnvVar, runner.enableConv1x1TextureConvolution);
            runner.enableLayerPathDebugLog = ResolveBoolEnv(YoloEnableLayerPathLogEnvVar, false);
            runner.logAllLayerHeartbeats = ResolveBoolEnv(YoloLogAllLayerHeartbeatsEnvVar, false);
            runner.logAllLayerOutputs = ResolveBoolEnv(YoloLogAllLayerOutputsEnvVar, false);
            runner.logAllBufferMaterialize = ResolveBoolEnv(YoloLogAllBufferMaterializeEnvVar, false);
            var yoloPack4OnlyGuard = ResolveBoolEnv(YoloPack4OnlyGuardEnvVar, true);
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
        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        Aexis.Execution.AexisGpuResourceTracker.Reset("NcnnDebugRunner.MONAI");
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
            runner.releaseTransientResourcesAfterEachSlidingWindowPatch = ResolveBoolEnv(MonaiClearTempPoolEachPatchEnvVar, true);
            runner.slidingWindowTransientReleaseInterval = ResolvePositiveIntEnvAllowZero(MonaiTempPoolClearIntervalEnvVar, 1);
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
            WriteD1RuntimeBenchmark(
                "monai-probe",
                result.elapsedMs,
                result.error,
                result.outputDir,
                "probe_completed",
                string.IsNullOrWhiteSpace(result.error) ? "1" : "0",
                result.caseName ?? string.Empty);
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
                    Aexis.Execution.AexisGpuResourceTracker.WriteReport(monaiOutputDir, "gpu_resource_stats.txt");
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
                    Aexis.Execution.AexisGpuResourceTracker.WriteReport(monaiOutputDir, "gpu_resource_stats_after_release.txt");
            }
            catch
            {
            }
            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
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
            yolo.forceBufferBinaryOp = ResolveBoolEnv(YoloForceBufferBinaryEnvVar, false);
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
        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        Aexis.Execution.AexisGpuResourceTracker.Reset("NcnnDebugRunner.MONAI.Pack4SelfTest");
        try
        {
            var runner = go.AddComponent<MONAINcnnReproRunner>();
            runner.enableDebugDump = true;
            runner.releaseTransientResourcesAfterEachSlidingWindowPatch = true;
            runner.slidingWindowTransientReleaseInterval = 1;
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
                + " | gpu=" + Aexis.Execution.AexisGpuResourceTracker.BuildSummary()
                + " | summary=" + summary.Replace(Environment.NewLine, " | "));
        }
        finally
        {
            try { LogResourceSnapshot("monai_pack4_selftest_before_destroy"); } catch { }
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats.txt"); } catch { }
            UnityEngine.Object.DestroyImmediate(go);
            await ReleaseGpuPressureAsync();
            try { LogResourceSnapshot("monai_pack4_selftest_after_release"); } catch { }
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats_after_release.txt"); } catch { }
            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
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
            var pack4OnlyGuard = ResolveBoolEnv(SdPack4OnlyGuardEnvVar, false);
            runner.enableDebugDump = ResolveBoolEnv(SdEnableDumpEnvVar, true);
            runner.tensorTextureFormat = ResolveRenderTextureFormatEnv(SdTensorFormatEnvVar, runner.tensorTextureFormat);
            runner.decoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdDecoderTensorFormatEnvVar, runner.decoderTensorTextureFormat);
            runner.encoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdEncoderTensorFormatEnvVar, runner.encoderTensorTextureFormat);
            runner.syncStageTimings = ResolveBoolEnv(SdSyncStageTimingsEnvVar, runner.syncStageTimings);
            runner.keepRawConvWeightsForTexturePath = ResolveBoolEnv(SdKeepRawConvWeightsEnvVar, runner.keepRawConvWeightsForTexturePath);
            runner.disallowBufferAccess = pack4OnlyGuard;
            runner.disallowBufferOutputs = pack4OnlyGuard;
            runner.disallowBufferToTextureMaterialization = pack4OnlyGuard;
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
                        : CreateGenericDumpDir("AIImage_SD_AexisRepro");
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
            var pack4OnlyGuard = ResolveBoolEnv(SdPack4OnlyGuardEnvVar, false);
            runner.enableDebugDump = enableDump;
            runner.useOfficialUnetCache = false;
            runner.tensorTextureFormat = tensorFormat;
            runner.decoderTensorTextureFormat = decoderTensorFormat;
            runner.encoderTensorTextureFormat = encoderTensorFormat;
            runner.keepRawConvWeightsForTexturePath = keepRawConvWeights;
            runner.enableAttentionMatMulPack4Specializations = true;
            runner.useCommandBuffer = ResolveBoolEnv(SdUseCommandBufferEnvVar, false);
            runner.useAsyncComputeCommandBuffer = ResolveBoolEnv(SdUseAsyncComputeEnvVar, runner.useAsyncComputeCommandBuffer);
            runner.disallowInferenceTempComputeBuffers = ResolveBoolEnv(SdDisallowTempComputeBuffersEnvVar, true);
            runner.disallowBufferAccess = pack4OnlyGuard;
            runner.disallowBufferOutputs = pack4OnlyGuard;
            runner.disallowBufferToTextureMaterialization = pack4OnlyGuard;
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
        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        Aexis.Execution.AexisGpuResourceTracker.Reset("NcnnDebugRunner.YoloInpaint");
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
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats.txt"); } catch { }
            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
        }
    }

    private static async UniTask RunYoloAndDeepFillV2DebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultDeepFillV2DebugImagePath);
        var outputDir = CreateGenericDumpDir("AIImage_YoloDeepFillV2");
        Directory.CreateDirectory(outputDir);
        var tex = LoadTexture(inputPath);
        if (tex == null)
            throw new InvalidOperationException("Failed to load DeepFillV2 debug input: " + inputPath);

        TryWriteTexturePng(tex, outputDir, "00_source.png");
        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        Aexis.Execution.AexisGpuResourceTracker.Reset("NcnnDebugRunner.YoloDeepFillV2");

        var go = new GameObject("YoloAndDeepFillV2DebugRunner");
        YoloSegResult yoloResult = default;
        try
        {
            var yoloRunner = go.AddComponent<YoloSegNcnnReproRunner>();
            yoloRunner.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
            yoloRunner.enableDebugDump = ResolveBoolEnv(SdEnableDumpEnvVar, false);
            yoloRunner.targetPersonOnly = true;
            yoloRunner.enableMaskClose = true;
            yoloRunner.enableMaskDilate = true;
            yoloRunner.flipYInput = ResolveBoolEnv(YoloFlipYEnvVar, yoloRunner.flipYInput);
            yoloRunner.ProgressChanged += (value, message) =>
            {
                Debug.Log("[DeepFillV2Batch][YOLO] progress=" + value.ToString("0.000", CultureInfo.InvariantCulture) + " | " + (message ?? string.Empty));
            };

            yoloResult = await yoloRunner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log(
                "[DeepFillV2Batch] yoloError=" + (yoloResult.error ?? string.Empty)
                + " | elapsedMs=" + yoloResult.elapsedMs.ToString(CultureInfo.InvariantCulture)
                + " | personCount=" + yoloResult.personCount.ToString(CultureInfo.InvariantCulture)
                + " | maskCoverage=" + yoloResult.maskCoverage01.ToString("0.000000", CultureInfo.InvariantCulture)
                + " | dump=" + (yoloRunner.LastDumpDir ?? string.Empty));

            if (!string.IsNullOrWhiteSpace(yoloResult.error))
                throw new InvalidOperationException("YOLO failed: " + yoloResult.error);
            if (yoloResult.mask == null)
                throw new InvalidOperationException("YOLO mask is null.");
            if (yoloResult.personCount <= 0)
                throw new InvalidOperationException("YOLO detected no person regions for DeepFillV2.");
            if (yoloResult.maskCoverage01 <= 0f)
                throw new InvalidOperationException("YOLO person mask coverage is zero.");

            TryWriteTexturePng(yoloResult.mask, outputDir, "01_person_mask.png");
            TryWriteTexturePng(yoloResult.overlay, outputDir, "02_overlay.png");
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

            var backends = ResolveDeepFillV2Backends();
            if (backends.Length == 0)
                throw new InvalidOperationException("No DeepFillV2 backend selected.");

            var summaryLines = new List<string>
            {
                "input=" + inputPath,
                "person_count=" + yoloResult.personCount.ToString(CultureInfo.InvariantCulture),
                "mask_coverage=" + yoloResult.maskCoverage01.ToString("0.000000", CultureInfo.InvariantCulture)
            };

            for (var i = 0; i < backends.Length; i++)
            {
                var backend = backends[i];
                var backendDir = Path.Combine(outputDir, backend.ToString());
                Directory.CreateDirectory(backendDir);
                var runner = go.AddComponent<DeepFillV2Runner>();
                DeepFillV2Result result = default;
                try
                {
                    runner.backend = backend;
                    runner.enableDebugDump = true;
                    runner.precisionMode = Aexis.Execution.AexisPrecisionMode.Auto;
                    runner.useArgbFloatTensor = ResolveBoolEnv(DeepFillV2UseArgbFloatEnvVar, runner.useArgbFloatTensor);
                    runner.flipYInput = ResolveBoolEnv(DeepFillV2FlipYInputEnvVar, runner.flipYInput);
                    runner.flipYOutput = ResolveBoolEnv(DeepFillV2FlipYOutputEnvVar, runner.flipYOutput);
                    runner.enableLayerPathDebugLog = ResolveBoolEnv(DeepFillV2EnableLayerPathLogEnvVar, runner.enableLayerPathDebugLog);
                    runner.sourceOnnxRelativePath = ResolveStringEnv(DeepFillV2OnnxPathEnvVar, runner.sourceOnnxRelativePath);
                    runner.ncnnParamRelativePath = ResolveStringEnv(DeepFillV2ParamPathEnvVar, runner.ncnnParamRelativePath);
                    runner.ncnnBinRelativePath = ResolveStringEnv(DeepFillV2BinPathEnvVar, runner.ncnnBinRelativePath);
                    runner.debugTensorBlobName = ResolveStringEnv(DeepFillV2DebugTensorBlobEnvVar, runner.debugTensorBlobName);
                    runner.enableGeneralTextureConvolution = true;
                    runner.enableDepthWiseTextureConvolution = true;
                    runner.enableConv1x1TextureConvolution = true;

                    result = await runner.ProcessAsync(tex, yoloResult.mask, CancellationToken.None);
                    Debug.Log(
                        "[DeepFillV2Batch][" + backend + "] error=" + (result.error ?? string.Empty)
                        + " | elapsedMs=" + result.elapsedMs.ToString(CultureInfo.InvariantCulture)
                        + " | loadMs=" + result.loadElapsedMs.ToString(CultureInfo.InvariantCulture)
                        + " | inferMs=" + result.inferenceElapsedMs.ToString(CultureInfo.InvariantCulture)
                        + " | dump=" + (result.dumpDir ?? string.Empty)
                        + " | model=" + (result.modelReport ?? string.Empty));

                    if (!string.IsNullOrWhiteSpace(result.error))
                        throw new InvalidOperationException("DeepFillV2 " + backend + " failed: " + result.error);
                    if (result.texture == null)
                        throw new InvalidOperationException("DeepFillV2 " + backend + " output texture is null.");

                    TryWriteTexturePng(result.texture, backendDir, "03_final_output.png");
                    var maskedDiff = ComputeMaskedMeanAbsDiff(tex, result.texture, yoloResult.mask, false, out var maskedPixels);
                    ComputeMaskedChangeByHorizontalHalf(
                        tex,
                        result.texture,
                        yoloResult.mask,
                        out var leftMaskedPixels,
                        out var rightMaskedPixels,
                        out var leftChangedPixels,
                        out var rightChangedPixels);
                    var backendSummary = string.Join(
                        Environment.NewLine,
                        "backend=" + backend,
                        "elapsed_ms=" + result.elapsedMs.ToString(CultureInfo.InvariantCulture),
                        "load_ms=" + result.loadElapsedMs.ToString(CultureInfo.InvariantCulture),
                        "inference_ms=" + result.inferenceElapsedMs.ToString(CultureInfo.InvariantCulture),
                        "masked_pixels=" + maskedPixels.ToString(CultureInfo.InvariantCulture),
                        "masked_mean_abs_diff_rgb=" + maskedDiff.ToString("0.0000", CultureInfo.InvariantCulture),
                        "left_masked_pixels=" + leftMaskedPixels.ToString(CultureInfo.InvariantCulture),
                        "right_masked_pixels=" + rightMaskedPixels.ToString(CultureInfo.InvariantCulture),
                        "left_changed_pixels=" + leftChangedPixels.ToString(CultureInfo.InvariantCulture),
                        "right_changed_pixels=" + rightChangedPixels.ToString(CultureInfo.InvariantCulture),
                        "deepfill_dump=" + (result.dumpDir ?? string.Empty),
                        "model_report=" + (result.modelReport ?? string.Empty));
                    File.WriteAllText(Path.Combine(backendDir, "summary.txt"), backendSummary);
                    summaryLines.Add(backend + "_elapsed_ms=" + result.elapsedMs.ToString(CultureInfo.InvariantCulture));
                    summaryLines.Add(backend + "_masked_pixels=" + maskedPixels.ToString(CultureInfo.InvariantCulture));
                    summaryLines.Add(backend + "_masked_mean_abs_diff_rgb=" + maskedDiff.ToString("0.0000", CultureInfo.InvariantCulture));
                    summaryLines.Add(backend + "_left_masked_pixels=" + leftMaskedPixels.ToString(CultureInfo.InvariantCulture));
                    summaryLines.Add(backend + "_right_masked_pixels=" + rightMaskedPixels.ToString(CultureInfo.InvariantCulture));
                    summaryLines.Add(backend + "_left_changed_pixels=" + leftChangedPixels.ToString(CultureInfo.InvariantCulture));
                    summaryLines.Add(backend + "_right_changed_pixels=" + rightChangedPixels.ToString(CultureInfo.InvariantCulture));
                    summaryLines.Add(backend + "_dump=" + (result.dumpDir ?? string.Empty));

                    if (maskedPixels <= 0)
                        throw new InvalidOperationException("DeepFillV2 " + backend + " mask has zero pixels.");
                    if (maskedDiff <= 1f)
                        throw new InvalidOperationException("DeepFillV2 " + backend + " masked RGB diff is too small: " + maskedDiff.ToString("0.0000", CultureInfo.InvariantCulture));
                    if (leftMaskedPixels > 0 && leftChangedPixels == 0)
                        throw new InvalidOperationException("DeepFillV2 " + backend + " did not modify any masked pixels in the left half.");
                    if (rightMaskedPixels > 0 && rightChangedPixels == 0)
                        throw new InvalidOperationException("DeepFillV2 " + backend + " did not modify any masked pixels in the right half.");
                }
                finally
                {
                    if (result.texture != null)
                        UnityEngine.Object.DestroyImmediate(result.texture);
                    TryInvokeReleaseMethod(runner);
                    UnityEngine.Object.DestroyImmediate(runner);
                    await ReleaseGpuPressureAsync();
                }
            }

            File.WriteAllText(Path.Combine(outputDir, "summary.txt"), string.Join(Environment.NewLine, summaryLines));
            Debug.Log("[DeepFillV2Batch] summary\n" + string.Join(Environment.NewLine, summaryLines));
        }
        finally
        {
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats.txt"); } catch { }
            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
            if (yoloResult.texture != null) UnityEngine.Object.DestroyImmediate(yoloResult.texture);
            if (yoloResult.mask != null) UnityEngine.Object.DestroyImmediate(yoloResult.mask);
            if (yoloResult.overlay != null) UnityEngine.Object.DestroyImmediate(yoloResult.overlay);
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
            await ReleaseGpuPressureAsync();
        }
    }

    private static async UniTask RunDeepFillV2Case1DebugInternal()
    {
        var outputDir = CreateGenericDumpDir("AIImage_DeepFillV2_Case1");
        Directory.CreateDirectory(outputDir);
        var source = LoadTexture(DefaultDeepFillV2Case1ImagePath);
        var maskedExample = LoadTexture(DefaultDeepFillV2Case1MaskedPath);
        var reference = LoadTexture(DefaultDeepFillV2Case1OutputPath);
        var hasGoldenCase = source != null && maskedExample != null && reference != null;
        Texture2D mask = null;
        if (hasGoldenCase)
        {
            if (source.width != maskedExample.width || source.height != maskedExample.height
                || source.width != reference.width || source.height != reference.height)
                throw new InvalidOperationException("DeepFillV2 case1 source/masked/reference dimensions do not match.");
            mask = BuildPureWhiteMask(maskedExample);
        }
        else
        {
            UnityEngine.Object.DestroyImmediate(source);
            UnityEngine.Object.DestroyImmediate(maskedExample);
            UnityEngine.Object.DestroyImmediate(reference);
            source = LoadTexture(ResolveInputPath(DefaultDeepFillV2DebugImagePath));
            maskedExample = null;
            reference = null;
            if (source == null)
                throw new InvalidOperationException("Failed to load the packaged DeepFillV2 sample input.");
            mask = BuildSampleDeepFillMask(source);
        }

        TryWriteTexturePng(source, outputDir, "00_source.png");
        TryWriteTexturePng(mask, outputDir, "01_mask_from_masked_example.png");
        if (reference != null)
            TryWriteTexturePng(reference, outputDir, "02_reference_case1_out.png");
        var go = new GameObject("DeepFillV2Case1DebugRunner");
        var summary = new List<string>
        {
            "source=" + (hasGoldenCase ? DefaultDeepFillV2Case1ImagePath : DefaultDeepFillV2DebugImagePath),
            "validation_mode=" + (hasGoldenCase ? "golden_case1" : "packaged_smoke"),
            "preprocess=resize_full_image_to_400x512"
        };
        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        Aexis.Execution.AexisGpuResourceTracker.Reset("NcnnDebugRunner.DeepFillV2Case1");
        try
        {
            var backends = ResolveDeepFillV2Backends();
            for (var i = 0; i < backends.Length; i++)
            {
                var backend = backends[i];
                var backendDir = Path.Combine(outputDir, backend.ToString());
                Directory.CreateDirectory(backendDir);
                var runner = go.AddComponent<DeepFillV2Runner>();
                DeepFillV2Result result = default;
                try
                {
                    runner.backend = backend;
                    runner.enableDebugDump = true;
                    runner.preserveUnmaskedPixels = true;
                    runner.useArgbFloatTensor = ResolveBoolEnv(DeepFillV2UseArgbFloatEnvVar, true);
                    runner.flipYInput = ResolveBoolEnv(DeepFillV2FlipYInputEnvVar, runner.flipYInput);
                    runner.flipYOutput = ResolveBoolEnv(DeepFillV2FlipYOutputEnvVar, runner.flipYOutput);
                    runner.enableLayerPathDebugLog = ResolveBoolEnv(DeepFillV2EnableLayerPathLogEnvVar, runner.enableLayerPathDebugLog);
                    runner.sourceOnnxRelativePath = ResolveStringEnv(DeepFillV2OnnxPathEnvVar, runner.sourceOnnxRelativePath);
                    runner.ncnnParamRelativePath = ResolveStringEnv(DeepFillV2ParamPathEnvVar, runner.ncnnParamRelativePath);
                    runner.ncnnBinRelativePath = ResolveStringEnv(DeepFillV2BinPathEnvVar, runner.ncnnBinRelativePath);
                    runner.debugTensorBlobName = ResolveStringEnv(DeepFillV2DebugTensorBlobEnvVar, runner.debugTensorBlobName);
                    result = await runner.ProcessAsync(source, mask, CancellationToken.None);
                    if (!string.IsNullOrWhiteSpace(result.error))
                        throw new InvalidOperationException("DeepFillV2 case1 " + backend + " failed: " + result.error);
                    if (result.texture == null)
                        throw new InvalidOperationException("DeepFillV2 case1 " + backend + " returned no output texture.");

                    TryWriteTexturePng(result.texture, backendDir, "case1_unity.png");
                    var maskedMae = 0d;
                    var fullMae = 0d;
                    var maxAbs = 0;
                    var maskedPixels = 0;
                    if (reference != null)
                    {
                        maskedMae = ComputeMaskedMeanAbsDiff(reference, result.texture, mask, false, out maskedPixels);
                        fullMae = ComputeFullRgbMeanAbsDiff(reference, result.texture, out maxAbs);
                    }
                    var backendSummary = string.Join(
                        Environment.NewLine,
                        "backend=" + backend,
                        "elapsed_ms=" + result.elapsedMs.ToString(CultureInfo.InvariantCulture),
                        "load_ms=" + result.loadElapsedMs.ToString(CultureInfo.InvariantCulture),
                        "inference_ms=" + result.inferenceElapsedMs.ToString(CultureInfo.InvariantCulture),
                        "masked_pixels=" + (reference != null ? maskedPixels.ToString(CultureInfo.InvariantCulture) : "not_available"),
                        "full_mae_rgb=" + (reference != null ? fullMae.ToString("0.000000", CultureInfo.InvariantCulture) : "not_available"),
                        "masked_mae_rgb=" + (reference != null ? maskedMae.ToString("0.000000", CultureInfo.InvariantCulture) : "not_available"),
                        "max_abs_rgb=" + (reference != null ? maxAbs.ToString(CultureInfo.InvariantCulture) : "not_available"),
                        "model_report=" + (result.modelReport ?? string.Empty),
                        "deepfill_dump=" + (result.dumpDir ?? string.Empty));
                    File.WriteAllText(Path.Combine(backendDir, "summary.txt"), backendSummary);
                    if (reference != null)
                    {
                        summary.Add(backend + "_full_mae_rgb=" + fullMae.ToString("0.000000", CultureInfo.InvariantCulture));
                        summary.Add(backend + "_masked_mae_rgb=" + maskedMae.ToString("0.000000", CultureInfo.InvariantCulture));
                        summary.Add(backend + "_max_abs_rgb=" + maxAbs.ToString(CultureInfo.InvariantCulture));
                    }
                    summary.Add(backend + "_elapsed_ms=" + result.elapsedMs.ToString(CultureInfo.InvariantCulture));
                    Debug.Log("[DeepFillV2Case1][" + backend + "]\n" + backendSummary);

                    if (hasGoldenCase && maskedPixels != 14229)
                        throw new InvalidOperationException("DeepFillV2 case1 effective mask pixel count mismatch: " + maskedPixels + ".");
                    if (hasGoldenCase && (fullMae > DeepFillV2Case1MaxFullMae
                        || maskedMae > DeepFillV2Case1MaxMaskedMae
                        || maxAbs > DeepFillV2Case1MaxAbs))
                    {
                        throw new InvalidOperationException(
                            "DeepFillV2 case1 alignment failed: fullMae=" + fullMae.ToString("0.000000", CultureInfo.InvariantCulture)
                            + " maskedMae=" + maskedMae.ToString("0.000000", CultureInfo.InvariantCulture)
                            + " maxAbs=" + maxAbs.ToString(CultureInfo.InvariantCulture));
                    }
                }
                finally
                {
                    if (result.texture != null) UnityEngine.Object.DestroyImmediate(result.texture);
                    TryInvokeReleaseMethod(runner);
                    UnityEngine.Object.DestroyImmediate(runner);
                    await ReleaseGpuPressureAsync();
                }
            }
            File.WriteAllText(Path.Combine(outputDir, "summary.txt"), string.Join(Environment.NewLine, summary));
            Debug.Log("[DeepFillV2Case1] summary\n" + string.Join(Environment.NewLine, summary));
        }
        finally
        {
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats.txt"); } catch { }
            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(source);
            UnityEngine.Object.DestroyImmediate(maskedExample);
            UnityEngine.Object.DestroyImmediate(reference);
            UnityEngine.Object.DestroyImmediate(mask);
            await ReleaseGpuPressureAsync();
        }
    }

    private static Texture2D BuildSampleDeepFillMask(Texture2D source)
    {
        var mask = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, true)
        {
            name = "DeepFillV2SampleMask",
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear
        };
        var pixels = new Color32[source.width * source.height];
        for (var i = 0; i < pixels.Length; i++)
            pixels[i] = new Color32(0, 0, 0, 255);
        var minX = source.width / 4;
        var maxX = source.width - minX;
        var minY = source.height / 4;
        var maxY = source.height - minY;
        for (var y = minY; y < maxY; y++)
        {
            for (var x = minX; x < maxX; x++)
                pixels[y * source.width + x] = new Color32(255, 255, 255, 255);
        }
        mask.SetPixels32(pixels);
        mask.Apply(false, false);
        return mask;
    }

    private static DeepFillV2Backend[] ResolveDeepFillV2Backends()
    {
        var configured = ResolveStringEnv(DeepFillV2BackendEnvVar, null);
        if (string.IsNullOrWhiteSpace(configured))
            configured = ResolveStringEnv(InpaintBackendEnvVar, "ncnn");
        configured = (configured ?? "ncnn").Trim();
        if (string.Equals(configured, "both", StringComparison.OrdinalIgnoreCase)
            || string.Equals(configured, "all", StringComparison.OrdinalIgnoreCase)
            || string.Equals(configured, "deepfillv2", StringComparison.OrdinalIgnoreCase))
            return new[] { DeepFillV2Backend.OnnxDirect, DeepFillV2Backend.NcnnBin };
        if (string.Equals(configured, "onnx", StringComparison.OrdinalIgnoreCase)
            || string.Equals(configured, "onnxdirect", StringComparison.OrdinalIgnoreCase)
            || string.Equals(configured, "deepfillv2_onnx", StringComparison.OrdinalIgnoreCase))
            return new[] { DeepFillV2Backend.OnnxDirect };
        if (string.Equals(configured, "bin", StringComparison.OrdinalIgnoreCase)
            || string.Equals(configured, "ncnn", StringComparison.OrdinalIgnoreCase)
            || string.Equals(configured, "ncnnbin", StringComparison.OrdinalIgnoreCase)
            || string.Equals(configured, "deepfillv2_ncnn", StringComparison.OrdinalIgnoreCase))
            return new[] { DeepFillV2Backend.NcnnBin };
        if (string.Equals(configured, "sd", StringComparison.OrdinalIgnoreCase)
            || string.Equals(configured, "sd15", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("RunYoloAndDeepFillV2DebugBatch cannot run SD backend; use RunYoloAndInpaintingDebugBatch for SD1.5.");
        throw new InvalidOperationException("Unsupported DeepFillV2 backend: " + configured);
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
            var pack4OnlyGuard = ResolveBoolEnv(SdPack4OnlyGuardEnvVar, false);
            runner.enableDebugDump = ResolveBoolEnv(SdEnableDumpEnvVar, true);
            runner.tensorTextureFormat = ResolveRenderTextureFormatEnv(SdTensorFormatEnvVar, runner.tensorTextureFormat);
            runner.decoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdDecoderTensorFormatEnvVar, runner.decoderTensorTextureFormat);
            runner.encoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdEncoderTensorFormatEnvVar, runner.encoderTensorTextureFormat);
            runner.keepRawConvWeightsForTexturePath = ResolveBoolEnv(SdKeepRawConvWeightsEnvVar, runner.keepRawConvWeightsForTexturePath);
            runner.enableAttentionMatMulPack4Specializations = true;
            runner.useCommandBuffer = ResolveBoolEnv(SdUseCommandBufferEnvVar, false);
            runner.disallowInferenceTempComputeBuffers = ResolveBoolEnv(SdDisallowTempComputeBuffersEnvVar, true);
            runner.disallowBufferAccess = pack4OnlyGuard;
            runner.disallowBufferOutputs = pack4OnlyGuard;
            runner.disallowBufferToTextureMaterialization = pack4OnlyGuard;

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
            Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
            Aexis.Execution.AexisGpuResourceTracker.Reset("NcnnDebugRunner.SDUnetReplay");
            var runner = go.AddComponent<SDInpaintingNcnnReproRunner>();
            var pack4OnlyGuard = ResolveBoolEnv(SdPack4OnlyGuardEnvVar, false);
            runner.enableDebugDump = true;
            runner.tensorTextureFormat = ResolveRenderTextureFormatEnv(SdTensorFormatEnvVar, runner.tensorTextureFormat);
            runner.keepRawConvWeightsForTexturePath = ResolveBoolEnv(SdKeepRawConvWeightsEnvVar, runner.keepRawConvWeightsForTexturePath);
            runner.enableAttentionMatMulPack4Specializations = true;
            runner.disallowInferenceTempComputeBuffers = ResolveBoolEnv(SdDisallowTempComputeBuffersEnvVar, true);
            runner.disallowBufferAccess = pack4OnlyGuard;
            runner.disallowBufferOutputs = pack4OnlyGuard;
            runner.disallowBufferToTextureMaterialization = pack4OnlyGuard;

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
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(outputDir, "gpu_resource_stats.txt"); } catch { }
            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
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

    [MenuItem("Aexis/Examples/Debug/Run VISTA3D Debug")]
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

        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        try
        {
            for (var i = 0; i < iterations; i++)
            {
                var iteration = i + 1;
                var iterationDir = Path.Combine(outputDir, "iter_" + iteration.ToString("00", CultureInfo.InvariantCulture));
                Directory.CreateDirectory(iterationDir);

                Aexis.Execution.AexisGpuResourceTracker.Reset("NcnnDebugRunner.YoloInpaint.Probe." + iteration.ToString(CultureInfo.InvariantCulture));
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
                try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(iterationDir, "gpu_resource_stats.txt"); } catch { }
                sw.Flush();
            }
        }
        finally
        {
            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
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
            var pack4OnlyGuard = ResolveBoolEnv(SdPack4OnlyGuardEnvVar, false);
            inpaintRunner.enableDebugDump = enableDump;
            inpaintRunner.ApplyPeopleRemovalPreset();
            var configuredInpaintPrecision = ResolveStringEnv(SdInpaintingPrecisionEnvVar, "Auto");
            if (!Enum.TryParse(configuredInpaintPrecision, true, out Aexis.Execution.AexisPrecisionMode inpaintPrecisionMode))
                throw new InvalidOperationException("Invalid " + SdInpaintingPrecisionEnvVar + ": " + configuredInpaintPrecision);
            inpaintRunner.precisionMode = inpaintPrecisionMode;
            inpaintRunner.useOfficialUnetCache = ResolveBoolEnv("AIIMAGE_SD_USE_OFFICIAL_UNET_CACHE", inpaintRunner.useOfficialUnetCache);
            inpaintRunner.tensorTextureFormat = ResolveRenderTextureFormatEnv(SdTensorFormatEnvVar, inpaintRunner.tensorTextureFormat);
            inpaintRunner.decoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdDecoderTensorFormatEnvVar, inpaintRunner.decoderTensorTextureFormat);
            inpaintRunner.encoderTensorTextureFormat = ResolveRenderTextureFormatEnv(SdEncoderTensorFormatEnvVar, inpaintRunner.encoderTensorTextureFormat);
            inpaintRunner.keepRawConvWeightsForTexturePath = ResolveBoolEnv(SdKeepRawConvWeightsEnvVar, inpaintRunner.keepRawConvWeightsForTexturePath);
            inpaintRunner.enableAttentionMatMulPack4Specializations = true;
            inpaintRunner.useCommandBuffer = useCommandBuffer;
            inpaintRunner.useAsyncComputeCommandBuffer = ResolveBoolEnv(SdUseAsyncComputeEnvVar, inpaintRunner.useAsyncComputeCommandBuffer);
            inpaintRunner.disallowInferenceTempComputeBuffers = ResolveBoolEnv(SdDisallowTempComputeBuffersEnvVar, true);
            inpaintRunner.disallowBufferAccess = pack4OnlyGuard;
            inpaintRunner.disallowBufferOutputs = pack4OnlyGuard;
            inpaintRunner.disallowBufferToTextureMaterialization = pack4OnlyGuard;
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
                + " | " + Aexis.Execution.AexisGpuResourceTracker.BuildSummary());
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
            + EscapeTsv(Aexis.Execution.AexisGpuResourceTracker.BuildSummary()) + "\t"
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

    // Runs the real CodeFormer encoder without depending on the face detector.
    // This is deliberately a separate, model-backed regression because the
    // packaged generic debug texture is not a face and therefore stops before
    // the encoder in RunCodeFormerDebugBatch.
    public static void RunCodeFormerEncoderPack4RegressionBatch() => RunBatchBlocking(
        nameof(RunCodeFormerEncoderPack4RegressionBatch),
        RunCodeFormerEncoderPack4RegressionInternal,
        TimeSpan.FromMinutes(20));

    public static void RunClipDebugBatch() => RunBatchBlocking(nameof(RunClipDebugBatch), RunClipDebugInternal);

    public static void RunClipDirectoryDebugBatch() => RunBatchBlocking(nameof(RunClipDirectoryDebugBatch), RunClipDirectoryDebugInternal);

    public static void RunGfpganDebugBatch() => RunBatchBlocking(nameof(RunGfpganDebugBatch), RunGfpganDebugInternal);

    public static void RunCodeFormerStressBatch() => RunBatchBlocking(nameof(RunCodeFormerStressBatch), RunCodeFormerStressInternal, TimeSpan.FromHours(2));

    public static void RunReproSuiteStressBatch() => RunBatchBlocking(nameof(RunReproSuiteStressBatch), RunReproSuiteStressInternal, TimeSpan.FromHours(2));

    // Keeps the pack4 production admission, GPU kernels, and CommandBuffer
    // lifetimes runnable from a graphics-enabled Unity batch session. It is
    // intentionally independent of model assets, so it is a fast gate before
    // the model runner regression suite.
    public static void RunAexisStrictTextureValidationBatch()
    {
        try
        {
            Aexis.Editor.AexisPackageValidation.RunBatchSmoke();
            Aexis.Editor.AexisP1PackageValidation.RunBatchSmoke();
            NcnnProductionPathAuditTests.RunBatchValidation();
            NcnnStrictTextureExecutionPlanTests.RunBatchValidation();
            NcnnTemporaryRtLifecycleTests.RunBatchValidation();
            NcnnC4Pack4LayoutTests.RunBatchValidation();
            NcnnC2CdhwCmdPack4Tests.RunBatchValidation();
            NcnnConvCmdPack4GoldenTests.RunBatchValidation();
            NcnnLinearCmdPack4GoldenTests.RunBatchValidation();
            Debug.Log("[NcnnDebugRunner] strict Pack4/CommandBuffer validation passed");
            EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            EditorApplication.Exit(1);
        }
    }

    public static void RunMonaiDebugBatch() => RunBatchBlocking(nameof(RunMonaiDebugBatch), () => RunMonaiDebugInternal(), TimeSpan.FromMinutes(10));

    public static void RunQwen35TokenizerContractBatch()
    {
        var start = Stopwatch.StartNew();
        var projectRoot = Directory.GetParent(Application.dataPath).FullName;
        var modelDir = ResolveQwen35ModelDirectory(projectRoot, "qwen3.5_0.8b_mobile_q4");
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_TOKENIZER_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_tokenizer_contract.json");

        string error = null;
        var valid = false;
        var ids = new JArray();
        var decoded = string.Empty;
        var vocabularySize = 0;
        var specialIds = new JObject();
        const string text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>";
        try
        {
            using (var runner = new Qwen35Runner(modelDir, 1))
            {
                vocabularySize = runner.Tokenizer.VocabularySize;
                var encoded = runner.Tokenizer.Encode(text);
                for (var i = 0; i < encoded.Count; i++) ids.Add(encoded[i]);
                decoded = runner.Tokenizer.Decode(encoded, false);
                specialIds["im_start"] = runner.Tokenizer.IdOf("<|im_start|>");
                specialIds["im_end"] = runner.Tokenizer.IdOf("<|im_end|>");
                specialIds["vision_start"] = runner.Tokenizer.IdOf("<|vision_start|>");
                specialIds["image_pad"] = runner.Tokenizer.IdOf("<|image_pad|>");
                specialIds["vision_end"] = runner.Tokenizer.IdOf("<|vision_end|>");
                valid = vocabularySize == 248077
                    && decoded == text
                    && (int)specialIds["im_start"] == 248045
                    && (int)specialIds["im_end"] == 248046
                    && (int)specialIds["vision_start"] == 248053
                    && (int)specialIds["image_pad"] == 248056
                    && (int)specialIds["vision_end"] == 248054;
            }
        }
        catch (Exception exception)
        {
            error = exception.ToString();
        }

        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.tokenizer-contract/v1",
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["vocabulary_size"] = vocabularySize,
            ["text"] = text,
            ["token_ids"] = ids,
            ["decoded"] = decoded,
            ["special_ids"] = specialIds,
            ["error"] = error ?? string.Empty,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -quit -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35TokenizerContractBatch"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] tokenizer contract report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen3.5 tokenizer contract failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35TextGenerationBatch()
    {
        var start = Stopwatch.StartNew();
        var projectRoot = Directory.GetParent(Application.dataPath).FullName;
        var modelDir = ResolveQwen35ModelDirectory(projectRoot, "qwen3.5_0.8b_mobile_q4");
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_TEXT_GENERATION_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_text_generation.json");
        var prompt = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_TEXT_PROMPT");
        if (string.IsNullOrEmpty(prompt))
            prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
        var maxNewTokens = 1;
        if (int.TryParse(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_MAX_NEW_TOKENS"), out var configuredMax))
            maxNewTokens = Mathf.Clamp(configuredMax, 1, 256);

        string error = null;
        var valid = false;
        var entryPoint = string.Empty;
        var promptIds = new JArray();
        var generatedIds = new JArray();
        var generatedText = string.Empty;
        var finalPosition = 0;
        var cacheTextures = 0;
        long decoderRuns = 0;
        long sharedWeightBytes = 0;
        long peakWorkingSetBytes = 0;
        var strictPlanNodes = new JArray();
        var dumpPrefix = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_TEXT_DUMP_PREFIX");
        var dumpFiles = new JObject();
        try
        {
            using (var runner = new Qwen35Runner(modelDir, maxNewTokens))
            {
                entryPoint = runner.RequireInferenceEntryPoint();
                var encoded = runner.Tokenizer.Encode(prompt);
                for (var i = 0; i < encoded.Count; i++) promptIds.Add(encoded[i]);
                using (var session = runner.CreateDecoderSession())
                {
                    sharedWeightBytes = session.SharedWeightBytes;
                    if (string.Equals(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_DECODER_LAYER_LOG"), "1", StringComparison.Ordinal))
                        session.DebugLog = line => Debug.Log("[Qwen35][DecoderSession] " + line);
                    if (!string.IsNullOrWhiteSpace(dumpPrefix))
                    {
                        session.DebugTextureReadback = (name, values) =>
                        {
                            var dumpPath = dumpPrefix + "." + name + ".f32";
                            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(dumpPath)));
                            using (var stream = File.Create(dumpPath))
                            using (var writer = new BinaryWriter(stream))
                                for (var valueIndex = 0; valueIndex < values.Length; valueIndex++) writer.Write(values[valueIndex]);
                            dumpFiles[name] = dumpPath;
                        };
                    }
                    var checkpointText = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_DECODER_CHECKPOINT_BLOBS");
                    if (!string.IsNullOrWhiteSpace(checkpointText) && !string.IsNullOrWhiteSpace(dumpPrefix))
                    {
                        var checkpointBlobs = new HashSet<string>(
                            checkpointText.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries).Select(value => value.Trim()),
                            StringComparer.Ordinal);
                        session.ConfigureDebugLayerReadback(checkpointBlobs, (decodeIndex, layerName, blobName, values) =>
                        {
                            var dumpPath = dumpPrefix + ".decode" + decodeIndex + ".blob_" + blobName + ".f32";
                            using (var stream = File.Create(dumpPath))
                            using (var writer = new BinaryWriter(stream))
                                for (var valueIndex = 0; valueIndex < values.Length; valueIndex++) writer.Write(values[valueIndex]);
                            dumpFiles["decode" + decodeIndex + "/" + blobName] = dumpPath;
                        });
                    }
                    var generated = session.Generate(encoded, maxNewTokens, Qwen35SamplingConfig.Greedy());
                    generatedText = generated.Text;
                    finalPosition = generated.FinalPosition;
                    cacheTextures = generated.FinalCacheTextureCount;
                    decoderRuns = generated.DecoderStepCount;
                    for (var i = 0; i < generated.TokenIds.Count; i++) generatedIds.Add(generated.TokenIds[i]);
                    var expectedFinalPosition = encoded.Count + Mathf.Max(0, generated.TokenIds.Count - 1);
                    valid = generated.TokenIds.Count > 0
                        && cacheTextures == 48
                        && decoderRuns > 0
                        && finalPosition == expectedFinalPosition;
                }
            }
            peakWorkingSetBytes = Process.GetCurrentProcess().PeakWorkingSet64;
        }
        catch (Exception exception)
        {
            error = exception.ToString();
            if (exception is StrictTextureInferencePlanException strictPlanException)
            {
                foreach (var node in strictPlanException.Plan?.nodes ?? Array.Empty<AexisTextureExecutionPlanNode>())
                {
                    if (node == null
                        || (node.accepted && (node.layerIndex < 6 || node.layerIndex > 10)))
                        continue;
                    strictPlanNodes.Add(new JObject
                    {
                        ["layer_index"] = node.layerIndex,
                        ["layer"] = node.layer ?? string.Empty,
                        ["operator"] = node.operatorName ?? string.Empty,
                        ["accepted"] = node.accepted,
                        ["execution_path"] = node.executionPath ?? string.Empty,
                        ["inputs"] = new JArray((node.inputs ?? Array.Empty<AexisTexturePlanTensorDescriptor>()).Select(input => input == null
                            ? new JObject { ["missing"] = true }
                            : new JObject
                            {
                                ["blob"] = input.blob ?? string.Empty,
                                ["dtype"] = input.dtype ?? string.Empty,
                                ["logical_dtype"] = input.logicalDtype ?? string.Empty,
                                ["layout"] = input.layout ?? string.Empty,
                                ["logical_shape"] = new JArray(input.logicalShape ?? Array.Empty<int>()),
                                ["storage_shape"] = new JArray(input.storageShape ?? Array.Empty<int>())
                            })),
                        ["outputs"] = new JArray((node.outputs ?? Array.Empty<AexisTexturePlanTensorDescriptor>()).Select(output => output == null
                            ? new JObject { ["missing"] = true }
                            : new JObject
                            {
                                ["blob"] = output.blob ?? string.Empty,
                                ["dtype"] = output.dtype ?? string.Empty,
                                ["logical_dtype"] = output.logicalDtype ?? string.Empty,
                                ["layout"] = output.layout ?? string.Empty,
                                ["logical_shape"] = new JArray(output.logicalShape ?? Array.Empty<int>()),
                                ["storage_shape"] = new JArray(output.storageShape ?? Array.Empty<int>())
                            }))
                    });
                }
            }
            try { peakWorkingSetBytes = Process.GetCurrentProcess().PeakWorkingSet64; } catch { }
        }

        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.text-generation/v1",
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["model_directory"] = modelDir,
            ["entry_point"] = entryPoint,
            ["prompt"] = prompt,
            ["prompt_token_ids"] = promptIds,
            ["generated_token_ids"] = generatedIds,
            ["generated_text"] = generatedText,
            ["final_position"] = finalPosition,
            ["decoder_runs"] = decoderRuns,
            ["cache_texture_count"] = cacheTextures,
            ["shared_weight_bytes"] = sharedWeightBytes,
            ["peak_working_set_bytes"] = peakWorkingSetBytes,
            ["strict_plan_nodes"] = strictPlanNodes,
            ["debug_dump_files"] = dumpFiles,
            ["strict_texture_execution"] = true,
            ["compute_buffer_fallback"] = false,
            ["error"] = error ?? string.Empty,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen35_text_generation.log",
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -quit -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35TextGenerationBatch"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] text generation report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen3.5 text generation failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35MultimodalGenerationBatch()
    {
        var start = Stopwatch.StartNew();
        var projectRoot = Directory.GetParent(Application.dataPath).FullName;
        var modelDir = ResolveQwen35ModelDirectory(projectRoot, "qwen3.5_0.8b_mobile_q4");
        var imagePath = ResolveQwen35ImagePath(projectRoot);
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_MULTIMODAL_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_multimodal_generation.json");
        var referencePath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "reference_cli_validation.json");
        JObject reference = null;
        string referenceError = null;
        if (File.Exists(referencePath))
        {
            try { reference = JObject.Parse(File.ReadAllText(referencePath)); }
            catch (Exception exception) { referenceError = exception.Message; }
        }
        var prompt = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_IMAGE_PROMPT");
        if (string.IsNullOrWhiteSpace(prompt))
            prompt = reference == null
                ? "Describe the image accurately."
                : (string)reference["prompt"];
        var maxNewTokens = 1;
        if (int.TryParse(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_MAX_NEW_TOKENS"), out var configuredMax))
            maxNewTokens = Mathf.Clamp(configuredMax, 1, 512);
        var requireOcrMarkers = string.Equals(
            Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_REQUIRE_OCR_MARKERS"),
            "1",
            StringComparison.Ordinal);
        var expectedTokenIds = ResolveQwen35ExpectedTokenIds(modelDir);

        var valid = false;
        string error = null;
        var generatedIds = new JArray();
        var generatedText = string.Empty;
        var markerHits = new JArray();
        var markerGroupCount = 0;
        var markerHitCount = 0;
        var expectedTokenIdsMatched = false;
        var firstTokenIsKnown = false;
        var sourceWidth = 0;
        var sourceHeight = 0;
        var targetWidth = 0;
        var targetHeight = 0;
        var gridWidth = 0;
        var gridHeight = 0;
        var visionTokenCount = 0;
        var promptTokenCount = 0;
        var expandedPromptTokenCount = 0;
        var finalPosition = 0;
        var cacheTextureCount = 0;
        var contextTextureCapacity = 0;
        long decoderRuns = 0;
        long peakWorkingSetBytes = 0;
        long peakPrivateBytes = 0;
        long runnerInitializationMs = 0;
        long visionLoadMs = 0;
        long visionEncodeMs = 0;
        long decoderLoadMs = 0;
        long generationMs = 0;
        var memorySamples = new JArray();
        var visionLoadProfiles = new JObject();
        var decoderLoadProfiles = new JObject();
        var decoderRuntimeProfiles = new JObject();
        var decoderRuntimeProfileEnabled = ResolveBoolEnv("AIIMAGE_QWEN35_DECODER_LAYER_PROFILE", false);
        var decoderRuntimeProfileSyncGpu = decoderRuntimeProfileEnabled
            && ResolveBoolEnv("AIIMAGE_QWEN35_DECODER_LAYER_PROFILE_SYNC_GPU", false);
        AexisGpuResourceTracker.Reset("Qwen35MultimodalGenerationBatch");
        Action<string> sampleMemory = stage =>
        {
            try
            {
                using (var process = Process.GetCurrentProcess())
                {
                    peakWorkingSetBytes = Math.Max(peakWorkingSetBytes, process.WorkingSet64);
                    peakPrivateBytes = Math.Max(peakPrivateBytes, process.PrivateMemorySize64);
                    var gpu = AexisGpuResourceTracker.GetStatsSnapshot();
                    memorySamples.Add(new JObject
                    {
                        ["stage"] = stage ?? string.Empty,
                        ["elapsed_ms"] = start.ElapsedMilliseconds,
                        ["working_set_bytes"] = process.WorkingSet64,
                        ["private_bytes"] = process.PrivateMemorySize64,
                        ["tracked_gpu_buffer_bytes"] = gpu.currentBufferBytes,
                        ["tracked_gpu_texture_bytes"] = gpu.currentTextureBytes,
                        ["tracked_gpu_total_bytes"] = gpu.currentBufferBytes + gpu.currentTextureBytes,
                        ["tracked_gpu_peak_bytes"] = gpu.peakTotalBytes,
                        ["tracked_gpu_peak_temporary_texture_bytes"] = gpu.peakTemporaryTextureBytes
                    });
                }
            }
            catch { }
        };
        sampleMemory("start");
        try
        {
            if (requireOcrMarkers && reference == null)
                throw new InvalidOperationException(
                    "AIIMAGE_QWEN35_REQUIRE_OCR_MARKERS=1 requires the offline reference report: " + referencePath);

            var stageTimer = Stopwatch.StartNew();
            using (var runner = new Qwen35Runner(modelDir, maxNewTokens))
            {
                runnerInitializationMs = stageTimer.ElapsedMilliseconds;
                sampleMemory("runner_ready");

                Qwen35VisionEncoding vision = null;
                stageTimer.Restart();
                using (var visionSession = runner.CreateVisionEncoderSession())
                {
                    visionLoadMs = stageTimer.ElapsedMilliseconds;
                    visionLoadProfiles["patch_embedding"] = Qwen35LoadProfileToJson(visionSession.PatchEmbeddingLoadProfile);
                    visionLoadProfiles["position_embedding"] = Qwen35LoadProfileToJson(visionSession.PositionEmbeddingLoadProfile);
                    visionLoadProfiles["encoder"] = Qwen35LoadProfileToJson(visionSession.EncoderLoadProfile);
                    sampleMemory("vision_loaded");
                    stageTimer.Restart();
                    using (var encoded = visionSession.EncodeFile(imagePath))
                        vision = encoded.CloneStandalone();
                    sourceWidth = vision.SourceWidth;
                    sourceHeight = vision.SourceHeight;
                    targetWidth = vision.TargetWidth;
                    targetHeight = vision.TargetHeight;
                    gridWidth = vision.GridWidth;
                    gridHeight = vision.GridHeight;
                    visionTokenCount = vision.EmbeddingCount;
                    visionEncodeMs = stageTimer.ElapsedMilliseconds;
                    sampleMemory("vision_encoded");
                }
                sampleMemory("vision_released");

                using (vision)
                {
                    stageTimer.Restart();
                    using (var decoder = runner.CreateDecoderSession())
                    {
                        if (decoderRuntimeProfileEnabled)
                            decoder.ConfigureDecoderRuntimeProfiling(true, decoderRuntimeProfileSyncGpu);
                        decoderLoadMs = stageTimer.ElapsedMilliseconds;
                        decoderLoadProfiles["token_embedding"] = Qwen35LoadProfileToJson(decoder.TokenEmbeddingLoadProfile);
                        decoderLoadProfiles["decoder"] = Qwen35LoadProfileToJson(decoder.DecoderLoadProfile);
                        decoderLoadProfiles["projection"] = Qwen35LoadProfileToJson(decoder.ProjectionLoadProfile);
                        sampleMemory("decoder_loaded");
                        var promptIds = runner.EncodeImagePrompt(prompt);
                        promptTokenCount = promptIds.Count;
                        stageTimer.Restart();
                        var sampledTokens = 0;
                        var generated = decoder.GenerateMultimodal(
                            promptIds,
                            vision,
                            maxNewTokens,
                            Qwen35SamplingConfig.Greedy(),
                            (tokenId, piece) =>
                            {
                                sampledTokens++;
                                if ((sampledTokens & 7) == 0)
                                    sampleMemory("token_" + sampledTokens);
                            });
                        generationMs = stageTimer.ElapsedMilliseconds;
                        sampleMemory("generation_complete");
                        generatedText = generated.Text;
                        expandedPromptTokenCount = generated.ExpandedPromptTokenCount;
                        finalPosition = generated.FinalPosition;
                        cacheTextureCount = generated.FinalCacheTextureCount;
                        contextTextureCapacity = generated.ContextTextureCapacity;
                        decoderRuns = generated.DecoderStepCount;
                        if (decoderRuntimeProfileEnabled)
                        {
                            decoderRuntimeProfiles["prefill"] = Qwen35RuntimeProfileToJson(decoder.PrefillDecoderRuntimeProfile);
                            decoderRuntimeProfiles["first_decode"] = Qwen35RuntimeProfileToJson(decoder.FirstDecodeRuntimeProfile);
                            decoderRuntimeProfiles["last_decode"] = Qwen35RuntimeProfileToJson(decoder.LastDecodeRuntimeProfile);
                        }
                        for (var i = 0; i < generated.TokenIds.Count; i++)
                            generatedIds.Add(generated.TokenIds[i]);
                    }
                }

                if (reference != null && reference["marker_group_hits"] is JArray groups)
                {
                    markerGroupCount = groups.Count;
                    for (var groupIndex = 0; groupIndex < groups.Count; groupIndex++)
                    {
                        var groupResult = new JObject { ["matched"] = false, ["variants"] = new JArray() };
                        var matched = false;
                        if (groups[groupIndex] is JArray variants)
                        {
                            for (var variantIndex = 0; variantIndex < variants.Count; variantIndex++)
                            {
                                var variant = (string)variants[variantIndex] ?? string.Empty;
                                ((JArray)groupResult["variants"]).Add(variant);
                                if (!string.IsNullOrEmpty(variant) && generatedText.Contains(variant))
                                    matched = true;
                            }
                        }
                        groupResult["matched"] = matched;
                        if (matched) markerHitCount++;
                        markerHits.Add(groupResult);
                    }
                }

                firstTokenIsKnown = generatedIds.Count > 0
                    && (int)generatedIds[0] >= 0
                    && (int)generatedIds[0] < runner.Tokenizer.VocabularySize;
                expectedTokenIdsMatched = expectedTokenIds.Count == 0 || expectedTokenIds.Count <= generatedIds.Count;
                if (expectedTokenIdsMatched)
                {
                    for (var tokenIndex = 0; tokenIndex < expectedTokenIds.Count; tokenIndex++)
                    {
                        if ((int)expectedTokenIds[tokenIndex] != (int)generatedIds[tokenIndex])
                        {
                            expectedTokenIdsMatched = false;
                            break;
                        }
                    }
                }
                valid = firstTokenIsKnown
                    && cacheTextureCount == 48
                    && decoderRuns > 0
                    && visionTokenCount == (gridWidth / 2) * (gridHeight / 2)
                    && expandedPromptTokenCount == promptTokenCount - 1 + visionTokenCount
                    && (!requireOcrMarkers || markerHitCount == markerGroupCount)
                    && (expectedTokenIds.Count == 0 || expectedTokenIdsMatched);
            }
            sampleMemory("disposed");
        }
        catch (Exception exception)
        {
            error = exception.ToString();
            sampleMemory("error");
        }

        start.Stop();
        var gpuStats = AexisGpuResourceTracker.GetStatsSnapshot();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.multimodal-generation/v1",
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["model_directory"] = modelDir,
            ["image"] = imagePath,
            ["image_sha256"] = File.Exists(imagePath) ? ComputeQwen35Sha256(imagePath) : string.Empty,
            ["prompt"] = prompt,
            ["reference_report"] = referencePath,
            ["reference_report_available"] = reference != null,
            ["reference_report_error"] = referenceError ?? string.Empty,
            ["max_new_tokens"] = maxNewTokens,
            ["require_ocr_markers"] = requireOcrMarkers,
            ["generated_token_ids"] = generatedIds,
            ["expected_token_ids"] = expectedTokenIds,
            ["expected_token_ids_matched"] = expectedTokenIds.Count == 0 || expectedTokenIdsMatched,
            ["first_token_is_known"] = firstTokenIsKnown,
            ["generated_text"] = generatedText,
            ["marker_group_count"] = markerGroupCount,
            ["marker_hit_count"] = markerHitCount,
            ["marker_hits"] = markerHits,
            ["source_width"] = sourceWidth,
            ["source_height"] = sourceHeight,
            ["target_width"] = targetWidth,
            ["target_height"] = targetHeight,
            ["grid_width"] = gridWidth,
            ["grid_height"] = gridHeight,
            ["vision_token_count"] = visionTokenCount,
            ["prompt_token_count"] = promptTokenCount,
            ["expanded_prompt_token_count"] = expandedPromptTokenCount,
            ["final_position"] = finalPosition,
            ["decoder_runs"] = decoderRuns,
            ["cache_texture_count"] = cacheTextureCount,
            ["context_texture_capacity"] = contextTextureCapacity,
            ["peak_working_set_bytes"] = peakWorkingSetBytes,
            ["peak_private_bytes"] = peakPrivateBytes,
            ["tracked_gpu_peak_buffer_bytes"] = gpuStats.peakBufferBytes,
            ["tracked_gpu_peak_texture_bytes"] = gpuStats.peakTextureBytes,
            ["tracked_gpu_peak_total_bytes"] = gpuStats.peakTotalBytes,
            ["tracked_gpu_peak_temporary_texture_bytes"] = gpuStats.peakTemporaryTextureBytes,
            ["memory_samples"] = memorySamples,
            ["stage_timings_ms"] = new JObject
            {
                ["runner_initialization"] = runnerInitializationMs,
                ["vision_load"] = visionLoadMs,
                ["vision_encode"] = visionEncodeMs,
                ["decoder_load"] = decoderLoadMs,
                ["generation"] = generationMs
            },
            ["vision_load_profiles"] = visionLoadProfiles,
            ["decoder_load_profiles"] = decoderLoadProfiles,
            ["decoder_runtime_profiles"] = decoderRuntimeProfiles,
            ["decoder_runtime_profile_enabled"] = decoderRuntimeProfileEnabled,
            ["decoder_runtime_profile_sync_gpu"] = decoderRuntimeProfileSyncGpu,
            ["managed_load_gc_interval_mb"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_LOAD_GC_INTERVAL_MB") ?? "0",
            ["strict_texture_execution"] = true,
            ["activation_storage"] = "texture-backed",
            ["compute_buffer_fallback"] = false,
            ["exit_code"] = valid && string.IsNullOrEmpty(error) ? 0 : 1,
            ["error"] = error ?? string.Empty,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -quit -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35MultimodalGenerationBatch",
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen35_multimodal_generation.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] multimodal generation report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen3.5 multimodal generation failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35AsyncMultimodalGenerationBatch()
    {
        RunBatchBlocking(
            nameof(RunQwen35AsyncMultimodalGenerationBatch),
            RunQwen35AsyncMultimodalGenerationInternal,
            TimeSpan.FromMinutes(45));
    }

    private static async UniTask RunQwen35AsyncMultimodalGenerationInternal()
    {
        var start = Stopwatch.StartNew();
        var projectRoot = Directory.GetParent(Application.dataPath).FullName;
        var modelDir = ResolveQwen35ModelDirectory(projectRoot, "qwen3.5_0.8b_mobile_q4");
        var imagePath = ResolveQwen35ImagePath(projectRoot);
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_ASYNC_MULTIMODAL_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            outputPath = Path.Combine(
                projectRoot,
                "Tools",
                "Qwen35NcnnBaseline",
                "reports",
                "unity_async_multimodal_smoke_mobile_q4.json");
        }
        var prompt = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_IMAGE_PROMPT");
        if (string.IsNullOrWhiteSpace(prompt))
        {
            prompt = "请对当前图像进行详细、客观的中文分析，涵盖主体与人物、场景环境、构图、色彩与光线、可见文字、关键细节和可能用途。"
                + "对无法确认的内容明确说明，不要编造。";
        }
        var maxNewTokens = 6;
        if (int.TryParse(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_MAX_NEW_TOKENS"), out var configuredMax))
            maxNewTokens = Mathf.Clamp(configuredMax, 1, 512);
        var expectedTokenIds = ResolveQwen35ExpectedTokenIds(modelDir);

        var stageCallbacks = new JArray();
        var initializationProgressCallbacks = new JArray();
        var pipelineProgressCallbacks = new JArray();
        var progressCallbacks = new JArray();
        var streamedTokenIds = new JArray();
        var streamedText = new System.Text.StringBuilder();
        var generatedTokenIds = new JArray();
        var generatedText = string.Empty;
        var progressMonotonic = true;
        var previousCompleted = 0;
        var progressTotalConsistent = true;
        var initializationProgressMonotonic = true;
        var pipelineProgressMonotonic = true;
        var previousInitializationProgress = -1f;
        var previousPipelineProgress = -1f;
        var pipelineStages = new HashSet<string>(StringComparer.Ordinal);
        var finalCacheTextureCount = 0;
        var decoderRuns = 0L;
        var promptTokenCount = 0;
        var visionTokenCount = 0;
        var expandedPromptTokenCount = 0;
        var stoppedOnEndOfTurn = false;
        var firstTokenIsKnown = false;
        var expectedTokenIdsMatched = false;
        var valid = false;
        string error = null;
        Texture2D image = null;

        try
        {
            image = LoadTexture(imagePath);
            if (image == null)
                throw new FileNotFoundException("Failed to load Qwen3.5 async smoke image.", imagePath);

            using (var runner = await Qwen35Runner.CreateAsync(
                modelDir,
                maxNewTokens,
                true,
                CancellationToken.None,
                progress =>
                {
                    if (progress.Progress01 + 1e-6f < previousInitializationProgress)
                        initializationProgressMonotonic = false;
                    previousInitializationProgress = progress.Progress01;
                    initializationProgressCallbacks.Add(Qwen35ProgressToJson(progress));
                }))
            {
                var generated = await runner.GenerateImageAsync(
                    image,
                    prompt,
                    Qwen35SamplingConfig.Greedy(),
                    CancellationToken.None,
                    (tokenId, piece) =>
                    {
                        streamedTokenIds.Add(tokenId);
                        if (!string.IsNullOrEmpty(piece))
                            streamedText.Append(piece);
                    },
                    (completed, total) =>
                    {
                        if (completed <= previousCompleted)
                            progressMonotonic = false;
                        previousCompleted = completed;
                        if (total != maxNewTokens)
                            progressTotalConsistent = false;
                        progressCallbacks.Add(new JObject
                        {
                            ["completed"] = completed,
                            ["total"] = total
                        });
                    },
                    stage => stageCallbacks.Add(stage ?? string.Empty),
                    progress =>
                    {
                        if (progress.Progress01 + 1e-6f < previousPipelineProgress)
                            pipelineProgressMonotonic = false;
                        previousPipelineProgress = progress.Progress01;
                        pipelineStages.Add(progress.Stage ?? string.Empty);
                        pipelineProgressCallbacks.Add(Qwen35ProgressToJson(progress));
                    });

                generatedText = generated.Text ?? string.Empty;
                promptTokenCount = generated.PromptTokenCount;
                visionTokenCount = generated.VisionTokenCount;
                expandedPromptTokenCount = generated.ExpandedPromptTokenCount;
                finalCacheTextureCount = generated.FinalCacheTextureCount;
                decoderRuns = generated.DecoderStepCount;
                stoppedOnEndOfTurn = generated.StoppedOnEndOfTurn;
                for (var i = 0; i < generated.TokenIds.Count; i++)
                    generatedTokenIds.Add(generated.TokenIds[i]);

                var expectedStages = new[] { "loading_vision", "encoding_image", "loading_decoder", "generating" };
                var stageSequenceMatches = stageCallbacks.Count == expectedStages.Length;
                for (var i = 0; stageSequenceMatches && i < expectedStages.Length; i++)
                    stageSequenceMatches = string.Equals((string)stageCallbacks[i], expectedStages[i], StringComparison.Ordinal);
                var callbackCountsMatch = generated.TokenIds.Count == streamedTokenIds.Count
                    && generated.TokenIds.Count == progressCallbacks.Count;
                firstTokenIsKnown = generated.TokenIds.Count > 0
                    && generated.TokenIds[0] >= 0
                    && generated.TokenIds[0] < runner.Tokenizer.VocabularySize;
                expectedTokenIdsMatched = expectedTokenIds.Count == 0 || expectedTokenIds.Count <= generated.TokenIds.Count;
                if (expectedTokenIdsMatched)
                {
                    for (var tokenIndex = 0; tokenIndex < expectedTokenIds.Count; tokenIndex++)
                    {
                        if (generated.TokenIds[tokenIndex] != (int)expectedTokenIds[tokenIndex])
                        {
                            expectedTokenIdsMatched = false;
                            break;
                        }
                    }
                }
                var detailedProgressStages = new[]
                {
                    "loading_vision",
                    "encoding_image",
                    "loading_decoder",
                    "prefill",
                    "generating",
                    "complete"
                };
                var detailedProgressStagesCovered = true;
                for (var i = 0; i < detailedProgressStages.Length; i++)
                    detailedProgressStagesCovered &= pipelineStages.Contains(detailedProgressStages[i]);
                valid = stageSequenceMatches
                    && callbackCountsMatch
                    && progressMonotonic
                    && progressTotalConsistent
                    && initializationProgressMonotonic
                    && pipelineProgressMonotonic
                    // Runner initialization reports asset/contract/tokenizer progress;
                    // vision and decoder load progress is intentionally reported through
                    // the pipeline callback below. The compact initialization contract
                    // currently contains eleven events.
                    && initializationProgressCallbacks.Count >= 10
                    && pipelineProgressCallbacks.Count >= 100
                    && previousInitializationProgress >= 0.999f
                    && previousPipelineProgress >= 0.999f
                    && detailedProgressStagesCovered
                    && previousCompleted == generated.TokenIds.Count
                    && firstTokenIsKnown
                    && (expectedTokenIds.Count == 0 || expectedTokenIdsMatched)
                    && !string.IsNullOrWhiteSpace(generatedText)
                    && finalCacheTextureCount == 48
                    && decoderRuns > 0
                    && promptTokenCount > 0
                    && visionTokenCount > 0
                    && expandedPromptTokenCount == promptTokenCount - 1 + visionTokenCount;
            }
        }
        catch (Exception exception)
        {
            error = exception.ToString();
        }
        finally
        {
            if (image != null)
                UnityEngine.Object.DestroyImmediate(image);
        }

        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.async-multimodal-generation/v1",
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["model_directory"] = modelDir,
            ["image"] = imagePath,
            ["image_sha256"] = File.Exists(imagePath) ? ComputeQwen35Sha256(imagePath) : string.Empty,
            ["prompt"] = prompt,
            ["max_new_tokens"] = maxNewTokens,
            ["stage_callbacks"] = stageCallbacks,
            ["initialization_progress_callbacks"] = initializationProgressCallbacks,
            ["pipeline_progress_callbacks"] = pipelineProgressCallbacks,
            ["progress_callbacks"] = progressCallbacks,
            ["progress_monotonic"] = progressMonotonic,
            ["progress_total_consistent"] = progressTotalConsistent,
            ["initialization_progress_monotonic"] = initializationProgressMonotonic,
            ["pipeline_progress_monotonic"] = pipelineProgressMonotonic,
            ["initialization_progress_callback_count"] = initializationProgressCallbacks.Count,
            ["pipeline_progress_callback_count"] = pipelineProgressCallbacks.Count,
            ["streamed_token_ids"] = streamedTokenIds,
            ["streamed_text"] = streamedText.ToString(),
            ["generated_token_ids"] = generatedTokenIds,
            ["expected_token_ids"] = expectedTokenIds,
            ["expected_token_ids_matched"] = expectedTokenIds.Count == 0 || expectedTokenIdsMatched,
            ["first_token_is_known"] = firstTokenIsKnown,
            ["generated_text"] = generatedText,
            ["prompt_token_count"] = promptTokenCount,
            ["vision_token_count"] = visionTokenCount,
            ["expanded_prompt_token_count"] = expandedPromptTokenCount,
            ["decoder_runs"] = decoderRuns,
            ["cache_texture_count"] = finalCacheTextureCount,
            ["stopped_on_end_of_turn"] = stoppedOnEndOfTurn,
            ["strict_texture_execution"] = true,
            ["activation_storage"] = "texture-backed",
            ["compute_buffer_fallback"] = false,
            ["exit_code"] = valid && string.IsNullOrEmpty(error) ? 0 : 1,
            ["error"] = error ?? string.Empty,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35AsyncMultimodalGenerationBatch",
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen35_async_multimodal_smoke.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] async multimodal generation report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen3.5 async multimodal generation failed; see " + outputPath);
    }

    private static JObject Qwen35ProgressToJson(Qwen35Progress progress)
    {
        return new JObject
        {
            ["stage"] = progress.Stage ?? string.Empty,
            ["detail"] = progress.Detail ?? string.Empty,
            ["progress01"] = progress.Progress01,
            ["completed"] = progress.Completed,
            ["total"] = progress.Total
        };
    }

    private static JObject Qwen35LoadProfileToJson(AexisGraphSession.ModelLoadProfile profile)
    {
        if (profile == null)
            return new JObject { ["available"] = false };

        var layerTypes = new JObject();
        foreach (var pair in profile.layerTypes)
        {
            var metrics = pair.Value;
            layerTypes[pair.Key] = new JObject
            {
                ["count"] = metrics.count,
                ["total_ms"] = metrics.totalMs,
                ["bytes_read"] = metrics.bytesRead,
                ["read_ms"] = metrics.readMs,
                ["upload_ms"] = metrics.uploadMs,
                ["pack_ms"] = metrics.packMs
            };
        }
        return new JObject
        {
            ["available"] = true,
            ["layer_count"] = profile.layerCount,
            ["release_ms"] = profile.releaseMs,
            ["parse_param_ms"] = profile.parseParamMs,
            ["build_blob_use_count_ms"] = profile.buildBlobUseCountMs,
            ["total_ms"] = profile.totalMs,
            ["total_bytes_read"] = profile.totalBytesRead,
            ["managed_cleanup_count"] = profile.managedCleanupCount,
            ["managed_cleanup_ms"] = profile.managedCleanupMs,
            ["layer_types"] = layerTypes
        };
    }

    private static JObject Qwen35RuntimeProfileToJson(AexisGraphSession.LayerRuntimeProfile profile)
    {
        if (profile == null)
            return new JObject { ["available"] = false };

        var layerTypes = new JObject();
        foreach (var pair in profile.layerTypes.OrderByDescending(pair => pair.Value.totalTicks))
        {
            var metrics = pair.Value;
            layerTypes[pair.Key] = new JObject
            {
                ["count"] = metrics.count,
                ["total_ms"] = metrics.totalMs,
                ["average_ms"] = metrics.avgMs
            };
        }

        var layers = new JArray();
        foreach (var layer in profile.layers)
        {
            layers.Add(new JObject
            {
                ["index"] = layer.layerIndex,
                ["name"] = layer.layerName ?? string.Empty,
                ["type"] = layer.layerType ?? string.Empty,
                ["path"] = layer.path ?? string.Empty,
                ["elapsed_ms"] = layer.elapsedMs
            });
        }

        return new JObject
        {
            ["available"] = true,
            ["inference_index"] = profile.inferenceIndex,
            ["path_kind"] = profile.pathKind ?? string.Empty,
            ["synchronized_gpu"] = profile.syncGpu,
            ["layer_count"] = profile.layers.Count,
            ["total_ms"] = profile.totalMs,
            ["layer_types"] = layerTypes,
            ["layers"] = layers
        };
    }

    public static void RunQwen35ContractBatch()
    {
        var validationStart = Stopwatch.StartNew();
        var modelDir = ResolveQwen35ModelDirectory(Directory.GetParent(Application.dataPath).FullName, "qwen3.5_0.8b");
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_CONTRACT_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Tools", "Qwen35NcnnBaseline", "reports", "unity_contract_validation.json");
        var contract = Qwen35ModelContract.Validate(modelDir, requireWeights: true);
        var registration = AexisLayerFactory.IsRegistered(AexisLayerTypes.ShortConv)
            && AexisLayerFactory.IsRegistered(AexisLayerTypes.GatedDeltaRule);
        var shaderKernels = false;
        try
        {
            using (var ops = new AexisOps())
                shaderKernels = true;
        }
        catch (Exception shaderError)
        {
            contract.Errors.Add("Qwen35 compute shader kernel load failed: " + shaderError.Message);
        }
        var includeHashes = string.Equals(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_CONTRACT_HASHES"), "1", StringComparison.Ordinal);
        var report = contract.ToJson(includeHashes: includeHashes);
        report["hashes_skipped"] = !includeHashes;
        report["custom_layer_registration"] = registration;
        report["shader_kernel_registration"] = shaderKernels;
        report["strict_texture_execution"] = true;
        report["native_fallback"] = false;
        var manifestPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Tools", "Qwen35NcnnBaseline", "reports", "qwen35_0_8b_compare_manifest.json");
        var compareManifestAvailable = File.Exists(manifestPath);
        var requireCompareManifest = string.Equals(
            Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_REQUIRE_COMPARE_MANIFEST"),
            "1",
            StringComparison.Ordinal);
        var compareManifest = Qwen35CompareManifest.Load(manifestPath);
        var compareManifestReport = compareManifest.ToJson();
        compareManifestReport["available"] = compareManifestAvailable;
        compareManifestReport["required"] = requireCompareManifest;
        report["compare_manifest"] = compareManifestReport;
        var catalog = Qwen35NetworkAssetCatalog.Create(contract);
        var catalogErrors = catalog.ValidateFiles();
        report["network_asset_catalog"] = new JObject
        {
            ["network_count"] = catalog.Networks.Length,
            ["single_shared_token_bin"] = catalog.UsesSingleTokenEmbeddingBin,
            ["errors"] = new JArray(catalogErrors)
        };
        var quantManifest = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_QUANT_MANIFEST");
        report["mobile_memory_policy"] = Qwen35MobileMemoryPolicy.Evaluate(contract, quantManifest).ToJson();
        validationStart.Stop();
        report["validation"] = new JObject
        {
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35ContractBatch",
            ["exit_code"] = 0,
            ["elapsed_ms"] = validationStart.ElapsedMilliseconds,
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen_final_contract.log",
            ["stdout"] = "Unity log",
            ["stderr"] = "Unity log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] contract report: " + outputPath + " valid=" + (contract.IsValid && registration));
        if (!contract.IsValid
            || !registration
            || !shaderKernels
            || catalogErrors.Count != 0
            || (compareManifestAvailable && !compareManifest.IsContractValid)
            || (requireCompareManifest && !compareManifestAvailable))
            throw new InvalidOperationException("Qwen35 contract validation failed: " + string.Join("; ", contract.Errors));
        EditorApplication.Exit(0);
    }

    public static void RunQwen35NetworkLoadBatch()
    {
        var start = Stopwatch.StartNew();
        var modelDir = ResolveQwen35ModelDirectory(Directory.GetParent(Application.dataPath).FullName, "qwen3.5_0.8b");
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_NETWORK_LOAD_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Tools", "Qwen35NcnnBaseline", "reports", "unity_network_load_validation.json");

        var contract = Qwen35ModelContract.Validate(modelDir, requireWeights: true);
        var report = contract.ToJson(includeHashes: string.Equals(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_CONTRACT_HASHES"), "1", StringComparison.Ordinal));
        var catalog = Qwen35NetworkAssetCatalog.Create(contract);
        var catalogErrors = catalog.ValidateFiles();
        report["network_asset_catalog"] = new JObject
        {
            ["network_count"] = catalog.Networks.Length,
            ["single_shared_token_bin"] = catalog.UsesSingleTokenEmbeddingBin,
            ["errors"] = new JArray(catalogErrors)
        };

        var results = new JArray();
        var loadValid = contract.IsValid && catalogErrors.Count == 0;
        if (loadValid)
        {
            using (var loader = new Qwen35NetworkLoader())
            {
                var loaded = loader.ValidateAllSequential(catalog, (network, progress) =>
                {
                    if (progress.stage == "layer" && (progress.layerIndex == 1 || progress.layerIndex == progress.layerCount || progress.layerIndex % 100 == 0))
                        Debug.Log("[Qwen35] load " + network + " " + progress.layerIndex + "/" + progress.layerCount + " " + progress.layerType + " " + progress.layerName);
                });
                for (var i = 0; i < loaded.Count; i++)
                {
                    var item = loaded[i];
                    var expected = NcnnParamParser.Parse(File.ReadAllText(catalog.Networks[i].ParamPath));
                    var exactShape = item.Success && item.LayerCount == expected.layerCount && item.BlobCount == expected.blobCount;
                    if (!exactShape)
                        loadValid = false;
                    results.Add(new JObject
                    {
                        ["name"] = item.Name,
                        ["success"] = item.Success,
                        ["layer_count"] = item.LayerCount,
                        ["blob_count"] = item.BlobCount,
                        ["expected_layer_count"] = expected.layerCount,
                        ["expected_blob_count"] = expected.blobCount,
                        ["exact_param_contract"] = exactShape,
                        ["weight_bytes"] = item.WeightBytes,
                        ["elapsed_ms"] = item.ElapsedMilliseconds,
                        ["error"] = item.Error ?? string.Empty
                    });
                    Debug.Log("[Qwen35] network=" + item.Name + " success=" + item.Success + " layers=" + item.LayerCount + " blobs=" + item.BlobCount + " elapsed_ms=" + item.ElapsedMilliseconds + (item.Success ? string.Empty : " error=" + item.Error));
                }
            }
        }
        else
        {
            loadValid = false;
        }

        start.Stop();
        report["network_load"] = new JObject
        {
            ["valid"] = loadValid,
            ["sequential_release"] = true,
            ["results"] = results
        };
        report["validation"] = new JObject
        {
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35NetworkLoadBatch",
            ["exit_code"] = loadValid ? 0 : 1,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen_network_load.log",
            ["stdout"] = "Unity log",
            ["stderr"] = "Unity log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] network load report: " + outputPath + " valid=" + loadValid);
        if (!loadValid)
            throw new InvalidOperationException("Qwen35 network load validation failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35EmbedProbeBatch()
    {
        var start = Stopwatch.StartNew();
        var modelDir = ResolveQwen35ModelDirectory(Directory.GetParent(Application.dataPath).FullName, "qwen3.5_0.8b");
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_EMBED_PROBE_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Tools", "Qwen35NcnnBaseline", "reports", "unity_embed_probe.json");
        var paramPath = Path.Combine(modelDir, "qwen3.5_embed_token.ncnn.param");
        var binPath = Path.Combine(modelDir, "qwen3.5_embed_token.ncnn.bin");
        var tokenId = 0;
        if (int.TryParse(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_PROBE_TOKEN"), out var parsedToken))
            tokenId = parsedToken;

        var values = Array.Empty<float>();
        var valid = false;
        string error = null;
        try
        {
            using (var ops = new AexisOps())
            using (var repro = new AexisGraphSession(ops))
            using (var indices = new AexisTensorBuffer(1, 1))
            {
                repro.ExecutionMode = AexisInferenceExecutionMode.ProductionTextureOnly;
                repro.DisallowInferenceTempComputeBuffers = true;
                repro.DisallowBufferToTextureMaterialization = true;
                indices.buffer.SetData(new[] { tokenId });
                using (var stream = File.OpenRead(binPath))
                using (var reader = new NcnnBinReader(stream))
                {
                    repro.LoadModel(File.ReadAllText(paramPath), reader);
                }
                using (var result = repro.InferWithMultiInputs(
                    null,
                    new Dictionary<string, AexisTensorBuffer>(StringComparer.Ordinal) { ["in0"] = indices }))
                {
                    values = result.ReadTextureDataForOutput("out0");
                    valid = values != null && values.Length >= 1024;
                }
            }
        }
        catch (Exception e)
        {
            error = e.ToString();
        }

        start.Stop();
        var preview = new JArray();
        for (var i = 0; i < Math.Min(16, values.Length); i++)
            preview.Add(values[i]);
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.embed-probe/v1",
            ["model_directory"] = modelDir,
            ["token_id"] = tokenId,
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["output_count"] = values.Length,
            ["preview"] = preview,
            ["error"] = error ?? string.Empty,
            ["strict_texture_execution"] = true,
            ["temporary_compute_buffer_execution"] = false,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen_embed_probe.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] embed probe report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen35 embed probe failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35DecoderPrefixProbeBatch()
    {
        var start = Stopwatch.StartNew();
        var modelDir = ResolveQwen35ModelDirectory(Directory.GetParent(Application.dataPath).FullName, "qwen3.5_0.8b");
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_DECODER_PROBE_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Tools", "Qwen35NcnnBaseline", "reports", "unity_decoder_prefix_probe.json");
        var stopAfter = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_DECODER_STOP_AFTER");
        if (string.IsNullOrWhiteSpace(stopAfter))
            stopAfter = "out_cache_gdr2";
        var runProjOut = string.Equals(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_RUN_PROJ_OUT"), "1", StringComparison.OrdinalIgnoreCase)
            || string.Equals(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_RUN_PROJ_OUT"), "true", StringComparison.OrdinalIgnoreCase);

        var valid = false;
        string error = null;
        var outputCount = 0;
        var outputMaxAbs = 0f;
        var outputNonFiniteCount = 0;
        var greedyToken = -1;
        var logitsWidth = 0;
        var logitsHeight = 0;
        long sharedWeightBytes = 0;
        var preview = new JArray();
        try
        {
            using (var shared = Qwen35SharedTokenEmbeddingWeights.Load(Path.Combine(modelDir, "qwen3.5_embed_token.ncnn.bin")))
            using (var ops = new AexisOps())
            using (var embed = new AexisGraphSession(ops))
            using (var decoder = new AexisGraphSession(ops))
            using (var indices = new AexisTensorBuffer(1, 1))
            {
                ConfigureQwen35StrictRepro(embed);
                ConfigureQwen35StrictRepro(decoder);
                if (string.Equals(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_DECODER_LAYER_LOG"), "1", StringComparison.Ordinal))
                    decoder.DebugLog = line => Debug.Log("[Qwen35][DecoderPrefix] " + line);
                shared.Attach(embed);
                sharedWeightBytes = shared.ByteCount;
                embed.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                decoder.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                indices.buffer.SetData(new[] { 0 });

                using (var embedStream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_embed_token.ncnn.bin")))
                using (var embedReader = new NcnnBinReader(embedStream))
                    embed.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_embed_token.ncnn.param")), embedReader);

                RenderTexture embedding;
                using (var embeddingResult = embed.InferWithMultiInputs(
                    null,
                    new Dictionary<string, AexisTensorBuffer>(StringComparer.Ordinal) { ["in0"] = indices }))
                {
                    embedding = embeddingResult.ExtractTexture("out0");
                }

                using (var decoderStream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_decoder.ncnn.bin")))
                using (var decoderReader = new NcnnBinReader(decoderStream))
                    decoder.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_decoder.ncnn.param")), decoderReader);

                var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
                {
                    ["in0"] = embedding,
                    ["in1"] = CreateQwen35ZeroTexture(decoder, ops, 1, 1, 1),
                    ["in2"] = CreateQwen35CosSinTexture(decoder, ops, 32, 1f),
                    ["in3"] = CreateQwen35CosSinTexture(decoder, ops, 32, 0f),
                    ["cache_conv0"] = CreateQwen35ZeroTexture(decoder, ops, 1536, 4, 1),
                    ["cache_conv1"] = CreateQwen35ZeroTexture(decoder, ops, 1536, 4, 1),
                    ["cache_conv2"] = CreateQwen35ZeroTexture(decoder, ops, 1536, 4, 1),
                    ["cache_gdr0"] = CreateQwen35ZeroTexture(decoder, ops, 32, 128, 16),
                    ["cache_gdr1"] = CreateQwen35ZeroTexture(decoder, ops, 32, 128, 16),
                    ["cache_gdr2"] = CreateQwen35ZeroTexture(decoder, ops, 32, 128, 16)
                };
                var textureShapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                {
                    ["in0"] = new AexisGraphSession.BufferShape(2, 1024, 1, 1, 1),
                    ["in1"] = new AexisGraphSession.BufferShape(2, 1, 1, 1, 1),
                    ["in2"] = new AexisGraphSession.BufferShape(3, 32, 1, 1, 1),
                    ["in3"] = new AexisGraphSession.BufferShape(3, 32, 1, 1, 1),
                    ["cache_conv0"] = new AexisGraphSession.BufferShape(2, 1536, 4, 1, 1),
                    ["cache_conv1"] = new AexisGraphSession.BufferShape(2, 1536, 4, 1, 1),
                    ["cache_conv2"] = new AexisGraphSession.BufferShape(2, 1536, 4, 1, 1),
                    // GDR state is stored as value_dim/4 texture columns; the custom layer
                    // addresses it in this packed storage shape directly.
                    ["cache_gdr0"] = new AexisGraphSession.BufferShape(3, 32, 128, 1, 16),
                    ["cache_gdr1"] = new AexisGraphSession.BufferShape(3, 32, 128, 1, 16),
                    ["cache_gdr2"] = new AexisGraphSession.BufferShape(3, 32, 128, 1, 16)
                };
                for (var cacheIndex = 3; cacheIndex < 18; cacheIndex++)
                {
                    var convName = "cache_conv" + cacheIndex;
                    var gdrName = "cache_gdr" + cacheIndex;
                    textureInputs[convName] = CreateQwen35ZeroTexture(decoder, ops, 1536, 4, 1);
                    textureInputs[gdrName] = CreateQwen35ZeroTexture(decoder, ops, 32, 128, 16);
                    textureShapes[convName] = new AexisGraphSession.BufferShape(2, 1536, 4, 1, 1);
                    textureShapes[gdrName] = new AexisGraphSession.BufferShape(3, 32, 128, 1, 16);
                }

                RenderTexture decoderOutput = null;
                using (var result = decoder.InferWithMultiInputs(
                    textureInputs,
                    null,
                    null,
                    textureShapes,
                    stopAfter))
                {
                    if (runProjOut)
                    {
                        if (!string.Equals(stopAfter, "out0", StringComparison.Ordinal))
                            throw new InvalidOperationException("proj_out chaining requires decoder stop blob out0");
                        decoderOutput = result.ExtractTexture(stopAfter);
                    }
                    else
                    {
                        var output = result.ReadTextureDataForOutput(stopAfter);
                        outputCount = output != null ? output.Length : 0;
                        for (var i = 0; i < outputCount; i++)
                        {
                            if (float.IsNaN(output[i]) || float.IsInfinity(output[i]))
                                outputNonFiniteCount++;
                            else
                                outputMaxAbs = Mathf.Max(outputMaxAbs, Mathf.Abs(output[i]));
                            if (i < 16)
                                preview.Add(output[i]);
                        }
                        valid = outputCount > 0 && outputNonFiniteCount == 0 && outputMaxAbs > 0f;
                    }
                }
                if (runProjOut)
                {
                    using (var proj = new AexisGraphSession(ops))
                    {
                        ConfigureQwen35StrictRepro(proj);
                        shared.Attach(proj);
                        proj.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                        using (var projStream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_embed_token.ncnn.bin")))
                        using (var projReader = new NcnnBinReader(projStream))
                            proj.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_proj_out.ncnn.param")), projReader);
                        using (var projResult = proj.InferWithMultiInputs(
                            new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = decoderOutput },
                            null,
                            null,
                            new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                            {
                                ["in0"] = new AexisGraphSession.BufferShape(2, 1024, 1, 1, 1)
                            }))
                        {
                            var logitsTexture = projResult.GetTexture("out0");
                            logitsWidth = logitsTexture.width;
                            logitsHeight = logitsTexture.height;
                            var logits = projResult.ReadTextureDataForOutput("out0");
                            outputCount = logits != null ? logits.Length : 0;
                            var bestValue = float.NegativeInfinity;
                            for (var i = 0; i < outputCount; i++)
                            {
                                if (!float.IsNaN(logits[i]) && logits[i] > bestValue)
                                {
                                    bestValue = logits[i];
                                    greedyToken = i;
                                }
                            }
                            for (var i = 0; i < Math.Min(16, outputCount); i++)
                                preview.Add(logits[i]);
                            valid = outputCount == 248320 && greedyToken >= 0;
                        }
                    }
                    if (decoderOutput != null)
                    {
                        RenderTexture.ReleaseTemporary(decoderOutput);
                        decoderOutput = null;
                    }
                }
                foreach (var texture in textureInputs.Values)
                {
                    if (texture != null && texture != embedding)
                        decoder.ReturnTempArray(texture);
                }
                if (embedding != null)
                    RenderTexture.ReleaseTemporary(embedding);
            }
        }
        catch (Exception e)
        {
            error = e.ToString();
        }

        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.decoder-prefix-probe/v1",
            ["model_directory"] = modelDir,
            ["stop_after"] = stopAfter,
            ["run_proj_out"] = runProjOut,
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["output_count"] = outputCount,
            ["output_max_abs"] = outputMaxAbs,
            ["output_nonfinite_count"] = outputNonFiniteCount,
            ["greedy_token"] = greedyToken,
            ["logits_texture_width"] = logitsWidth,
            ["logits_texture_height"] = logitsHeight,
            ["shared_weight_bytes"] = sharedWeightBytes,
            ["shared_gpu_weight_instances"] = 1,
            ["preview"] = preview,
            ["error"] = error ?? string.Empty,
            ["strict_texture_execution"] = true,
            ["compute_buffer_fallback"] = false,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen_decoder_prefix_probe.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] decoder prefix probe report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen35 decoder prefix probe failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35ProjOutProbeBatch()
    {
        var start = Stopwatch.StartNew();
        var modelDir = ResolveQwen35ModelDirectory(Directory.GetParent(Application.dataPath).FullName, "qwen3.5_0.8b");
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_PROJ_PROBE_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Tools", "Qwen35NcnnBaseline", "reports", "unity_proj_out_probe.json");

        var valid = false;
        string error = null;
        var logitsCount = 0;
        var greedyToken = -1;
        var logitsWidth = 0;
        var logitsHeight = 0;
        var maxAbsLogit = 0f;
        long sharedWeightBytes = 0;
        long sharedLoadMs = 0;
        var preview = new JArray();
        try
        {
            using (var shared = Qwen35SharedTokenEmbeddingWeights.Load(Path.Combine(modelDir, "qwen3.5_embed_token.ncnn.bin")))
            using (var ops = new AexisOps())
            using (var proj = new AexisGraphSession(ops))
            {
                sharedWeightBytes = shared.ByteCount;
                sharedLoadMs = shared.LoadMilliseconds;
                shared.Attach(proj);
                ConfigureQwen35StrictRepro(proj);
                proj.TensorTextureFormat = RenderTextureFormat.ARGBFloat;

                using (var stream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_embed_token.ncnn.bin")))
                using (var reader = new NcnnBinReader(stream))
                    proj.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_proj_out.ncnn.param")), reader);

                var hidden = CreateQwen35ZeroTexture(proj, ops, 256, 1, 1);
                ops.FillScalarTexture(new[] { 1f }, hidden);
                using (var result = proj.InferWithMultiInputs(
                    new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = hidden },
                    null,
                    null,
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["in0"] = new AexisGraphSession.BufferShape(2, 1024, 1, 1, 1)
                    }))
                {
                    var logitsTexture = result.GetTexture("out0");
                    logitsWidth = logitsTexture.width;
                    logitsHeight = logitsTexture.height;
                    var logits = result.ReadTextureDataForOutput("out0");
                    logitsCount = logits != null ? logits.Length : 0;
                    var bestValue = float.NegativeInfinity;
                    for (var i = 0; i < logitsCount; i++)
                    {
                        if (!float.IsNaN(logits[i]) && logits[i] > bestValue)
                        {
                            bestValue = logits[i];
                            greedyToken = i;
                        }
                        if (!float.IsNaN(logits[i]) && !float.IsInfinity(logits[i]))
                            maxAbsLogit = Mathf.Max(maxAbsLogit, Mathf.Abs(logits[i]));
                    }
                    for (var i = 0; i < Math.Min(16, logitsCount); i++)
                        preview.Add(logits[i]);
                    valid = logitsCount == 248320
                        && greedyToken >= 0
                        && maxAbsLogit > 0f
                        && logitsWidth <= SystemInfo.maxTextureSize
                        && logitsHeight <= SystemInfo.maxTextureSize;
                }
                proj.ReturnTempArray(hidden);
            }
        }
        catch (Exception e)
        {
            error = e.ToString();
        }

        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.proj-out-probe/v1",
            ["model_directory"] = modelDir,
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["logits_count"] = logitsCount,
            ["greedy_token"] = greedyToken,
            ["logits_texture_width"] = logitsWidth,
            ["logits_texture_height"] = logitsHeight,
            ["max_abs_logit"] = maxAbsLogit,
            ["shared_weight_bytes"] = sharedWeightBytes,
            ["shared_gpu_weight_instances"] = 1,
            ["shared_weight_load_ms"] = sharedLoadMs,
            ["preview"] = preview,
            ["error"] = error ?? string.Empty,
            ["strict_texture_execution"] = true,
            ["compute_buffer_fallback"] = false,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen_proj_out_probe.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] proj_out probe report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen35 proj_out probe failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35VisionPatchProbeBatch()
    {
        var start = Stopwatch.StartNew();
        var projectRoot = Directory.GetParent(Application.dataPath).FullName;
        var modelDir = ResolveQwen35ModelDirectory(projectRoot, "qwen3.5_0.8b");
        var imagePath = ResolveQwen35ImagePath(projectRoot);
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_VISION_PATCH_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_vision_patch_probe.json");

        var valid = false;
        string error = null;
        var sourceWidth = 0;
        var sourceHeight = 0;
        var targetWidth = 0;
        var targetHeight = 0;
        var patchCount = 0;
        var outputCount = 0;
        var finiteCount = 0;
        var nonZeroCount = 0;
        var maxAbs = 0f;
        var inputPreview = new JArray();
        var outputPreview = new JArray();
        var patchValues = Array.Empty<float>();
        var outputValues = Array.Empty<float>();
        Texture2D image = null;
        try
        {
            var imageBytes = File.ReadAllBytes(imagePath);
            image = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
            if (!ImageConversion.LoadImage(image, imageBytes, false))
                throw new InvalidDataException("Unity failed to decode Qwen3.5 probe image: " + imagePath);
            sourceWidth = image.width;
            sourceHeight = image.height;
            var target = Qwen35VisionPreprocessor.TargetImageSize(sourceHeight, sourceWidth);
            targetWidth = target.x;
            targetHeight = target.y;
            var normalized = Qwen35VisionPreprocessor.ResizeNormalize(image, targetWidth, targetHeight);
            var patches = Qwen35VisionPreprocessor.BuildDuplicatedPatches(normalized, targetWidth, targetHeight);
            patchCount = (targetWidth / 16) * (targetHeight / 16);
            var patch = new float[16 * 16 * 2 * 3];
            Array.Copy(patches, patch, patch.Length);
            patchValues = patch;
            for (var i = 0; i < Math.Min(16, patch.Length); i++)
                inputPreview.Add(patch[i]);

            using (var ops = new AexisOps())
            using (var repro = new AexisGraphSession(ops))
            {
                ConfigureQwen35StrictRepro(repro);
                repro.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                using (var stream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_vision_embed_patch.ncnn.bin")))
                using (var reader = new NcnnBinReader(stream))
                    repro.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_vision_embed_patch.ncnn.param")), reader);

                var patchTexture = CreateQwen35Pack4TextureFromCdhw(repro, patch, 16, 16, 2, 3);
                try
                {
                    using (var result = repro.InferWithMultiInputs(
                        new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = patchTexture },
                        null,
                        null,
                        new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                        {
                            ["in0"] = new AexisGraphSession.BufferShape(4, 16, 16, 2, 3)
                        }))
                    {
                        var values = result.ReadTextureDataForOutput("out0") ?? Array.Empty<float>();
                        outputValues = values;
                        outputCount = values.Length;
                        for (var i = 0; i < values.Length; i++)
                        {
                            var value = values[i];
                            if (!float.IsNaN(value) && !float.IsInfinity(value)) finiteCount++;
                            if (value != 0f) nonZeroCount++;
                            maxAbs = Mathf.Max(maxAbs, Mathf.Abs(value));
                        }
                        for (var i = 0; i < Math.Min(16, values.Length); i++)
                            outputPreview.Add(values[i]);
                        valid = outputCount == 768 && finiteCount == outputCount && nonZeroCount > 0;
                    }
                }
                finally
                {
                    repro.ReturnTempArray(patchTexture);
                }
            }
        }
        catch (Exception exception)
        {
            error = exception.ToString();
        }
        finally
        {
            if (image != null) UnityEngine.Object.DestroyImmediate(image);
        }

        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.vision-patch-probe/v1",
            ["model_directory"] = modelDir,
            ["image"] = imagePath,
            ["image_sha256"] = File.Exists(imagePath) ? ComputeQwen35Sha256(imagePath) : string.Empty,
            ["source_width"] = sourceWidth,
            ["source_height"] = sourceHeight,
            ["target_width"] = targetWidth,
            ["target_height"] = targetHeight,
            ["patch_count"] = patchCount,
            ["patch_index"] = 0,
            ["patch_input_count"] = 16 * 16 * 2 * 3,
            ["input_preview"] = inputPreview,
            ["input_values"] = new JArray(patchValues),
            ["output_count"] = outputCount,
            ["finite_count"] = finiteCount,
            ["nonzero_count"] = nonZeroCount,
            ["max_abs"] = maxAbs,
            ["output_preview"] = outputPreview,
            ["output_values"] = new JArray(outputValues),
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["error"] = error ?? string.Empty,
            ["strict_texture_execution"] = true,
            ["fixed_input_texture_upload"] = "Texture2DArray.CopyTexture",
            ["compute_buffer_fallback"] = false,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -quit -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35VisionPatchProbeBatch",
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen35_vision_patch_probe.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] vision patch probe report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen35 vision patch probe failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35VisionPositionProbeBatch()
    {
        var start = Stopwatch.StartNew();
        var projectRoot = Directory.GetParent(Application.dataPath).FullName;
        var modelDir = ResolveQwen35ModelDirectory(projectRoot, "qwen3.5_0.8b");
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_VISION_POSITION_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_vision_position_probe.json");
        var dumpPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_VISION_POSITION_DUMP");
        if (string.IsNullOrWhiteSpace(dumpPath))
            dumpPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_vision_position_probe.f32");

        var valid = false;
        string error = null;
        var outputCount = 0;
        var finiteCount = 0;
        var nonZeroCount = 0;
        var maxAbs = 0f;
        var preview = new JArray();
        try
        {
            using (var ops = new AexisOps())
            using (var repro = new AexisGraphSession(ops))
            {
                ConfigureQwen35StrictRepro(repro);
                repro.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                using (var stream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_vision_embed_pos.ncnn.bin")))
                using (var reader = new NcnnBinReader(stream))
                    repro.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_vision_embed_pos.ncnn.param")), reader);

                var grid = CreateQwen35ZeroLinearTexture(repro, ops, 64, 48);
                try
                {
                    using (var result = repro.InferWithMultiInputs(
                        new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = grid },
                        null,
                        null,
                        new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                        {
                            ["in0"] = new AexisGraphSession.BufferShape(2, 64, 48, 1, 1)
                        }))
                    {
                        var values = result.ReadTextureDataForOutput("out0") ?? Array.Empty<float>();
                        outputCount = values.Length;
                        var dumpBytes = new byte[values.Length * sizeof(float)];
                        Buffer.BlockCopy(values, 0, dumpBytes, 0, dumpBytes.Length);
                        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(dumpPath)));
                        File.WriteAllBytes(dumpPath, dumpBytes);
                        for (var i = 0; i < values.Length; i++)
                        {
                            var value = values[i];
                            if (!float.IsNaN(value) && !float.IsInfinity(value)) finiteCount++;
                            if (value != 0f) nonZeroCount++;
                            maxAbs = Mathf.Max(maxAbs, Mathf.Abs(value));
                        }
                        for (var i = 0; i < Math.Min(32, values.Length); i++) preview.Add(values[i]);
                        valid = outputCount == 768 * 3072 && finiteCount == outputCount && nonZeroCount > 0;
                    }
                }
                finally
                {
                    RenderTexture.ReleaseTemporary(grid);
                }
            }
        }
        catch (Exception exception)
        {
            error = exception.ToString();
        }

        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.vision-position-probe/v1",
            ["model_directory"] = modelDir,
            ["grid_width"] = 64,
            ["grid_height"] = 48,
            ["output_count"] = outputCount,
            ["finite_count"] = finiteCount,
            ["nonzero_count"] = nonZeroCount,
            ["max_abs"] = maxAbs,
            ["preview"] = preview,
            ["fp32_dump"] = dumpPath,
            ["fp32_dump_sha256"] = File.Exists(dumpPath) ? ComputeQwen35Sha256(dumpPath) : string.Empty,
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["error"] = error ?? string.Empty,
            ["strict_texture_execution"] = true,
            ["memory_data_storage"] = "load-time pack4 RenderTexture",
            ["compute_buffer_fallback"] = false,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -quit -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35VisionPositionProbeBatch",
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen35_vision_position_probe.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] vision position probe report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen35 vision position probe failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35VisionPatchAtlasProbeBatch()
    {
        var start = Stopwatch.StartNew();
        var projectRoot = Directory.GetParent(Application.dataPath).FullName;
        var modelDir = ResolveQwen35ModelDirectory(projectRoot, "qwen3.5_0.8b");
        var imagePath = ResolveQwen35ImagePath(projectRoot);
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_VISION_PATCH_ATLAS_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_vision_patch_atlas_probe.json");
        var dumpPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_VISION_PATCH_ATLAS_DUMP");
        if (string.IsNullOrWhiteSpace(dumpPath))
            dumpPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_vision_patch_atlas_probe.f32");

        var valid = false;
        string error = null;
        var sourceWidth = 0;
        var sourceHeight = 0;
        var targetWidth = 0;
        var targetHeight = 0;
        var patchCount = 0;
        var outputCount = 0;
        var finiteCount = 0;
        var nonZeroCount = 0;
        var maxAbs = 0f;
        var selected = new JObject();
        Texture2D image = null;
        try
        {
            image = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
            if (!ImageConversion.LoadImage(image, File.ReadAllBytes(imagePath), false))
                throw new InvalidDataException("Unity failed to decode Qwen3.5 atlas image: " + imagePath);
            sourceWidth = image.width;
            sourceHeight = image.height;
            var target = Qwen35VisionPreprocessor.TargetImageSize(sourceHeight, sourceWidth);
            targetWidth = target.x;
            targetHeight = target.y;
            var normalized = Qwen35VisionPreprocessor.ResizeNormalize(image, targetWidth, targetHeight);
            var patches = Qwen35VisionPreprocessor.BuildDuplicatedPatches(normalized, targetWidth, targetHeight);
            patchCount = (targetWidth / 16) * (targetHeight / 16);

            using (var ops = new AexisOps())
            using (var patchRepro = new AexisGraphSession(ops))
            {
                ConfigureQwen35StrictRepro(patchRepro);
                patchRepro.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                using (var stream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_vision_embed_patch.ncnn.bin")))
                using (var reader = new NcnnBinReader(stream))
                    patchRepro.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_vision_embed_patch.ncnn.param")), reader);

                var atlas = CreateQwen35PatchAtlasTexture(patchRepro, patches, targetWidth, targetHeight, 16, 2, 3);
                RenderTexture spatialOutput = null;
                RenderTexture linearOutput = null;
                try
                {
                    using (var result = patchRepro.InferWithMultiInputs(
                        new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = atlas },
                        null,
                        null,
                        new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                        {
                            ["in0"] = new AexisGraphSession.BufferShape(4, targetWidth, targetHeight, 2, 3)
                        }))
                    {
                        spatialOutput = result.ExtractTexture("out0");
                    }

                    linearOutput = patchRepro.RentTempArray(768 / 4, patchCount, 1, RenderTextureFormat.ARGBFloat);
                    ops.Pack4SpatialToPack4Linear(spatialOutput, linearOutput);

                    const string identityParam = "7767517\n2 2\nInput in0 0 1 in0\nNoop copy 1 1 in0 out0\n";
                    using (var viewRepro = new AexisGraphSession(ops))
                    using (var emptyStream = new MemoryStream())
                    using (var emptyReader = new NcnnBinReader(emptyStream))
                    {
                        ConfigureQwen35StrictRepro(viewRepro);
                        viewRepro.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                        viewRepro.LoadModel(identityParam, emptyReader);
                        using (var result = viewRepro.InferWithMultiInputs(
                            new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = linearOutput },
                            null,
                            null,
                            new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                            {
                                ["in0"] = new AexisGraphSession.BufferShape(2, 768, patchCount, 1, 1)
                            }))
                        {
                            var values = result.ReadTextureDataForOutput("out0") ?? Array.Empty<float>();
                            outputCount = values.Length;
                            var dumpBytes = new byte[values.Length * sizeof(float)];
                            Buffer.BlockCopy(values, 0, dumpBytes, 0, dumpBytes.Length);
                            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(dumpPath)));
                            File.WriteAllBytes(dumpPath, dumpBytes);
                            for (var i = 0; i < values.Length; i++)
                            {
                                var value = values[i];
                                if (!float.IsNaN(value) && !float.IsInfinity(value)) finiteCount++;
                                if (value != 0f) nonZeroCount++;
                                maxAbs = Mathf.Max(maxAbs, Mathf.Abs(value));
                            }
                            // Atlas size follows the input aspect ratio; a fixed preview row can be out of range.
                            if (patchCount > 0 && values.Length >= patchCount * 768)
                            {
                                var selectedIndices = new[] { 0, 1, 2, 3, Math.Min(1024, patchCount - 1), patchCount - 1 };
                                for (var index = 0; index < selectedIndices.Length; index++)
                                {
                                    var patchIndex = selectedIndices[index];
                                    var row = new JArray();
                                    for (var feature = 0; feature < 16; feature++)
                                        row.Add(values[patchIndex * 768 + feature]);
                                    selected[patchIndex.ToString(CultureInfo.InvariantCulture)] = row;
                                }
                            }
                            valid = outputCount == patchCount * 768 && finiteCount == outputCount && nonZeroCount > 0;
                        }
                    }
                }
                finally
                {
                    patchRepro.ReturnTempArray(atlas);
                    if (spatialOutput != null) patchRepro.ReturnTempArray(spatialOutput);
                    if (linearOutput != null) patchRepro.ReturnTempArray(linearOutput);
                }
            }
        }
        catch (Exception exception)
        {
            error = exception.ToString();
        }
        finally
        {
            if (image != null) UnityEngine.Object.DestroyImmediate(image);
        }

        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.vision-patch-atlas-probe/v1",
            ["model_directory"] = modelDir,
            ["image"] = imagePath,
            ["image_sha256"] = File.Exists(imagePath) ? ComputeQwen35Sha256(imagePath) : string.Empty,
            ["source_width"] = sourceWidth,
            ["source_height"] = sourceHeight,
            ["target_width"] = targetWidth,
            ["target_height"] = targetHeight,
            ["patch_count"] = patchCount,
            ["output_count"] = outputCount,
            ["finite_count"] = finiteCount,
            ["nonzero_count"] = nonZeroCount,
            ["max_abs"] = maxAbs,
            ["selected_patch_previews"] = selected,
            ["fp32_dump"] = dumpPath,
            ["fp32_dump_sha256"] = File.Exists(dumpPath) ? ComputeQwen35Sha256(dumpPath) : string.Empty,
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["error"] = error ?? string.Empty,
            ["strict_texture_execution"] = true,
            ["patch_execution"] = "single Convolution3D atlas dispatch",
            ["layout_transform"] = "Pack4SpatialToPack4Linear",
            ["compute_buffer_fallback"] = false,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -quit -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35VisionPatchAtlasProbeBatch",
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen35_vision_patch_atlas_probe.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] vision patch atlas probe report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen35 vision patch atlas probe failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35VisionEncoderPrefixProbeBatch()
    {
        var start = Stopwatch.StartNew();
        var projectRoot = Directory.GetParent(Application.dataPath).FullName;
        var modelDir = ResolveQwen35ModelDirectory(projectRoot, "qwen3.5_0.8b");
        var imagePath = ResolveQwen35ImagePath(projectRoot);
        var stopAfter = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_VISION_ENCODER_STOP_AFTER");
        if (string.IsNullOrWhiteSpace(stopAfter)) stopAfter = "52";
        var outputPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_VISION_ENCODER_REPORT");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_vision_encoder_prefix_probe.json");
        var dumpPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_VISION_ENCODER_DUMP");
        if (string.IsNullOrWhiteSpace(dumpPath))
            dumpPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "unity_vision_encoder_prefix_probe.f32");

        var valid = false;
        string error = null;
        var outputCount = 0;
        var finiteCount = 0;
        var nonZeroCount = 0;
        var maxAbs = 0f;
        var patchTextureShape = string.Empty;
        var positionTextureShape = string.Empty;
        var ropeTextureShape = string.Empty;
        var preview = new JArray();
        Texture2D image = null;
        try
        {
            image = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
            if (!ImageConversion.LoadImage(image, File.ReadAllBytes(imagePath), false))
                throw new InvalidDataException("Unity failed to decode Qwen3.5 encoder image: " + imagePath);
            var target = Qwen35VisionPreprocessor.TargetImageSize(image.height, image.width);
            var gridWidth = target.x / 16;
            var gridHeight = target.y / 16;
            var patchCount = gridWidth * gridHeight;
            var normalized = Qwen35VisionPreprocessor.ResizeNormalize(image, target.x, target.y);
            var patches = Qwen35VisionPreprocessor.BuildDuplicatedPatches(normalized, target.x, target.y);
            Qwen35VisionPreprocessor.BuildVisionRope2D(gridHeight, gridWidth, out var ropeCos, out var ropeSin);

            using (var ops = new AexisOps())
            using (var patchRepro = new AexisGraphSession(ops))
            using (var positionRepro = new AexisGraphSession(ops))
            using (var encoderRepro = new AexisGraphSession(ops))
            {
                ConfigureQwen35StrictRepro(patchRepro);
                ConfigureQwen35StrictRepro(positionRepro);
                ConfigureQwen35StrictRepro(encoderRepro);
                patchRepro.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                positionRepro.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                encoderRepro.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
                if (ResolveBoolEnv("AIIMAGE_QWEN35_VISION_ENCODER_LAYER_LOG", false))
                {
                    encoderRepro.DebugLog = line => Debug.Log("[Qwen35][VisionEncoder] " + line);
                    encoderRepro.DebugLogAllLayerOutputs = true;
                    encoderRepro.DebugLogAllLayerHeartbeats = true;
                }

                using (var stream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_vision_embed_patch.ncnn.bin")))
                using (var reader = new NcnnBinReader(stream))
                    patchRepro.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_vision_embed_patch.ncnn.param")), reader);
                using (var stream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_vision_embed_pos.ncnn.bin")))
                using (var reader = new NcnnBinReader(stream))
                    positionRepro.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_vision_embed_pos.ncnn.param")), reader);
                using (var stream = File.OpenRead(Path.Combine(modelDir, "qwen3.5_vision_encoder.ncnn.bin")))
                using (var reader = new NcnnBinReader(stream))
                    encoderRepro.LoadModel(File.ReadAllText(Path.Combine(modelDir, "qwen3.5_vision_encoder.ncnn.param")), reader);

                RenderTexture atlas = null;
                RenderTexture patchSpatial = null;
                RenderTexture patchLinear = null;
                RenderTexture positionGrid = null;
                RenderTexture positionRaw = null;
                RenderTexture positionMerged = null;
                RenderTexture cosTexture = null;
                RenderTexture sinTexture = null;
                try
                {
                    atlas = CreateQwen35PatchAtlasTexture(patchRepro, patches, target.x, target.y, 16, 2, 3);
                    using (var result = patchRepro.InferWithMultiInputs(
                        new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = atlas },
                        null,
                        null,
                        new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                        {
                            ["in0"] = new AexisGraphSession.BufferShape(4, target.x, target.y, 2, 3)
                        }))
                    {
                        patchSpatial = result.ExtractTexture("out0");
                    }
                    patchLinear = patchRepro.RentTempArray(768 / 4, patchCount, 1, RenderTextureFormat.ARGBFloat);
                    ops.Pack4SpatialToPack4Linear(patchSpatial, patchLinear);
                    patchTextureShape = patchLinear.width + "x" + patchLinear.height + "x" + patchLinear.volumeDepth;

                    positionGrid = CreateQwen35ZeroLinearTexture(positionRepro, ops, gridWidth, gridHeight);
                    using (var result = positionRepro.InferWithMultiInputs(
                        new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = positionGrid },
                        null,
                        null,
                        new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                        {
                            ["in0"] = new AexisGraphSession.BufferShape(2, gridWidth, gridHeight, 1, 1)
                        }))
                    {
                        positionRaw = result.ExtractTexture("out0");
                    }
                    if (positionRaw.dimension != UnityEngine.Rendering.TextureDimension.Tex2D || positionRaw.width != 768 || positionRaw.height != patchCount)
                        throw new InvalidOperationException("Qwen3.5 position output has unexpected texture layout: " + positionRaw.width + "x" + positionRaw.height + "x" + positionRaw.volumeDepth);
                    positionMerged = positionRepro.RentTempMat(positionRaw.width, positionRaw.height, positionRaw.format);
                    ops.LinearMatReorderMergeRows(positionRaw, positionMerged, gridWidth, gridHeight, 2);
                    positionTextureShape = positionMerged.width + "x" + positionMerged.height + "x" + positionMerged.volumeDepth;

                    cosTexture = CreateQwen35ScalarRowsTexture(encoderRepro, ropeCos, 64, 32, patchCount);
                    sinTexture = CreateQwen35ScalarRowsTexture(encoderRepro, ropeSin, 64, 32, patchCount);
                    ropeTextureShape = cosTexture.width + "x" + cosTexture.height + "x" + cosTexture.volumeDepth;

                    var encoderInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
                    {
                        ["in0"] = patchLinear,
                        ["in1"] = positionMerged,
                        ["in2"] = cosTexture,
                        ["in3"] = sinTexture
                    };
                    var encoderShapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["in0"] = new AexisGraphSession.BufferShape(2, 768, patchCount, 1, 1),
                        ["in1"] = new AexisGraphSession.BufferShape(2, 768, patchCount, 1, 1),
                        ["in2"] = new AexisGraphSession.BufferShape(3, 32, patchCount, 1, 1),
                        ["in3"] = new AexisGraphSession.BufferShape(3, 32, patchCount, 1, 1)
                    };
                    using (var result = encoderRepro.InferWithMultiInputs(encoderInputs, null, null, encoderShapes, stopAfter))
                    {
                        var values = result.ReadTextureDataForOutput(stopAfter) ?? Array.Empty<float>();
                        outputCount = values.Length;
                        var dumpBytes = new byte[values.Length * sizeof(float)];
                        Buffer.BlockCopy(values, 0, dumpBytes, 0, dumpBytes.Length);
                        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(dumpPath)));
                        File.WriteAllBytes(dumpPath, dumpBytes);
                        for (var i = 0; i < values.Length; i++)
                        {
                            var value = values[i];
                            if (!float.IsNaN(value) && !float.IsInfinity(value)) finiteCount++;
                            if (value != 0f) nonZeroCount++;
                            maxAbs = Mathf.Max(maxAbs, Mathf.Abs(value));
                        }
                        for (var i = 0; i < Math.Min(32, values.Length); i++) preview.Add(values[i]);
                        valid = outputCount > 0 && finiteCount == outputCount && nonZeroCount > 0;
                    }
                }
                finally
                {
                    if (atlas != null) patchRepro.ReturnTempArray(atlas);
                    if (patchSpatial != null) patchRepro.ReturnTempArray(patchSpatial);
                    if (patchLinear != null) patchRepro.ReturnTempArray(patchLinear);
                    if (positionGrid != null) RenderTexture.ReleaseTemporary(positionGrid);
                    if (positionRaw != null) RenderTexture.ReleaseTemporary(positionRaw);
                    if (positionMerged != null) RenderTexture.ReleaseTemporary(positionMerged);
                    if (cosTexture != null) encoderRepro.ReturnTempArray(cosTexture);
                    if (sinTexture != null) encoderRepro.ReturnTempArray(sinTexture);
                }
            }
        }
        catch (Exception exception)
        {
            error = exception.ToString();
        }
        finally
        {
            if (image != null) UnityEngine.Object.DestroyImmediate(image);
        }

        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.vision-encoder-prefix-probe/v1",
            ["model_directory"] = modelDir,
            ["image"] = imagePath,
            ["image_sha256"] = File.Exists(imagePath) ? ComputeQwen35Sha256(imagePath) : string.Empty,
            ["stop_after"] = stopAfter,
            ["output_count"] = outputCount,
            ["finite_count"] = finiteCount,
            ["nonzero_count"] = nonZeroCount,
            ["max_abs"] = maxAbs,
            ["preview"] = preview,
            ["patch_texture_shape"] = patchTextureShape,
            ["position_texture_shape"] = positionTextureShape,
            ["rope_texture_shape"] = ropeTextureShape,
            ["fp32_dump"] = dumpPath,
            ["fp32_dump_sha256"] = File.Exists(dumpPath) ? ComputeQwen35Sha256(dumpPath) : string.Empty,
            ["valid"] = valid && string.IsNullOrEmpty(error),
            ["error"] = error ?? string.Empty,
            ["strict_texture_execution"] = true,
            ["position_reorder"] = "Pack4ReorderMergeRows(grid=64x48,merge=2)",
            ["rope_generation"] = "C# vision RoPE 2D theta=10000 section=(16,16)",
            ["compute_buffer_fallback"] = false,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -quit -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35VisionEncoderPrefixProbeBatch",
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen35_vision_encoder_prefix_probe.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)));
        File.WriteAllText(outputPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] vision encoder prefix probe report: " + outputPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen35 vision encoder prefix probe failed; see " + outputPath);
        EditorApplication.Exit(0);
    }

    public static void RunQwen35FullCheckpointAuditBatch()
    {
        var start = Stopwatch.StartNew();
        var projectRoot = Directory.GetParent(Application.dataPath).FullName;
        var modelDir = ResolveQwen35ModelDirectory(projectRoot, "qwen3.5_0.8b");
        var imagePath = ResolveQwen35ImagePath(projectRoot);
        var manifestPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_COMPARE_MANIFEST");
        if (string.IsNullOrWhiteSpace(manifestPath))
            manifestPath = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "qwen35_0_8b_compare_manifest.json");
        var outputRoot = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_AUDIT_OUTPUT_ROOT");
        if (string.IsNullOrWhiteSpace(outputRoot))
            outputRoot = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "reports", "full_checkpoint_audit");
        var reportPath = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_AUDIT_REPORT");
        if (string.IsNullOrWhiteSpace(reportPath))
            reportPath = Path.Combine(outputRoot, "unity_full_audit_report.json");

        var allNetworkNames = new[] { "embed_token", "decoder", "proj_out", "vision_embed_patch", "vision_embed_pos", "vision_encoder" };
        var configuredNetworks = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_AUDIT_NETWORKS");
        var networkNames = string.IsNullOrWhiteSpace(configuredNetworks)
            ? allNetworkNames
            : configuredNetworks.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(value => value.Trim())
                .Where(value => allNetworkNames.Contains(value, StringComparer.Ordinal))
                .Distinct(StringComparer.Ordinal)
                .ToArray();
        if (networkNames.Length == 0)
            throw new InvalidDataException("AIIMAGE_QWEN35_AUDIT_NETWORKS did not select a known network.");
        var blobSets = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        var expectedCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var manifest = JObject.Parse(File.ReadAllText(manifestPath));
        foreach (var network in networkNames)
        {
            var blobs = new HashSet<string>(StringComparer.Ordinal);
            var layers = manifest["networks"]?[network]?["layers"] as JArray;
            if (layers == null)
                throw new InvalidDataException("Compare manifest network is missing: " + network);
            foreach (var layer in layers)
            {
                var type = (string)layer["type"] ?? string.Empty;
                if (type == "Input" || type == "MemoryData")
                    continue;
                if (layer["tops"] is JArray tops)
                    foreach (var top in tops)
                        if (!string.IsNullOrWhiteSpace((string)top)) blobs.Add((string)top);
            }
            blobSets[network] = blobs;
            expectedCounts[network] = blobs.Count;
        }

        var files = new JObject();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        Action<string, string, string, float[]> writeCheckpoint = (network, layerName, blobName, values) =>
        {
            if (!blobSets.TryGetValue(network, out var requested) || !requested.Contains(blobName))
                return;
            var directory = Path.Combine(outputRoot, network);
            Directory.CreateDirectory(directory);
            var fileName = string.Equals(network, "decoder", StringComparison.Ordinal)
                ? "unity.decode1.blob_" + blobName + ".f32"
                : "unity.blob_" + blobName + ".f32";
            var path = Path.Combine(directory, fileName);
            using (var stream = File.Create(path))
            using (var writer = new BinaryWriter(stream))
                for (var i = 0; i < values.Length; i++) writer.Write(values[i]);
            files[network + "/" + blobName] = path;
            seen.Add(network + "/" + blobName);
        };

        string error = null;
        try
        {
            using (var runner = new Qwen35Runner(modelDir, 1))
            {
                if (networkNames.Any(value => value.StartsWith("vision_", StringComparison.Ordinal)))
                {
                    using (var vision = runner.CreateVisionEncoderSession())
                    {
                        vision.ConfigureDebugLayerReadback(
                            blobSets.TryGetValue("vision_embed_patch", out var patchBlobs) ? patchBlobs : null,
                            blobSets.TryGetValue("vision_embed_pos", out var positionBlobs) ? positionBlobs : null,
                            blobSets.TryGetValue("vision_encoder", out var encoderBlobs) ? encoderBlobs : null,
                            writeCheckpoint);
                        using (var encoding = vision.EncodeFile(imagePath)) { }
                    }
                }
                if (networkNames.Contains("embed_token", StringComparer.Ordinal)
                    || networkNames.Contains("decoder", StringComparer.Ordinal)
                    || networkNames.Contains("proj_out", StringComparer.Ordinal))
                {
                    using (var decoder = runner.CreateDecoderSession())
                    {
                        if (string.Equals(Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_DECODER_LAYER_LOG"), "1", StringComparison.Ordinal))
                            decoder.DebugLog = line => Debug.Log("[Qwen35][FullAudit] " + line);
                        decoder.ConfigureDebugAuxiliaryReadback(
                            blobSets.TryGetValue("embed_token", out var embedBlobs) ? embedBlobs : null,
                            blobSets.TryGetValue("proj_out", out var projectionBlobs) ? projectionBlobs : null,
                            writeCheckpoint);
                        Qwen35OwnedTexture auxiliaryHidden = null;
                        Qwen35DecoderStep decoderStep = null;
                        Qwen35DecoderState warmState = null;
                        try
                        {
                            if (networkNames.Contains("embed_token", StringComparer.Ordinal)
                                || networkNames.Contains("proj_out", StringComparer.Ordinal))
                            {
                                auxiliaryHidden = decoder.EmbedTokens(new[] { 0 });
                                if (networkNames.Contains("proj_out", StringComparer.Ordinal))
                                    decoder.ProjectLogits(auxiliaryHidden);
                                auxiliaryHidden.Dispose();
                                auxiliaryHidden = null;
                                decoder.ConfigureDebugAuxiliaryReadback(null, null, null);
                            }
                            if (networkNames.Contains("decoder", StringComparer.Ordinal))
                            {
                                var prefix = new[] { 248045, 846, 198, 9419, 248046, 198, 248045, 74455 };
                                using (var prefixHidden = decoder.EmbedTokens(prefix))
                                using (var initialState = decoder.CreateInitialState())
                                using (var warmupStep = decoder.Decode(prefixHidden, 0, initialState))
                                    warmState = warmupStep.DetachState();

                                decoder.ConfigureDebugLayerReadback(
                                    blobSets.TryGetValue("decoder", out var decoderBlobs) ? decoderBlobs : null,
                                    (layerIndex, layerName, blobName, values) => writeCheckpoint("decoder", layerName, blobName, values));
                                using (var inputHidden = decoder.EmbedTokens(new[] { 198 }))
                                    decoderStep = decoder.Decode(inputHidden, 8, warmState);
                            }
                        }
                        finally
                        {
                            decoderStep?.Dispose();
                            warmState?.Dispose();
                            auxiliaryHidden?.Dispose();
                        }
                    }
                }
            }
        }
        catch (Exception exception)
        {
            error = exception.ToString();
        }

        var missing = new JArray();
        foreach (var network in networkNames)
            foreach (var blob in blobSets[network])
                if (!seen.Contains(network + "/" + blob)) missing.Add(network + "/" + blob);
        start.Stop();
        var report = new JObject
        {
            ["schema"] = "qwen35.unity.full-checkpoint-audit/v1",
            ["model_directory"] = modelDir,
            ["image"] = imagePath,
            ["manifest"] = manifestPath,
            ["output_root"] = outputRoot,
            ["expected_numeric_checkpoint_count"] = expectedCounts.Values.Sum(),
            ["dumped_numeric_checkpoint_count"] = seen.Count,
            ["missing_numeric_checkpoints"] = missing,
            ["dump_files"] = files,
            ["valid"] = string.IsNullOrEmpty(error) && missing.Count == 0,
            ["strict_texture_execution"] = true,
            ["compute_buffer_fallback"] = false,
            ["error"] = error ?? string.Empty,
            ["elapsed_ms"] = start.ElapsedMilliseconds,
            ["command"] = "C:\\Program Files\\Unity 6000.2.7f2\\Editor\\Unity.exe -batchmode -quit -projectPath E:\\Projects\\AIImage -executeMethod NcnnDebugRunner.RunQwen35FullCheckpointAuditBatch",
            ["unity_log"] = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_UNITY_LOG") ?? "Logs/qwen35_full_checkpoint_audit.log"
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(reportPath)));
        File.WriteAllText(reportPath, report.ToString(Aexis.Samples.Json.Formatting.Indented));
        Debug.Log("[Qwen35] full checkpoint audit report: " + reportPath + " valid=" + report["valid"]);
        if (!(bool)report["valid"])
            throw new InvalidOperationException("Qwen3.5 full checkpoint audit dump failed; see " + reportPath);
        EditorApplication.Exit(0);
    }

    private static void ConfigureQwen35StrictRepro(AexisGraphSession repro)
    {
        repro.ExecutionMode = AexisInferenceExecutionMode.ProductionTextureOnly;
        // QWEN probe sessions explicitly allocate ARGBFloat activations below.
        // Keep the strict-plan target dtype aligned with that real GPU storage so
        // Input descriptors are admitted without any conversion or fallback.
        repro.StrictTextureTargetDtype = "FP32";
        repro.DisallowInferenceTempComputeBuffers = true;
        repro.DisallowBufferToTextureMaterialization = true;
        repro.DisallowBufferOutputs = true;
        repro.EnableAttentionMatMulPack4Specializations = true;
        repro.EnableConv1x1TextureConvolution = true;
        repro.EnableDepthWiseTextureConvolution = true;
    }

    private static RenderTexture CreateQwen35ZeroTexture(AexisGraphSession repro, AexisOps ops, int width, int height, int depth)
    {
        var texture = repro.RentTempArray(width, height, depth, RenderTextureFormat.ARGBFloat);
        ops.FillScalarTexture(new[] { 0f }, texture);
        return texture;
    }

    private static RenderTexture CreateQwen35ScalarRowsTexture(AexisGraphSession repro, float[] values, int sourceRowWidth, int outputWidth, int rows)
    {
        if (values == null || values.Length != sourceRowWidth * rows)
            throw new ArgumentException("Qwen3.5 scalar-row upload value count mismatch", nameof(values));
        if (outputWidth <= 0 || outputWidth > sourceRowWidth || rows <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputWidth));
        var texture = repro.RentTempArray(outputWidth, rows, 1, RenderTextureFormat.ARGBFloat);
        var upload = new Texture2DArray(outputWidth, rows, 1, TextureFormat.RGBAFloat, false, true)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp,
            anisoLevel = 0,
            name = "Qwen35ScalarRowsUpload"
        };
        try
        {
            var pixels = new Color[outputWidth * rows];
            for (var row = 0; row < rows; row++)
            for (var x = 0; x < outputWidth; x++)
                pixels[row * outputWidth + x] = new Color(values[row * sourceRowWidth + x], 0f, 0f, 0f);
            upload.SetPixels(pixels, 0, 0);
            upload.Apply(false, true);
            Graphics.CopyTexture(upload, 0, 0, texture, 0, 0);
            return texture;
        }
        catch
        {
            repro.ReturnTempArray(texture);
            throw;
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(upload);
        }
    }

    private static RenderTexture CreateQwen35ZeroLinearTexture(AexisGraphSession repro, AexisOps ops, int width, int height)
    {
        var texture = repro.RentTempMat(width, height, RenderTextureFormat.ARGBFloat);
        ops.FillScalarTexture(new[] { 0f }, texture);
        return texture;
    }

    private static RenderTexture CreateQwen35CosSinTexture(AexisGraphSession repro, AexisOps ops, int width, float value)
    {
        var texture = repro.RentTempArray(width, 1, 1, RenderTextureFormat.ARGBFloat);
        var values = new float[width];
        if (!Mathf.Approximately(value, 0f))
            for (var i = 0; i < values.Length; i++) values[i] = value;
        using (var buffer = new ComputeBuffer(values.Length, sizeof(float), ComputeBufferType.Structured))
        {
            buffer.SetData(values);
            ops.FillPack4FromBufferCHW(buffer, width, 1, 1, texture);
        }
        return texture;
    }

    private static RenderTexture CreateQwen35Pack4TextureFromCdhw(
        AexisGraphSession repro,
        float[] values,
        int width,
        int height,
        int depth,
        int channels)
    {
        var expected = width * height * depth * channels;
        if (values == null || values.Length != expected)
            throw new ArgumentException("Qwen3.5 CDHW upload shape mismatch", nameof(values));
        var packs = Mathf.Max(1, (channels + 3) / 4);
        var slices = depth * packs;
        var texture = repro.RentTempArray(width, height, slices, RenderTextureFormat.ARGBFloat);
        var upload = new Texture2DArray(width, height, slices, TextureFormat.RGBAFloat, false, true)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp,
            anisoLevel = 0,
            name = "Qwen35FixedInputCdhwUpload"
        };
        try
        {
            var pixelCount = width * height;
            for (var z = 0; z < depth; z++)
            {
                for (var pack = 0; pack < packs; pack++)
                {
                    var pixels = new Color[pixelCount];
                    for (var y = 0; y < height; y++)
                    {
                        for (var x = 0; x < width; x++)
                        {
                            var spatial = y * width + x;
                            var channel0 = pack * 4;
                            var r = channel0 < channels ? values[((channel0 * depth + z) * height + y) * width + x] : 0f;
                            var g = channel0 + 1 < channels ? values[(((channel0 + 1) * depth + z) * height + y) * width + x] : 0f;
                            var b = channel0 + 2 < channels ? values[(((channel0 + 2) * depth + z) * height + y) * width + x] : 0f;
                            var a = channel0 + 3 < channels ? values[(((channel0 + 3) * depth + z) * height + y) * width + x] : 0f;
                            pixels[spatial] = new Color(r, g, b, a);
                        }
                    }
                    var slice = z * packs + pack;
                    upload.SetPixels(pixels, slice, 0);
                }
            }
            upload.Apply(false, true);
            for (var slice = 0; slice < slices; slice++)
                Graphics.CopyTexture(upload, slice, 0, texture, slice, 0);
            return texture;
        }
        catch
        {
            repro.ReturnTempArray(texture);
            throw;
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(upload);
        }
    }

    private static RenderTexture CreateQwen35PatchAtlasTexture(
        AexisGraphSession repro,
        float[] patches,
        int atlasWidth,
        int atlasHeight,
        int patchSize,
        int depth,
        int channels)
    {
        var patchGridWidth = atlasWidth / patchSize;
        var patchGridHeight = atlasHeight / patchSize;
        var patchCount = patchGridWidth * patchGridHeight;
        var patchElements = patchSize * patchSize * depth * channels;
        if (atlasWidth % patchSize != 0 || atlasHeight % patchSize != 0)
            throw new ArgumentException("Qwen3.5 patch atlas must be patch aligned.");
        if (patches == null || patches.Length != patchCount * patchElements)
            throw new ArgumentException("Qwen3.5 patch atlas value count mismatch.", nameof(patches));

        var packs = Mathf.Max(1, (channels + 3) / 4);
        var slices = depth * packs;
        var texture = repro.RentTempArray(atlasWidth, atlasHeight, slices, RenderTextureFormat.ARGBFloat);
        var upload = new Texture2DArray(atlasWidth, atlasHeight, slices, TextureFormat.RGBAFloat, false, true)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp,
            anisoLevel = 0,
            name = "Qwen35PatchAtlasUpload"
        };
        try
        {
            for (var z = 0; z < depth; z++)
            {
                for (var pack = 0; pack < packs; pack++)
                {
                    var pixels = new Color[atlasWidth * atlasHeight];
                    for (var y = 0; y < atlasHeight; y++)
                    {
                        var patchY = y / patchSize;
                        var localY = y - patchY * patchSize;
                        for (var x = 0; x < atlasWidth; x++)
                        {
                            var patchX = x / patchSize;
                            var localX = x - patchX * patchSize;
                            var patchIndex = patchY * patchGridWidth + patchX;
                            var channel = pack * 4;
                            var patchBase = patchIndex * patchElements;
                            float Read(int c)
                            {
                                if (c >= channels) return 0f;
                                return patches[patchBase + (((c * depth + z) * patchSize + localY) * patchSize + localX)];
                            }
                            pixels[y * atlasWidth + x] = new Color(Read(channel), Read(channel + 1), Read(channel + 2), Read(channel + 3));
                        }
                    }
                    upload.SetPixels(pixels, z * packs + pack, 0);
                }
            }
            upload.Apply(false, true);
            for (var slice = 0; slice < slices; slice++)
                Graphics.CopyTexture(upload, slice, 0, texture, slice, 0);
            return texture;
        }
        catch
        {
            repro.ReturnTempArray(texture);
            throw;
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(upload);
        }
    }

    private static string ComputeQwen35Sha256(string path)
    {
        using (var stream = File.OpenRead(path))
        using (var sha = System.Security.Cryptography.SHA256.Create())
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty).ToLowerInvariant();
    }

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

        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        Aexis.Execution.AexisGpuResourceTracker.Reset("clip_dir_batch");

        var go = new GameObject("ClipDirectoryDebugRunner");
        try
        {
            var runner = go.AddComponent<ClipNcnnReproRunner>();
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
                    var gpuSummary = Aexis.Execution.AexisGpuResourceTracker.BuildSummary();
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

            Aexis.Execution.AexisGpuResourceTracker.WriteReport(outputDir);
            Debug.Log("[CLIP-DIR] summary=" + summaryPath);
        }
        finally
        {
            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
            UnityEngine.Object.DestroyImmediate(go);
        }
    }

    private static void ConfigureClipRunnerFromEnv(ClipNcnnReproRunner runner, bool defaultEnableDebugDump)
    {
        if (runner == null)
            return;

        var pack4OnlyGuard = ResolveBoolEnv(ClipPack4OnlyGuardEnvVar, false);
        var configuredPrecision = ResolveStringEnv(ClipPrecisionEnvVar, "Auto");
        if (!Enum.TryParse(configuredPrecision, true, out Aexis.Execution.AexisPrecisionMode precisionMode))
            throw new InvalidOperationException("Invalid " + ClipPrecisionEnvVar + ": " + configuredPrecision);
        runner.precisionMode = precisionMode;
        runner.enableDebugDump = ResolveBoolEnv(ClipEnableDumpEnvVar, defaultEnableDebugDump);
        runner.forceFullRenderTexturePath = ResolveBoolEnv(ClipForceFullRtEnvVar, runner.forceFullRenderTexturePath)
            || pack4OnlyGuard;
        runner.useCommandBuffer = ResolveBoolEnv(ClipUseCommandBufferEnvVar, runner.useCommandBuffer);
        runner.useAsyncComputeCommandBuffer = ResolveBoolEnv(ClipUseAsyncComputeEnvVar, runner.useAsyncComputeCommandBuffer);
        runner.verifyCommandBufferParity = ResolveBoolEnv(ClipVerifyCommandBufferParityEnvVar, false);
        runner.commandBufferParityTolerance = Mathf.Max(0f, ResolveFloatEnvOrDefault(
            ClipCommandBufferParityToleranceEnvVar,
            runner.commandBufferParityTolerance));
        runner.commandBufferParityProbeBlob = ResolveStringEnv(ClipCommandBufferParityProbeBlobEnvVar, string.Empty);
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
            var configuredPrecision = ResolveStringEnv(CodeFormerPrecisionEnvVar, "Auto");
            if (!Enum.TryParse(configuredPrecision, true, out Aexis.Execution.AexisPrecisionMode precisionMode))
                throw new InvalidOperationException("Invalid " + CodeFormerPrecisionEnvVar + ": " + configuredPrecision);
            runner.precisionMode = precisionMode;
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

    private static async UniTask RunCodeFormerEncoderPack4RegressionInternal()
    {
        var go = new GameObject("CodeFormerEncoderPack4Regression");
        var trackingStarted = false;
        try
        {
            var runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
            // The original rejection occurred with FP16 Pack4 activations.
            runner.precisionMode = AexisPrecisionMode.FP16;

            var runnerType = typeof(CodeFormerNcnnReproRunner2);
            var ensureLoaded = runnerType.GetMethod("EnsureLoaded", BindingFlags.Instance | BindingFlags.NonPublic);
            if (ensureLoaded == null || !(ensureLoaded.Invoke(runner, null) is UniTask loadTask))
                throw new InvalidOperationException("CodeFormer encoder load entry point is unavailable.");
            await loadTask;

            var encoderField = runnerType.GetField("_encoderRepro", BindingFlags.Instance | BindingFlags.NonPublic);
            var encoder = encoderField?.GetValue(runner) as AexisGraphSession;
            if (encoder == null || encoder.Model == null)
                throw new InvalidOperationException("CodeFormer encoder session was not loaded.");
            if (!encoder.DisallowBufferAccess
                || !encoder.DisallowBufferOutputs
                || !encoder.DisallowBufferToTextureMaterialization
                || !encoder.DisallowInferenceTempComputeBuffers)
            {
                throw new InvalidOperationException("CodeFormer encoder regression must run with the production Pack4 no-buffer guard.");
            }

            using var ops = new AexisOps();
            using var commandBuffer = new UnityEngine.Rendering.CommandBuffer
            {
                name = "CodeFormerEncoderPack4Regression"
            };

            encoder.BeginInferenceTempResourceTracking();
            trackingStarted = true;
            var input = encoder.RentTempArray(commandBuffer, 512, 512, 1, RenderTextureFormat.ARGBHalf);
            ComputeTexture output = null;
            try
            {
                // A fixed texture value is sufficient for this dispatch regression:
                // it covers the actual encoder graph through MatMul_911 while
                // keeping all activations in texture-backed CommandBuffer storage.
                ops.FillScalarTexture(commandBuffer, new[] { 0f, 0f, 0f, 0f }, input);
                var inputShape = new AexisGraphSession.BufferShape(3, 512, 512, 1, 3);
                output = encoder.ForwardPack4(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["input"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal) { ["input"] = inputShape },
                    out var outputShape,
                    new[] { "1428" },
                    "1428",
                    "1428");

                var plan = encoder.LastTextureExecutionPlan;
                var matMul = plan?.nodes?.FirstOrDefault(node => string.Equals(node.layer, "MatMul_911", StringComparison.Ordinal));
                if (output == null
                    || outputShape.dims != 2
                    || outputShape.w != 1024
                    || outputShape.h != 256
                    || outputShape.d != 1
                    || outputShape.c != 1
                    || plan == null
                    || !plan.strictEligible
                    || !plan.dispatchAllowed
                    || matMul == null
                    || !matMul.accepted
                    || !string.Equals(matMul.executionPath, "command-buffer-pack4:inner-product-linear-mat", StringComparison.Ordinal))
                {
                    throw new InvalidOperationException(
                        "CodeFormer MatMul_911 did not admit the verified CommandBuffer LinearMat texture path. "
                        + (plan?.summary ?? "plan unavailable"));
                }

                var stats = encoder.GetInferenceTempResourceStats();
                if (stats.tempBufferRentCount != 0 || stats.tempBufferPeakLiveCount != 0)
                {
                    throw new InvalidOperationException(
                        "CodeFormer encoder allocated a temporary ComputeBuffer during Pack4 inference. "
                        + "rents=" + stats.tempBufferRentCount + " peak=" + stats.tempBufferPeakLiveCount);
                }

                encoder.ReturnTempArray(commandBuffer, output);
                output = null;
                encoder.ReturnTempArray(commandBuffer, input);
                input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);
                Debug.Log("[NcnnDebugRunner] CodeFormer encoder MatMul_911 CommandBuffer LinearMat regression passed");
            }
            finally
            {
                // On the successful path releases are recorded into the command
                // buffer above. The fallback cleanup is only for planning errors
                // before a retained output can be scheduled for release.
                if (output != null)
                    encoder.ReturnTempArray(commandBuffer, output);
                if (input != null)
                    encoder.ReturnTempArray(commandBuffer, input);
            }
        }
        finally
        {
            if (trackingStarted)
            {
                var runner = go.GetComponent<CodeFormerNcnnReproRunner2>();
                var encoderField = typeof(CodeFormerNcnnReproRunner2).GetField("_encoderRepro", BindingFlags.Instance | BindingFlags.NonPublic);
                if (encoderField?.GetValue(runner) is AexisGraphSession encoder)
                    encoder.EndInferenceTempResourceTracking();
            }
            UnityEngine.Object.DestroyImmediate(go);
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
            var configuredPrecision = ResolveStringEnv(GfpganPrecisionEnvVar, "Auto");
            if (!Enum.TryParse(configuredPrecision, true, out Aexis.Execution.AexisPrecisionMode precisionMode))
                throw new InvalidOperationException("Invalid " + GfpganPrecisionEnvVar + ": " + configuredPrecision);
            runner.precisionMode = precisionMode;
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

        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        Aexis.Execution.AexisGpuResourceTracker.Reset("CodeFormerStress");

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
                        + " | " + Aexis.Execution.AexisGpuResourceTracker.BuildSummary());

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
                Aexis.Execution.AexisGpuResourceTracker.WriteReport(dumpDir, "stress_gpu_resources.txt");
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

            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
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
        var dumpDir = CreateGenericDumpDir("AIImage_ReproSuiteStress");
        var summaryPath = Path.Combine(dumpDir, "suite_summary.txt");
        var lines = new List<string>(256)
        {
            "input=" + inputPath,
            "iterations=" + iterations.ToString(CultureInfo.InvariantCulture),
            "started_at=" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
            string.Empty
        };
        var failures = new List<string>();

        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        try
        {
            await RunRealEsrganStressAsync(tex, iterations, dumpDir, lines, failures);
            await RunYoloSegStressAsync(tex, iterations, dumpDir, lines, failures);
            await RunMattingStressAsync(tex, iterations, dumpDir, lines, failures);
            await RunGfpganStressAsync(tex, iterations, dumpDir, lines, failures);
            await RunCodeFormerStressAsync(tex, iterations, dumpDir, lines, failures);
        }
        finally
        {
            lines.Add(string.Empty);
            lines.Add("finished_at=" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
            if (failures.Count > 0)
                lines.Add("failures=" + string.Join(" | ", failures));

            try { File.WriteAllLines(summaryPath, lines); } catch { }

            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
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

        Aexis.Execution.AexisGpuResourceTracker.Enabled = true;
        Aexis.Execution.AexisGpuResourceTracker.Reset("D1.RealESRGAN");
        using var sw = new StreamWriter(summaryPath, false);
        sw.WriteLine("model\timage\tmode\tpath_kind\tstatus\telapsed_ms\twidth\theight\tmean_abs_rgb\tmax_abs_rgb\terror\toutput");

        try
        {
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
                                Aexis.Execution.AexisGpuResourceTracker.Reset("D1.RealESRGAN.Pack4Rt");
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
                                WriteD1RuntimeBenchmark(
                                    "esrgan-pack4_rt",
                                    immediate.elapsedMs,
                                    immediate.error,
                                    dumpDir,
                                    "backend_mean_abs_rgb",
                                    "0",
                                    model + " | " + Path.GetFileName(inputPath));
                                sw.Flush();
                                AppendRealEsrganValidationFailure(failures, model, Path.GetFileName(inputPath), "immediate", immediate, maxElapsedMs);
                            }

                            if (runCommandBuffer)
                            {
                                Aexis.Execution.AexisGpuResourceTracker.Reset("D1.RealESRGAN.CommandBuffer");
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
                                WriteD1RuntimeBenchmark(
                                    "esrgan-command_buffer",
                                    commandBuffer.elapsedMs,
                                    commandBuffer.error,
                                    dumpDir,
                                    "backend_mean_abs_rgb",
                                    meanAbs.ToString("0.######", CultureInfo.InvariantCulture),
                                    model + " | " + Path.GetFileName(inputPath)
                                        + " | backend_max_abs_rgb=" + maxAbs.ToString(CultureInfo.InvariantCulture));
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
        finally
        {
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(dumpDir, "gpu_resource_stats.txt"); } catch { }
            Aexis.Execution.AexisGpuResourceTracker.Enabled = false;
        }
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
                modelName = models.Count > 0 ? models[0] : "realesr-animevideov3-x4",
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

    private static async UniTask RunRealEsrganStressAsync(Texture2D tex, int iterations, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "ESRGAN";
        Aexis.Execution.AexisGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        RealEsrganNcnnReproRunner runner = null;
        try
        {
            runner = go.AddComponent<RealEsrganNcnnReproRunner>();
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
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(dumpDir, "esrgan_gpu_resources.txt"); } catch { }
        }
    }

    private static async UniTask RunYoloSegStressAsync(Texture2D tex, int iterations, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "YoloSeg";
        Aexis.Execution.AexisGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        YoloSegNcnnReproRunner runner = null;
        try
        {
            runner = go.AddComponent<YoloSegNcnnReproRunner>();
            runner.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
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
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(dumpDir, "yoloseg_gpu_resources.txt"); } catch { }
        }
    }

    private static async UniTask RunMattingStressAsync(Texture2D tex, int iterations, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "Matting";
        Aexis.Execution.AexisGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        MatterNcnnReproRunner runner = null;
        try
        {
            runner = go.AddComponent<MatterNcnnReproRunner>();
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
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(dumpDir, "matting_gpu_resources.txt"); } catch { }
        }
    }

    private static async UniTask RunGfpganStressAsync(Texture2D tex, int iterations, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "GFPGAN";
        Aexis.Execution.AexisGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        GfpganNcnnReproRunner runner = null;
        try
        {
            runner = go.AddComponent<GfpganNcnnReproRunner>();
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
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(dumpDir, "gfpgan_gpu_resources.txt"); } catch { }
        }
    }

    private static async UniTask RunCodeFormerStressAsync(Texture2D tex, int iterations, string dumpDir, List<string> lines, List<string> failures)
    {
        const string runnerName = "CodeFormer";
        Aexis.Execution.AexisGpuResourceTracker.Reset(runnerName);
        var go = new GameObject("ReproSuiteStress_" + runnerName);
        CodeFormerNcnnReproRunner2 runner = null;
        try
        {
            runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
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
            try { Aexis.Execution.AexisGpuResourceTracker.WriteReport(dumpDir, "codeformer_gpu_resources.txt"); } catch { }
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
            + " | " + Aexis.Execution.AexisGpuResourceTracker.BuildSummary();
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
            models.Add("realesr-animevideov3-x4");

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

    private static void ComputeMaskedChangeByHorizontalHalf(
        Texture2D source,
        Texture2D candidate,
        Texture2D mask,
        out int leftMaskedPixels,
        out int rightMaskedPixels,
        out int leftChangedPixels,
        out int rightChangedPixels)
    {
        leftMaskedPixels = 0;
        rightMaskedPixels = 0;
        leftChangedPixels = 0;
        rightChangedPixels = 0;
        if (source == null || candidate == null || mask == null
            || source.width != candidate.width || source.height != candidate.height
            || source.width != mask.width || source.height != mask.height)
            return;

        var sourcePixels = source.GetPixels32();
        var candidatePixels = candidate.GetPixels32();
        var maskPixels = mask.GetPixels32();
        var midpoint = source.width / 2;
        for (var i = 0; i < sourcePixels.Length; i++)
        {
            var masked = maskPixels[i].r >= 128 || maskPixels[i].g >= 128 || maskPixels[i].b >= 128;
            if (!masked)
                continue;

            var changed = sourcePixels[i].r != candidatePixels[i].r
                          || sourcePixels[i].g != candidatePixels[i].g
                          || sourcePixels[i].b != candidatePixels[i].b;
            if (i % source.width < midpoint)
            {
                leftMaskedPixels++;
                if (changed)
                    leftChangedPixels++;
            }
            else
            {
                rightMaskedPixels++;
                if (changed)
                    rightChangedPixels++;
            }
        }
    }

    private static Texture2D BuildPureWhiteMask(Texture2D maskedExample)
    {
        if (maskedExample == null)
            return null;
        var source = maskedExample.GetPixels32();
        var output = new Color32[source.Length];
        for (var i = 0; i < source.Length; i++)
        {
            var white = source[i].r == 255 && source[i].g == 255 && source[i].b == 255;
            output[i] = white ? new Color32(255, 255, 255, 255) : new Color32(0, 0, 0, 255);
        }
        var mask = new Texture2D(maskedExample.width, maskedExample.height, TextureFormat.RGBA32, false);
        mask.SetPixels32(output);
        mask.Apply(false, false);
        return mask;
    }

    private static double ComputeFullRgbMeanAbsDiff(Texture2D reference, Texture2D candidate, out int maxAbs)
    {
        maxAbs = 0;
        if (reference == null || candidate == null
            || reference.width != candidate.width || reference.height != candidate.height)
            return double.PositiveInfinity;
        var a = reference.GetPixels32();
        var b = candidate.GetPixels32();
        double sum = 0d;
        for (var i = 0; i < a.Length; i++)
        {
            var dr = Mathf.Abs(a[i].r - b[i].r);
            var dg = Mathf.Abs(a[i].g - b[i].g);
            var db = Mathf.Abs(a[i].b - b[i].b);
            sum += dr + dg + db;
            maxAbs = Mathf.Max(maxAbs, Mathf.Max(dr, Mathf.Max(dg, db)));
        }
        return a.Length > 0 ? sum / (a.Length * 3d) : 0d;
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

    private static float ComputeMatteMeanAlpha(Texture2D matte)
    {
        if (matte == null)
            return 0f;
        var pixels = matte.GetPixels32();
        if (pixels == null || pixels.Length == 0)
            return 0f;
        double sum = 0d;
        for (var i = 0; i < pixels.Length; i++)
            sum += pixels[i].r / 255d;
        return (float)(sum / pixels.Length);
    }

    private static void WriteD1RuntimeBenchmark(
        string runner,
        long elapsedMs,
        string error,
        string debugDumpPath,
        string taskMetricName,
        string taskMetricValue,
        string taskMetricDetail)
    {
        var manifestPath = Environment.GetEnvironmentVariable(Aexis.Execution.AexisModelManifestLoader.ManifestEnvironmentVariable);
        Aexis.ModelManifest manifest = null;
        try
        {
            if (!string.IsNullOrWhiteSpace(manifestPath))
                manifest = Aexis.Execution.AexisModelManifestLoader.LoadFromFile(manifestPath);
            else if (string.Equals(runner, "clip", StringComparison.Ordinal))
                manifest = Aexis.Execution.AexisModelManifestLoader.ResolveRunnerManifest(
                    "mobileclip_s0_export",
                    ResolveRunnerPrecision(ClipPrecisionEnvVar));
            else if (string.Equals(runner, "matting", StringComparison.Ordinal))
                manifest = Aexis.Execution.AexisModelManifestLoader.ResolveRunnerManifest(
                    "matting.ncnn",
                    ResolveRunnerPrecision(MattingPrecisionEnvVar));
            else
            {
                manifestPath = ResolveD1DefaultManifestPath(runner);
                if (!string.IsNullOrWhiteSpace(manifestPath))
                    manifest = Aexis.Execution.AexisModelManifestLoader.LoadFromFile(manifestPath);
            }
        }
        catch (Exception e)
        {
            error = string.IsNullOrWhiteSpace(error) ? "Benchmark manifest read failed: " + e.Message : error;
        }

        var stats = Aexis.Execution.AexisGpuResourceTracker.GetStatsSnapshot();
        var report = new D1RuntimeBenchmarkReport
        {
            runner = runner ?? string.Empty,
            modelId = manifest?.modelId ?? string.Empty,
            manifestPath = manifestPath ?? string.Empty,
            activationDtype = manifest?.precision?.activationDataType.ToString() ?? string.Empty,
            weightDtype = manifest?.precision?.weightDataType.ToString() ?? string.Empty,
            quantizationVersion = manifest?.quantization?.quantizationVersion ?? string.Empty,
            calibrationVersion = manifest?.quantization?.calibrationVersion ?? string.Empty,
            quantizedOperators = manifest?.quantization?.quantizedOperators ?? Array.Empty<string>(),
            strictTexturePlan = manifest?.precision?.requireStrictTexturePlan ?? false,
            status = string.IsNullOrWhiteSpace(error) ? "passed" : "failed",
            error = error ?? string.Empty,
            elapsedMs = elapsedMs,
            peakTemporaryTextureBytes = stats.peakTemporaryTextureBytes,
            peakTextureBytes = stats.peakTextureBytes,
            peakBufferBytes = stats.peakBufferBytes,
            peakTotalBytes = stats.peakTotalBytes,
            taskMetricName = taskMetricName ?? string.Empty,
            taskMetricValue = taskMetricValue ?? string.Empty,
            taskMetricDetail = taskMetricDetail ?? string.Empty,
            debugDumpPath = debugDumpPath ?? string.Empty,
            graphicsDevice = SystemInfo.graphicsDeviceType + ": " + SystemInfo.graphicsDeviceName
        };

        var outputDir = Path.Combine(Application.dataPath, "..", "Logs", "D1PrecisionBenchmarks");
        Directory.CreateDirectory(outputDir);
        var dtype = string.IsNullOrWhiteSpace(report.activationDtype) ? "unknown" : report.activationDtype;
        var fileName = (runner ?? "runner") + "-" + dtype + "-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff", CultureInfo.InvariantCulture) + ".json";
        var path = Path.Combine(outputDir, fileName);
        File.WriteAllText(path, JsonUtility.ToJson(report, true));
        Debug.Log("D1 runtime benchmark | path=" + path + " | status=" + report.status + " | peak_temp_rt_bytes=" + report.peakTemporaryTextureBytes + " | peak_total_bytes=" + report.peakTotalBytes);
    }

    private static string ResolveD1DefaultManifestPath(string runner)
    {
        string fileName = null;
        if (string.Equals(runner, "clip", StringComparison.Ordinal))
            fileName = "clip-mobileclip-s0.fp16.model.json";
        else if (string.Equals(runner, "esrgan-pack4_rt", StringComparison.Ordinal)
            || string.Equals(runner, "esrgan-command_buffer", StringComparison.Ordinal))
            fileName = "esrgan-realesrgan-x4plus.fp16.model.json";

        if (string.IsNullOrWhiteSpace(fileName))
            return null;

        return Aexis.Samples.AexisSampleStreamingAssets.TryResolveFilePath(Path.Combine("InferenceManifests", fileName), out var path) ? path : null;
    }

    private static Aexis.Execution.AexisPrecisionMode ResolveRunnerPrecision(string environmentVariable)
    {
        var raw = ResolveStringEnv(environmentVariable, "Auto");
        if (!Enum.TryParse(raw, true, out Aexis.Execution.AexisPrecisionMode precisionMode))
            throw new InvalidOperationException("Invalid " + environmentVariable + ": " + raw);
        return precisionMode;
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

    private static float ComputeMatteForegroundIou(Texture2D a, Texture2D b)
    {
        if (a == null || b == null || a.width != b.width || a.height != b.height)
            return 0f;
        var pa = a.GetPixels32();
        var pb = b.GetPixels32();
        var intersection = 0;
        var union = 0;
        for (var i = 0; i < pa.Length; i++)
        {
            var aForeground = pa[i].r >= 128;
            var bForeground = pb[i].r >= 128;
            if (aForeground && bForeground)
                intersection++;
            if (aForeground || bForeground)
                union++;
        }
        return union == 0 ? 1f : (float)intersection / union;
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
