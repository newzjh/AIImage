using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.Rendering;

public struct SDInpaintingNcnnReproResult
{
    public Texture2D texture;
    public string error;
    public long elapsedMs;
    public int seed;
    public string dumpDir;
}

public sealed class SDInpaintingNcnnReproRunner : MonoBehaviour
{
    private const string TextEncoderTokenBlobName = "token";
    private const string TextEncoderMultiplierBlobName = "multiplier";
    private const string TextEncoderCondBlobName = "cond";
    private const string TextEncoderOutputBlobName = "cal_12";
    private const string VaeEncoderInputBlobName = "in0";
    private const string VaeEncoderMeanBlobName = "out0";
    private const string VaeEncoderStdBlobName = "out1";
    private const string VaeDecoderInputBlobName = "input.1";
    private const string VaeDecoderOutputBlobName = "815";
    private const string UnetInputBlobName = "in0";
    private const string UnetTimestepBlobName = "in1";
    private const string UnetTextBlobName = "in2";
    private const string UnetOutputBlobName = "out0";
    private const int ModelImageSize = 512;
    private const int LatentSize = 64;
    private const int TokenCount = 77;
    private const int PromptChunkTokenCount = 75;
    private const int TextEmbeddingWidth = 768;
    private const int LatentChannels = 4;
    private const int UnetInputChannels = 9;
    private const string UnetConvInLayerName = "conv_160";
    private const string UnetConvInBlobName = "156";
    private const float LatentScale = 0.18215f;
    private const float InvLatentScale = 1f / LatentScale;
    private const int StartTokenId = 49406;
    private const int EndTokenId = 49407;
    private const int TrainTimestepCount = 1000;
    private const float BetaStart = 0.00085f;
    private const float BetaEnd = 0.012f;
    private const int StepsOffset = 1;
    private const bool SetAlphaToOne = false;
    private const string StageTraceEnvVar = "AIIMAGE_SD_STAGE_TRACE";
    private const string StageSyncGpuEnvVar = "AIIMAGE_SD_STAGE_SYNC_GPU";
    private const string StageUnloadUnusedAssetsEnvVar = "AIIMAGE_SD_STAGE_UNLOAD_UNUSED_ASSETS";
    private const string StageReleaseWinogradEnvVar = "AIIMAGE_SD_STAGE_RELEASE_WINOGRAD";
    private const string StageGcCollectEnvVar = "AIIMAGE_SD_STAGE_GC_COLLECT";
    private const string ResourceSnapshotEnvVar = "AIIMAGE_SD_RESOURCE_SNAPSHOT";
    private const string EditorLowVramEnvVar = "AIIMAGE_SD_EDITOR_LOW_VRAM";
    private const string AllowEditorFloatTensorEnvVar = "AIIMAGE_SD_ALLOW_EDITOR_FLOAT_TENSOR";
    private const string StableDiffusionModelRootEnvVar = "AIIMAGE_SD_MODEL_ROOT";
    private const string InpaintModelRootEnvVar = "AIIMAGE_SD_INPAINT_MODEL_ROOT";
    private static readonly string[] DebugUnetBlobNames = { "156", "out0" };
    private static readonly string[] OfficialUnetCacheBlobNames =
    {
        "44",
        "83",
        "116",
        "163",
        "251",
        "337",
        "425",
        "511",
        "599",
        "627",
        "711",
        "725",
        "741",
        "755",
        "772",
        "858",
        "944",
        "1032",
        "1118",
        "1204",
        "1292",
        "1378",
        "1439"
    };

    [Serializable]
    private readonly struct ResolvedPaths
    {
        public readonly string textParamPath;
        public readonly string textBinPath;
        public readonly string unetParamPath;
        public readonly string unetBinPath;
        public readonly string vaeParamPath;
        public readonly string vaeBinPath;
        public readonly string vaeEncoderParamPath;
        public readonly string vaeEncoderBinPath;
        public readonly string tokenizerVocabPath;

        public ResolvedPaths(
            string textParamPath,
            string textBinPath,
            string unetParamPath,
            string unetBinPath,
            string vaeParamPath,
            string vaeBinPath,
            string vaeEncoderParamPath,
            string vaeEncoderBinPath,
            string tokenizerVocabPath)
        {
            this.textParamPath = textParamPath;
            this.textBinPath = textBinPath;
            this.unetParamPath = unetParamPath;
            this.unetBinPath = unetBinPath;
            this.vaeParamPath = vaeParamPath;
            this.vaeBinPath = vaeBinPath;
            this.vaeEncoderParamPath = vaeEncoderParamPath;
            this.vaeEncoderBinPath = vaeEncoderBinPath;
            this.tokenizerVocabPath = tokenizerVocabPath;
        }
    }

    public string stableDiffusionRootRelativePath = "StableDiffusion";
    public string inpaintingModelRootRelativePath = "sdinpainting";
    public bool useReferenceAssetFallback = true;
    public bool enableTempPool = false;
    public int maxPooledPerShape = 0;
    public bool keepRawConvWeightsForTexturePath = false;
    public RenderTextureFormat tensorTextureFormat = RenderTextureFormat.ARGBHalf;
    public RenderTextureFormat encoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
    public RenderTextureFormat decoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
    public bool useNcnnStyleGroupNorm = true;
    public bool useOfficialUnetCache = false;
    public bool enableMhaParallelSoftmax = true;
    public bool enableMhaQkvFusion = true;
    public bool enableAttentionMatMulPack4Specializations = true;
    public bool enableDebugDump = false;
    public bool enableLayerRuntimeProfile = false;
    public bool syncLayerRuntimeProfileGpu = false;
    public bool useCommandBuffer = false;
    public bool useAsyncComputeCommandBuffer = false;
    public bool disallowInferenceTempComputeBuffers = false;
    public int layerRuntimeProfileTopN = 30;
    public bool blackMaskMeansInpaint = false;
    public bool useOfficialNoise = true;
    public bool enableEditorLowVramGuard = true;
    public const int PeopleRemovalRecommendedStepCount = 12;
    public const float PeopleRemovalRecommendedStrength = 1f;
    public const float PeopleRemovalRecommendedGuidanceScale = 10f;
    public const string PeopleRemovalRecommendedPositivePrompt =
        "best quality, realistic photo, empty indoor background, clean wall, shelf, table, furniture, coherent texture, seamless fill, background only, no people, no person, no human";
    public const string PeopleRemovalRecommendedNegativePrompt =
        "person, people, human, man, woman, child, face, portrait, head, body, skin, hands, arms, legs, crowd, group photo, selfie, mannequin, statue, reflection, silhouette, duplicate, blurry, deformed, extra limbs, artifacts, text, watermark";

    [Range(1, 50)] public int defaultStepCount = PeopleRemovalRecommendedStepCount;
    [Range(0f, 1f)] public float defaultStrength = PeopleRemovalRecommendedStrength;
    [Range(1f, 20f)] public float defaultGuidanceScale = PeopleRemovalRecommendedGuidanceScale;
    [TextArea(2, 5)] public string defaultPositivePrompt = PeopleRemovalRecommendedPositivePrompt;
    [TextArea(2, 5)] public string defaultNegativePrompt = PeopleRemovalRecommendedNegativePrompt;

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnRepro _textRepro;
    private NcnnRepro _unetRepro;
    private NcnnRepro _vaeRepro;
    private NcnnRepro _vaeEncoderRepro;
    private StableDiffusionSimpleTokenizer _tokenizer;
    private ResolvedPaths? _resolvedPaths;
    private string _loadedModelKey;
    private string _lastDumpDir;
    private float[] _alphasCumprod;
    private float _finalAlphaCumprod;
    private bool _unetDebugDumped;
    private bool _editorLowVramGuardLogged;
    private static readonly uint[] OpenCvNormalKn = new uint[128];
    private static readonly float[] OpenCvNormalWn = new float[128];
    private static readonly float[] OpenCvNormalFn = new float[128];
    private static bool _openCvNormalTablesReady;
    private readonly Dictionary<string, List<string>> _layerDebugLines = new Dictionary<string, List<string>>(StringComparer.Ordinal);
    private readonly List<string> _resourceSnapshotLines = new List<string>(256);

    private interface INormalRng
    {
        float NextNormal();
    }

    private sealed class TorchCpuNormalRng : INormalRng
    {
        private const int StateCount = 624;
        private const int StateM = 397;
        private const uint MatrixA = 0x9908b0dfu;
        private const uint UMask = 0x80000000u;
        private const uint LMask = 0x7fffffffu;
        private readonly uint[] _state = new uint[StateCount];
        private int _left;
        private int _next;
        private float? _nextNormal;

        public TorchCpuNormalRng(uint seed)
        {
            _state[0] = seed;
            for (var i = 1; i < StateCount; i++)
                _state[i] = 1812433253u * (_state[i - 1] ^ (_state[i - 1] >> 30)) + (uint)i;
            _left = 1;
            _next = 0;
            _nextNormal = null;
        }

        public float NextNormal()
        {
            if (_nextNormal.HasValue)
            {
                var sample = _nextNormal.Value;
                _nextNormal = null;
                return sample;
            }

            var u1 = NextUniform();
            var u2 = NextUniform();
            var radius = (float)Math.Sqrt(-2.0 * Math.Log(1.0 - u2));
            var theta = 2f * Mathf.PI * u1;
            _nextNormal = radius * Mathf.Sin(theta);
            return radius * Mathf.Cos(theta);
        }

        public float NextUniform()
        {
            return NextUniformInternal();
        }

        private float NextUniformInternal()
        {
            // Match PyTorch's uniform_real<float> which uses the low 24 mantissa bits.
            return (NextUInt32() & 0x00ffffffu) * (1f / 16777216f);
        }

        private uint NextUInt32()
        {
            if (--_left == 0)
                NextState();

            var y = _state[_next++];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            return y;
        }

        private void NextState()
        {
            var p = 0;
            _left = StateCount;
            _next = 0;

            for (var j = StateCount - StateM; j > 0; j--, p++)
                _state[p] = _state[p + StateM] ^ Twist(_state[p], _state[p + 1]);

            for (var j = StateM - 1; j > 0; j--, p++)
                _state[p] = _state[p + StateM - StateCount] ^ Twist(_state[p], _state[p + 1]);

            _state[p] = _state[p + StateM - StateCount] ^ Twist(_state[p], _state[0]);
        }

        private static uint Twist(uint u, uint v)
        {
            return (((u & UMask) | (v & LMask)) >> 1) ^ ((v & 1u) != 0u ? MatrixA : 0u);
        }
    }

    private sealed class FallbackNormalRng : INormalRng
    {
        private readonly System.Random _rng;
        private bool _hasNext;
        private float _next;

        public FallbackNormalRng(int seed)
        {
            _rng = new System.Random(seed);
        }

        public float NextNormal()
        {
            if (_hasNext)
            {
                _hasNext = false;
                return _next;
            }

            var u1 = 1.0 - _rng.NextDouble();
            var u2 = 1.0 - _rng.NextDouble();
            var radius = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-12)));
            var theta = 2.0 * Math.PI * u2;
            _next = (float)(radius * Math.Sin(theta));
            _hasNext = true;
            return (float)(radius * Math.Cos(theta));
        }
    }

    private sealed class UnetCacheBlob
    {
        public string name;
        public RenderTexture texture;
        public NcnnRepro.BufferShape shape;
    }

    public string LastDumpDir => _lastDumpDir;

    private void Awake()
    {
        ApplyEditorLowVramGuardIfNeeded();
        EnsureRuntimeObjects();
    }

    private void OnDestroy()
    {
        Release();
    }

    public UniTask<SDInpaintingNcnnReproResult> ProcessAsync(Texture sourceImage, Texture maskImage, CancellationToken ct)
    {
        return ProcessAsync(
            sourceImage,
            maskImage,
            defaultPositivePrompt,
            defaultNegativePrompt,
            Mathf.Max(1, defaultStepCount),
            0,
            Mathf.Clamp01(defaultStrength),
            defaultGuidanceScale,
            ct);
    }

    public void ApplyPeopleRemovalPreset()
    {
        defaultStepCount = Mathf.Max(1, PeopleRemovalRecommendedStepCount);
        defaultStrength = Mathf.Clamp01(PeopleRemovalRecommendedStrength);
        defaultGuidanceScale = Mathf.Max(1f, PeopleRemovalRecommendedGuidanceScale);
        defaultPositivePrompt = PeopleRemovalRecommendedPositivePrompt;
        defaultNegativePrompt = PeopleRemovalRecommendedNegativePrompt;
        blackMaskMeansInpaint = false;
    }

    public async UniTask<SDInpaintingNcnnReproResult> ProcessAsync(
        Texture sourceImage,
        Texture maskImage,
        string positivePrompt,
        string negativePrompt,
        int stepCount,
        int seed,
        float strength,
        float guidanceScale,
        CancellationToken ct)
    {
        ApplyEditorLowVramGuardIfNeeded();
        var totalSw = Stopwatch.StartNew();
        _lastDumpDir = null;
        _unetDebugDumped = false;
        _resourceSnapshotLines.Clear();
        var actualSeed = ResolveSeed(seed);
        var sourceOutputWidth = 0;
        var sourceOutputHeight = 0;
        var destroySourceReadable = false;
        var destroyMaskReadable = false;

        SDInpaintingNcnnReproResult Finish(SDInpaintingNcnnReproResult result)
        {
            result.elapsedMs = totalSw.ElapsedMilliseconds;
            result.seed = actualSeed;
            result.dumpDir = _lastDumpDir;
            return result;
        }

        Texture2D sourceReadable = null;
        Texture2D maskReadable = null;
        Texture2D resizedSource = null;
        Texture2D resizedMask = null;
        Texture2D normalizedMask = null;
        Texture2D maskedSource = null;
        Texture2D latentMaskTexture = null;
        Texture2D generated512 = null;
        Texture2D generatedResized = null;
        RenderTexture latentMaskTex = null;
        RenderTexture maskedLatentTex = null;
        RenderTexture cleanLatentTex = null;
        RenderTexture initNoiseTex = null;
        RenderTexture latentsTex = null;
        RenderTexture decodeLatentsTex = null;
        RenderTexture condTex = null;
        RenderTexture uncondTex = null;
        try
        {
            if (sourceImage == null || maskImage == null)
                return Finish(new SDInpaintingNcnnReproResult { error = "Source image or mask image is null." });

            stepCount = Mathf.Max(1, stepCount);
            strength = Mathf.Clamp01(strength);
            guidanceScale = Mathf.Max(1f, guidanceScale);

            if (enableDebugDump)
                _lastDumpDir = CreateDumpDir("AIImage_SDInpaint_NcnnRepro");

            EnsureRuntimeObjects();
            EnsureSchedulerTables();
            var paths = ResolvePaths();
            EnsureModelIdentity(paths);
            LogResourceSnapshot("process_begin");

            ReportProgress(0.02f, "Load models");
            await EnsureTextEncoderLoadedAsync(paths, ct);
            LogResourceSnapshot("after_load_text");
            ct.ThrowIfCancellationRequested();

            ReportProgress(0.12f, "Prepare images");
            sourceReadable = EnsureReadableTexture(sourceImage);
            maskReadable = EnsureReadableTexture(maskImage);
            if (sourceReadable == null || maskReadable == null)
                return Finish(new SDInpaintingNcnnReproResult { error = "Failed to read source or mask texture." });
            sourceOutputWidth = sourceReadable.width;
            sourceOutputHeight = sourceReadable.height;
            destroySourceReadable = !ReferenceEquals(sourceReadable, sourceImage);
            destroyMaskReadable = !ReferenceEquals(maskReadable, maskImage);

            resizedSource = ReadResizedTexture(sourceReadable, ModelImageSize, ModelImageSize);
            resizedMask = ReadResizedTexture(maskReadable, ModelImageSize, ModelImageSize);
            if (resizedSource == null || resizedMask == null)
                return Finish(new SDInpaintingNcnnReproResult { error = "Failed to resize inpainting inputs." });

            if (destroySourceReadable && sourceReadable != null)
            {
                DestroyImmediate(sourceReadable);
                sourceReadable = null;
                destroySourceReadable = false;
            }

            if (destroyMaskReadable && maskReadable != null)
            {
                DestroyImmediate(maskReadable);
                maskReadable = null;
                destroyMaskReadable = false;
            }

            normalizedMask = NormalizeInpaintMask(resizedMask, blackMaskMeansInpaint);
            maskedSource = BuildMaskedTexture(resizedSource, normalizedMask);
            latentMaskTexture = ReadResizedTextureNearest(normalizedMask, LatentSize, LatentSize);
            if (normalizedMask == null || maskedSource == null || latentMaskTexture == null)
                return Finish(new SDInpaintingNcnnReproResult { error = "Failed to prepare masked image inputs." });

            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                TryWriteTexturePng(resizedSource, _lastDumpDir, "01_source_512.png");
                TryWriteTexturePng(resizedMask, _lastDumpDir, "02_mask_input_512.png");
                TryWriteTexturePng(normalizedMask, _lastDumpDir, "02_mask_512.png");
                TryWriteTexturePng(maskedSource, _lastDumpDir, "03_masked_source_512.png");
                TryWriteTexturePng(latentMaskTexture, _lastDumpDir, "04_mask_64.png");
                WriteAllTextSafe(Path.Combine(_lastDumpDir, "positive_prompt.txt"), positivePrompt ?? string.Empty);
                WriteAllTextSafe(Path.Combine(_lastDumpDir, "negative_prompt.txt"), negativePrompt ?? string.Empty);
            }
            LogResourceSnapshot("after_prepare_images");

            ReportProgress(0.20f, "Encode prompts");
            LogStageTrace("prompt encode begin | positive");
            var cond = await EncodePromptAsync(positivePrompt ?? string.Empty, "positive", ct);
            LogStageTrace("prompt encode end | positive | values=" + (cond?.Length ?? 0).ToString(CultureInfo.InvariantCulture));
            LogStageTrace("prompt encode begin | negative");
            var uncond = await EncodePromptAsync(negativePrompt ?? string.Empty, "negative", ct);
            LogStageTrace("prompt encode end | negative | values=" + (uncond?.Length ?? 0).ToString(CultureInfo.InvariantCulture));
            if (cond == null || cond.Length != TokenCount * TextEmbeddingWidth
                || uncond == null || uncond.Length != TokenCount * TextEmbeddingWidth)
            {
                return Finish(new SDInpaintingNcnnReproResult { error = "Prompt conditioning failed." });
            }

            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                WriteFloatArrayStatsSafe(Path.Combine(_lastDumpDir, "prompt_cond_stats.txt"), cond);
                WriteFloatArrayStatsSafe(Path.Combine(_lastDumpDir, "prompt_uncond_stats.txt"), uncond);
                WriteFloatArrayRawF32Safe(Path.Combine(_lastDumpDir, "prompt_cond_f32.bin"), cond);
                WriteFloatArrayRawF32Safe(Path.Combine(_lastDumpDir, "prompt_uncond_f32.bin"), uncond);
            }

            LogStageTrace("dispose text model begin");
            DisposeTextModel();
            LogStageTrace("dispose text model end");
            await ForceStageCleanupAsync(ct, "after_text");
            LogResourceSnapshot("after_text_cleanup");

            ReportProgress(0.28f, "Encode images");
            await EnsureVaeEncoderLoadedAsync(paths, ct);
            LogResourceSnapshot("after_load_vae_encoder");
            var latentRng = CreateLatentNoiseRng(actualSeed);
            var useStrengthMax = strength >= 0.9999f;
            if (!useStrengthMax)
                cleanLatentTex = await EncodeImageLatentsPack4Async(resizedSource, latentRng, ct);
            initNoiseTex = CreateLatentNoisePack4(latentRng);
            maskedLatentTex = await EncodeImageLatentsPack4Async(maskedSource, latentRng, ct);
            if ((!useStrengthMax && cleanLatentTex == null) || initNoiseTex == null || maskedLatentTex == null)
                return Finish(new SDInpaintingNcnnReproResult { error = "VAE encoder failed." });
            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                if (cleanLatentTex != null)
                    DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "latent_clean_stats.txt"), cleanLatentTex, LatentChannels);
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "latent_masked_stats.txt"), maskedLatentTex, LatentChannels);
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "latent_noise_stats.txt"), initNoiseTex, LatentChannels);
            }

            DestroyRuntimeTexture(ref resizedSource);
            DestroyRuntimeTexture(ref resizedMask);
            DestroyRuntimeTexture(ref normalizedMask);
            DestroyRuntimeTexture(ref maskedSource);
            DisposeVaeEncoderModel();
            await ForceStageCleanupAsync(ct, "after_vae_encoder");
            LogResourceSnapshot("after_vae_encoder_cleanup");

            var timesteps = BuildTimesteps(stepCount);
            var activeTimesteps = SelectTimestepsByStrength(timesteps, strength);
            var debugMaxDenoiseSteps = ResolvePositiveIntEnvOrDefault("AIIMAGE_SD_DEBUG_MAX_DENOISE_STEPS", 0);
            if (debugMaxDenoiseSteps > 0 && activeTimesteps.Length > debugMaxDenoiseSteps)
            {
                var truncatedTimesteps = new int[debugMaxDenoiseSteps];
                Array.Copy(activeTimesteps, truncatedTimesteps, debugMaxDenoiseSteps);
                activeTimesteps = truncatedTimesteps;
            }
            if (activeTimesteps == null || activeTimesteps.Length < 1)
                return Finish(new SDInpaintingNcnnReproResult { error = "No valid timesteps for the requested strength." });

            await EnsureUnetLoadedAsync(paths, ct);
            LogResourceSnapshot("after_load_unet");
            latentMaskTex = CreateMaskPack4Texture(latentMaskTexture);
            DestroyRuntimeTexture(ref latentMaskTexture);
            condTex = CreateTensorPack4Texture(_unetRepro, cond, 2, TextEmbeddingWidth, TokenCount, 1, 1, tensorTextureFormat);
            uncondTex = CreateTensorPack4Texture(_unetRepro, uncond, 2, TextEmbeddingWidth, TokenCount, 1, 1, tensorTextureFormat);
            latentsTex = useStrengthMax ? initNoiseTex : BuildNoisyLatentsPack4(cleanLatentTex, initNoiseTex, activeTimesteps[0]);
            initNoiseTex = useStrengthMax ? null : initNoiseTex;
            if (cleanLatentTex != null)
            {
                _unetRepro?.ReturnTempArray(cleanLatentTex);
                cleanLatentTex = null;
            }
            if (initNoiseTex != null)
            {
                _unetRepro?.ReturnTempArray(initNoiseTex);
                initNoiseTex = null;
            }
            LogResourceSnapshot("after_init_latents");
            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "latent_mask_tex_stats.txt"), latentMaskTex, 1);
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "latent_masked_tex_stats.txt"), maskedLatentTex, LatentChannels);
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "latent_init_tex_stats.txt"), latentsTex, LatentChannels);
                DumpTextureRawF32Safe(Path.Combine(_lastDumpDir, "latent_init_f32.bin"), latentsTex, LatentChannels);
            }

            if (latentMaskTex == null || maskedLatentTex == null || latentsTex == null || condTex == null || uncondTex == null)
                return Finish(new SDInpaintingNcnnReproResult { error = "Failed to initialize latent textures." });

            ReportProgress(0.36f, "Sample latents");
            for (var i = 0; i < activeTimesteps.Length; i++)
            {
                ct.ThrowIfCancellationRequested();
                var timestep = activeTimesteps[i];
                var prevTimestep = i + 1 < activeTimesteps.Length ? activeTimesteps[i + 1] : -1;
                var stepProgress = (float)i / Mathf.Max(1, activeTimesteps.Length);
                ReportProgress(0.36f + 0.48f * stepProgress, "Denoise step " + (i + 1).ToString(CultureInfo.InvariantCulture) + "/" + activeTimesteps.Length.ToString(CultureInfo.InvariantCulture));
                LogResourceSnapshot("denoise_step_" + (i + 1).ToString(CultureInfo.InvariantCulture) + "_begin");

                var epsilonTex = await RunCfgUnetPack4Async(latentsTex, latentMaskTex, maskedLatentTex, timestep, condTex, uncondTex, guidanceScale, ct);
                if (epsilonTex == null)
                    return Finish(new SDInpaintingNcnnReproResult { error = "UNet inference failed at timestep " + timestep.ToString(CultureInfo.InvariantCulture) + "." });
                LogResourceSnapshot("denoise_step_" + (i + 1).ToString(CultureInfo.InvariantCulture) + "_after_unet");

                try
                {
                    if (!string.IsNullOrWhiteSpace(_lastDumpDir))
                    {
                        DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "epsilon_step_" + i.ToString(CultureInfo.InvariantCulture) + "_stats.txt"), epsilonTex, LatentChannels);
                        DumpTextureRawF32Safe(Path.Combine(_lastDumpDir, "epsilon_step_" + i.ToString(CultureInfo.InvariantCulture) + "_f32.bin"), epsilonTex, LatentChannels);
                    }

                    var nextLatents = DdimStepPack4(latentsTex, epsilonTex, timestep, prevTimestep);
                    if (nextLatents == null)
                        return Finish(new SDInpaintingNcnnReproResult { error = "DDIM step failed at timestep " + timestep.ToString(CultureInfo.InvariantCulture) + "." });

                    if (!string.IsNullOrWhiteSpace(_lastDumpDir))
                    {
                        DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "latent_step_" + i.ToString(CultureInfo.InvariantCulture) + "_stats.txt"), nextLatents, LatentChannels);
                        DumpTextureRawF32Safe(Path.Combine(_lastDumpDir, "latent_step_" + i.ToString(CultureInfo.InvariantCulture) + "_f32.bin"), nextLatents, LatentChannels);
                    }

                    _unetRepro.ReturnTempArray(latentsTex);
                    latentsTex = nextLatents;
                    LogResourceSnapshot("denoise_step_" + (i + 1).ToString(CultureInfo.InvariantCulture) + "_after_ddim");
                }
                finally
                {
                    _unetRepro.ReturnTempArray(epsilonTex);
                }

                await UniTask.Yield();
            }

            if (condTex != null)
            {
                _unetRepro?.ReturnTempArray(condTex);
                condTex = null;
            }
            if (uncondTex != null)
            {
                _unetRepro?.ReturnTempArray(uncondTex);
                uncondTex = null;
            }
            if (maskedLatentTex != null)
            {
                _unetRepro?.ReturnTempArray(maskedLatentTex);
                maskedLatentTex = null;
            }
            if (latentMaskTex != null)
            {
                _unetRepro?.ReturnTempArray(latentMaskTex);
                latentMaskTex = null;
            }

            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "latent_final_stats.txt"), latentsTex, LatentChannels);
            LogResourceSnapshot("after_final_latents");

            ReportProgress(0.88f, "Decode image");
            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "latent_decode_input_stats.txt"), latentsTex, LatentChannels);
            decodeLatentsTex = latentsTex;
            latentsTex = null;
            DisposeUnetModel();
            await ForceStageCleanupAsync(ct, "after_unet");
            LogResourceSnapshot("after_unet_cleanup");
            await EnsureVaeDecoderLoadedAsync(paths, ct);
            LogResourceSnapshot("after_load_vae_decoder");
            generated512 = await DecodeLatentsTextureAsync(decodeLatentsTex, ct);
            if (decodeLatentsTex != null)
            {
                _vaeRepro?.ReturnTempArray(decodeLatentsTex);
                decodeLatentsTex = null;
            }
            if (generated512 == null)
                return Finish(new SDInpaintingNcnnReproResult { error = "VAE decoder failed." });
            LogResourceSnapshot("after_decode");

            generatedResized = ReadResizedTexture(generated512, sourceOutputWidth, sourceOutputHeight);
            if (generatedResized == null)
                return Finish(new SDInpaintingNcnnReproResult { error = "Failed to scale generated image back to source size." });

            var finalTexture = generatedResized;
            generatedResized = null;

            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                TryWriteTexturePng(generated512, _lastDumpDir, "05_generated_512.png");
                TryWriteTexturePng(finalTexture, _lastDumpDir, "06_generated_fullres.png");
                TryWriteTexturePng(finalTexture, _lastDumpDir, "07_final_output.png");
                WriteAllTextSafe(
                    Path.Combine(_lastDumpDir, "run_config.txt"),
                    "seed=" + actualSeed.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "steps=" + stepCount.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "strength=" + strength.ToString("0.000000", CultureInfo.InvariantCulture) + Environment.NewLine
                    + "guidance_scale=" + guidanceScale.ToString("0.000000", CultureInfo.InvariantCulture) + Environment.NewLine
                    + "black_mask_means_inpaint=" + BoolText(blackMaskMeansInpaint) + Environment.NewLine
                    + "active_timesteps=" + activeTimesteps.Length.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "debug_max_denoise_steps=" + debugMaxDenoiseSteps.ToString(CultureInfo.InvariantCulture));
                WriteAllTextSafe(Path.Combine(_lastDumpDir, "resource_snapshots.txt"), string.Join(Environment.NewLine, _resourceSnapshotLines));
            }

            ReportProgress(1f, string.Empty);
            return Finish(new SDInpaintingNcnnReproResult { texture = finalTexture });
        }
        catch (OperationCanceledException)
        {
            return Finish(new SDInpaintingNcnnReproResult { error = "Cancelled" });
        }
        catch (Exception e)
        {
            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
                WriteAllTextSafe(Path.Combine(_lastDumpDir, "exception.txt"), e.ToString());
            UnityEngine.Debug.LogError("[SDInpaint] ProcessAsync failed: " + e);
            return Finish(new SDInpaintingNcnnReproResult { error = e.ToString() });
        }
        finally
        {
            LogResourceSnapshot("process_finally_begin");
            FlushDebugLogToFile("unet", "unet_layer_debug.txt");
            ReleaseTempArrayFallback(ref latentsTex, _unetRepro, "SDInpaint.FinalLatentsTex");
            ReleaseTempArrayFallback(ref decodeLatentsTex, _vaeRepro, "SDInpaint.DecodeLatentsTex");
            ReleaseTempArrayFallback(ref maskedLatentTex, _unetRepro, "SDInpaint.MaskedLatentsTex");
            ReleaseTempArrayFallback(ref latentMaskTex, _unetRepro, "SDInpaint.LatentMaskTex");
            ReleaseTempArrayFallback(ref cleanLatentTex, _unetRepro, "SDInpaint.CleanLatentsTex");
            ReleaseTempArrayFallback(ref initNoiseTex, _unetRepro, "SDInpaint.InitNoiseTex");
            ReleaseTempArrayFallback(ref condTex, _unetRepro, "SDInpaint.PromptCondTex");
            ReleaseTempArrayFallback(ref uncondTex, _unetRepro, "SDInpaint.PromptUncondTex");
            DisposeTextModel();
            DisposeUnetModel();
            DisposeVaeDecoderModel();
            DisposeVaeEncoderModel();
            if (destroySourceReadable && sourceReadable != null)
                DestroyImmediate(sourceReadable);
            if (destroyMaskReadable && maskReadable != null)
                DestroyImmediate(maskReadable);
            if (resizedSource != null)
                DestroyImmediate(resizedSource);
            if (resizedMask != null)
                DestroyImmediate(resizedMask);
            if (normalizedMask != null)
                DestroyImmediate(normalizedMask);
            if (maskedSource != null)
                DestroyImmediate(maskedSource);
            if (latentMaskTexture != null)
                DestroyImmediate(latentMaskTexture);
            if (generated512 != null)
                DestroyImmediate(generated512);
            if (generatedResized != null)
                DestroyImmediate(generatedResized);
            LogResourceSnapshot("process_finally_end");
            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
                WriteAllTextSafe(Path.Combine(_lastDumpDir, "resource_snapshots.txt"), string.Join(Environment.NewLine, _resourceSnapshotLines));
        }
    }

    private static void DestroyRuntimeTexture(ref Texture2D texture)
    {
        if (texture == null)
            return;

        DestroyImmediate(texture);
        texture = null;
    }

    private static void ReleaseTempArrayFallback(ref RenderTexture texture, NcnnRepro owner, string label)
    {
        if (texture == null)
            return;

        if (owner != null)
        {
            owner.ReturnTempArray(texture);
            texture = null;
            return;
        }

        NcnnGpuResourceTracker.ReleaseTexture(texture, label ?? "SDInpaint.TempArrayFallback");
        try { RenderTexture.ReleaseTemporary(texture); } catch { }
        texture = null;
    }

    private void LogResourceSnapshot(string stage)
    {
        if (!ResolveBoolEnv(ResourceSnapshotEnvVar, true))
            return;

        try
        {
            var process = Process.GetCurrentProcess();
            var privateMb = process.PrivateMemorySize64 / (1024.0 * 1024.0);
            var workingSetMb = process.WorkingSet64 / (1024.0 * 1024.0);
            var managedMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
            var gfxMb = GetGraphicsDriverMemoryMb();
            var rtCount = Resources.FindObjectsOfTypeAll<RenderTexture>().Length;
            var liveRtHandles = RenderTexture.active != null ? 1 : 0;
            var line =
                "[SDInpaint][Resources] stage=" + (stage ?? "")
                + " | private_mb=" + privateMb.ToString("F3", CultureInfo.InvariantCulture)
                + " | working_set_mb=" + workingSetMb.ToString("F3", CultureInfo.InvariantCulture)
                + " | managed_mb=" + managedMb.ToString("F3", CultureInfo.InvariantCulture)
                + " | gfx_mb=" + gfxMb.ToString("F3", CultureInfo.InvariantCulture)
                + " | rt_objects=" + rtCount.ToString(CultureInfo.InvariantCulture)
                + " | active_rt=" + liveRtHandles.ToString(CultureInfo.InvariantCulture)
                + " | " + NcnnGpuResourceTracker.BuildSummary();
            _resourceSnapshotLines.Add(line);
            UnityEngine.Debug.Log(line);
        }
        catch (Exception e)
        {
            try
            {
                UnityEngine.Debug.Log("[SDInpaint][Resources] stage=" + (stage ?? "") + " | snapshot_failed=" + e.Message);
            }
            catch
            {
            }
        }
    }

    private static float GetGraphicsDriverMemoryMb()
    {
        try
        {
            return UnityEngine.Profiling.Profiler.GetAllocatedMemoryForGraphicsDriver() / (1024f * 1024f);
        }
        catch
        {
            return 0f;
        }
    }

    private void EnsureRuntimeObjects()
    {
        ApplyEditorLowVramGuardIfNeeded();
        _ops ??= new NcnnOps();
        _textRepro ??= new NcnnRepro(_ops);
        _unetRepro ??= new NcnnRepro(_ops);
        _vaeRepro ??= new NcnnRepro(_ops);
        _vaeEncoderRepro ??= new NcnnRepro(_ops);
    }

    private void ApplyEditorLowVramGuardIfNeeded()
    {
        if (Application.isBatchMode)
            return;

        var guardEnabled = ResolveBoolEnv(EditorLowVramEnvVar, enableEditorLowVramGuard);
        if (!guardEnabled)
            return;

        var allowFloatTensor = ResolveBoolEnv(AllowEditorFloatTensorEnvVar, false);
        var changed = false;

        if (!allowFloatTensor && tensorTextureFormat != RenderTextureFormat.ARGBHalf)
        {
            tensorTextureFormat = RenderTextureFormat.ARGBHalf;
            changed = true;
        }

        if (encoderTensorTextureFormat != RenderTextureFormat.ARGBHalf)
        {
            encoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
            changed = true;
        }

        if (decoderTensorTextureFormat != RenderTextureFormat.ARGBHalf)
        {
            decoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
            changed = true;
        }

        if (enableTempPool)
        {
            enableTempPool = false;
            changed = true;
        }

        if (maxPooledPerShape != 0)
        {
            maxPooledPerShape = 0;
            changed = true;
        }

        if (keepRawConvWeightsForTexturePath)
        {
            keepRawConvWeightsForTexturePath = false;
            changed = true;
        }

        if (!_editorLowVramGuardLogged)
        {
            _editorLowVramGuardLogged = true;
            UnityEngine.Debug.Log(
                "[SDInpaint] editor low-VRAM guard active"
                + " | tensor=" + tensorTextureFormat
                + " | encoder=" + encoderTensorTextureFormat
                + " | decoder=" + decoderTensorTextureFormat
                + " | tempPool=" + BoolText(enableTempPool)
                + " | maxPooledPerShape=" + maxPooledPerShape.ToString(CultureInfo.InvariantCulture)
                + " | keepRawConvWeights=" + BoolText(keepRawConvWeightsForTexturePath)
                + " | allowFloatTensor=" + BoolText(allowFloatTensor)
                + " | changed=" + BoolText(changed));
        }
    }

    private void ApplyCommonOptions(NcnnRepro repro)
    {
        if (repro == null)
            return;

        repro.EnableTempPool = enableTempPool;
        repro.MaxPooledPerShape = maxPooledPerShape;
        repro.KeepRawConvWeightsForTexturePath = keepRawConvWeightsForTexturePath;
        repro.ForceBufferConvolutionAll = false;
        repro.ForceBufferBinaryOpAll = false;
        repro.ForceBufferGeluAll = false;
        repro.EnableGpuGeluBufferPath = true;
        repro.EnableConv1x1TextureConvolution = true;
        repro.EnableDepthWiseTextureConvolution = true;
        repro.EnableGeneralTextureConvolution = true;
        repro.EnableGroupNormTexturePath = true;
        repro.EnableMhaParallelSoftmax = ResolveBoolEnv("AIIMAGE_SD_MHA_PARALLEL_SOFTMAX", enableMhaParallelSoftmax);
        repro.EnableMhaQkvFusion = ResolveBoolEnv("AIIMAGE_SD_MHA_QKV_FUSION", enableMhaQkvFusion);
        repro.UseNcnnStyleGroupNorm = useNcnnStyleGroupNorm;
        repro.LayerRuntimeProfileEnabled = enableLayerRuntimeProfile;
        repro.LayerRuntimeProfileSyncGpu = syncLayerRuntimeProfileGpu;
        repro.TensorTextureFormat = tensorTextureFormat;
        repro.DisallowInferenceTempComputeBuffers = disallowInferenceTempComputeBuffers;
        repro.DebugBreakOnFirstNonFiniteLayerOutput = enableDebugDump;
        repro.DebugLogAllLayerOutputs = false;
        repro.DebugLogAllLayerHeartbeats = false;
        repro.DebugLogAllBufferMaterialize = false;
        repro.DebugCompareTextureConvLayers = ResolveDebugNameSet("AIIMAGE_SD_COMPARE_TEXTURE_CONV_LAYERS", "AIIMAGE_SD_COMPARE_TEXTURE_CONV");
        repro.DebugCompareTextureLayers = ResolveDebugNameSet("AIIMAGE_SD_COMPARE_TEXTURE_LAYERS", "AIIMAGE_SD_COMPARE_TEXTURE");
        repro.ForceBufferLayerTypes = ResolveDebugNameSet("AIIMAGE_SD_FORCE_BUFFER_LAYER_TYPES");
        repro.ForceBufferLayerNames = ResolveDebugNameSet("AIIMAGE_SD_FORCE_BUFFER_LAYER_NAMES");
    }

    private bool ResolveSpatialAttentionPack4()
    {
        return ResolveBoolEnv("AIIMAGE_SD_ATTENTION_PACK4", enableAttentionMatMulPack4Specializations);
    }

    private void ApplySpatialModelOptions(NcnnRepro repro)
    {
        if (repro == null)
            return;

        repro.EnableAttentionMatMulPack4Specializations = ResolveSpatialAttentionPack4();
        repro.EnableVistaTailPack4Specializations = false;
    }

    private static bool ResolveBoolEnv(string name, bool defaultValue)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(env))
                return defaultValue;

            env = env.Trim();
            return string.Equals(env, "1", StringComparison.OrdinalIgnoreCase)
                   || string.Equals(env, "true", StringComparison.OrdinalIgnoreCase)
                   || string.Equals(env, "yes", StringComparison.OrdinalIgnoreCase)
                   || string.Equals(env, "on", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return defaultValue;
        }
    }

    private static int ResolvePositiveIntEnvOrDefault(string name, int defaultValue)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(env))
                return defaultValue;

            if (!int.TryParse(env.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value))
                return defaultValue;

            return value > 0 ? value : defaultValue;
        }
        catch
        {
            return defaultValue;
        }
    }

    private static HashSet<string> ResolveDebugNameSet(string primaryEnv, string fallbackEnv = null)
    {
        try
        {
            var raw = Environment.GetEnvironmentVariable(primaryEnv);
            if (string.IsNullOrWhiteSpace(raw) && !string.IsNullOrWhiteSpace(fallbackEnv))
                raw = Environment.GetEnvironmentVariable(fallbackEnv);
            if (string.IsNullOrWhiteSpace(raw))
                return null;

            var set = new HashSet<string>(StringComparer.Ordinal);
            var parts = raw.Split(new[] { ',', ';', ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            for (var i = 0; i < parts.Length; i++)
            {
                var name = parts[i].Trim();
                if (!string.IsNullOrEmpty(name))
                    set.Add(name);
            }

            return set.Count > 0 ? set : null;
        }
        catch
        {
            return null;
        }
    }

    private void EnsureModelIdentity(ResolvedPaths paths)
    {
        var modelKey =
            paths.textParamPath + "|" +
            paths.textBinPath + "|" +
            paths.unetParamPath + "|" +
            paths.unetBinPath + "|" +
            paths.vaeParamPath + "|" +
            paths.vaeBinPath + "|" +
            paths.vaeEncoderParamPath + "|" +
            paths.vaeEncoderBinPath + "|" +
            paths.tokenizerVocabPath + "|" +
            tensorTextureFormat + "|" +
            encoderTensorTextureFormat + "|" +
            decoderTensorTextureFormat + "|" +
            BoolText(ResolveSpatialAttentionPack4()) + "|" +
            BoolText(keepRawConvWeightsForTexturePath);

        if (string.Equals(_loadedModelKey, modelKey, StringComparison.Ordinal) && _tokenizer != null)
            return;

        Release();
        EnsureRuntimeObjects();
        _tokenizer = new StableDiffusionSimpleTokenizer(paths.tokenizerVocabPath);
        _loadedModelKey = modelKey;
    }

    private async UniTask EnsureTextEncoderLoadedAsync(ResolvedPaths paths, CancellationToken ct)
    {
        EnsureModelIdentity(paths);
        EnsureRuntimeObjects();
        if (_textRepro?.Model != null)
            return;

        _textRepro ??= new NcnnRepro(_ops);
        ApplyCommonOptions(_textRepro);
        await LoadModelAsync(_textRepro, paths.textParamPath, paths.textBinPath, "text encoder", ct);
    }

    private async UniTask EnsureVaeEncoderLoadedAsync(ResolvedPaths paths, CancellationToken ct)
    {
        EnsureModelIdentity(paths);
        EnsureRuntimeObjects();
        if (_vaeEncoderRepro?.Model != null)
            return;

        _vaeEncoderRepro ??= new NcnnRepro(_ops);
        ApplyCommonOptions(_vaeEncoderRepro);
        ApplySpatialModelOptions(_vaeEncoderRepro);
        _vaeEncoderRepro.TensorTextureFormat = encoderTensorTextureFormat;
        await LoadModelAsync(_vaeEncoderRepro, paths.vaeEncoderParamPath, paths.vaeEncoderBinPath, "vae encoder", ct);
    }

    private async UniTask EnsureUnetLoadedAsync(ResolvedPaths paths, CancellationToken ct)
    {
        EnsureModelIdentity(paths);
        EnsureRuntimeObjects();
        if (_unetRepro?.Model != null)
        {
            ApplyCommonOptions(_unetRepro);
            ApplySpatialModelOptions(_unetRepro);
            AttachDebugLog(_unetRepro, "unet");
            return;
        }

        _unetRepro ??= new NcnnRepro(_ops);
        ApplyCommonOptions(_unetRepro);
        ApplySpatialModelOptions(_unetRepro);
        AttachDebugLog(_unetRepro, "unet");
        await LoadModelAsync(_unetRepro, paths.unetParamPath, paths.unetBinPath, "unet", ct);
    }

    private async UniTask EnsureVaeDecoderLoadedAsync(ResolvedPaths paths, CancellationToken ct)
    {
        EnsureModelIdentity(paths);
        EnsureRuntimeObjects();
        if (_vaeRepro?.Model != null)
            return;

        _vaeRepro ??= new NcnnRepro(_ops);
        ApplyCommonOptions(_vaeRepro);
        ApplySpatialModelOptions(_vaeRepro);
        _vaeRepro.TensorTextureFormat = decoderTensorTextureFormat;
        await LoadModelAsync(_vaeRepro, paths.vaeParamPath, paths.vaeBinPath, "vae decoder", ct);
    }

    private void DisposeTextModel()
    {
        try { _textRepro?.Dispose(); } catch { }
        _textRepro = null;
    }

    private void DisposeVaeEncoderModel()
    {
        try { _vaeEncoderRepro?.Dispose(); } catch { }
        _vaeEncoderRepro = null;
    }

    private void DisposeUnetModel()
    {
        FlushDebugLogToFile("unet", "unet_layer_debug.txt");
        try { _unetRepro?.Dispose(); } catch { }
        _unetRepro = null;
    }

    private void DisposeVaeDecoderModel()
    {
        try { _vaeRepro?.Dispose(); } catch { }
        _vaeRepro = null;
    }

    private void AttachDebugLog(NcnnRepro repro, string key)
    {
        if (repro == null)
            return;

        if (!enableDebugDump || string.IsNullOrWhiteSpace(_lastDumpDir) || string.IsNullOrWhiteSpace(key))
        {
            repro.DebugLog = null;
            return;
        }

        if (!_layerDebugLines.TryGetValue(key, out var lines) || lines == null)
        {
            lines = new List<string>(256);
            _layerDebugLines[key] = lines;
        }
        else
        {
            lines.Clear();
        }

        repro.DebugLog = line =>
        {
            if (string.IsNullOrWhiteSpace(line) || lines.Count >= 20000)
                return;

            lines.Add(line);
        };
    }

    private void FlushDebugLogToFile(string key, string fileName)
    {
        if (string.IsNullOrWhiteSpace(key) || string.IsNullOrWhiteSpace(fileName) || string.IsNullOrWhiteSpace(_lastDumpDir))
            return;

        if (!_layerDebugLines.TryGetValue(key, out var lines) || lines == null || lines.Count == 0)
            return;

        WriteAllTextSafe(Path.Combine(_lastDumpDir, fileName), string.Join(Environment.NewLine, lines));
    }

    private async UniTask ForceStageCleanupAsync(CancellationToken ct, string label = null)
    {
        var syncGpu = ResolveBoolEnv(StageSyncGpuEnvVar, false);
        var unloadUnusedAssets = ResolveBoolEnv(StageUnloadUnusedAssetsEnvVar, false);
        var releaseWinograd = ResolveBoolEnv(StageReleaseWinogradEnvVar, true);
        var runGc = ResolveBoolEnv(StageGcCollectEnvVar, true);
        LogStageTrace(
            "cleanup begin"
            + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label)
            + " | syncGpu=" + BoolText(syncGpu)
            + " | unloadUnusedAssets=" + BoolText(unloadUnusedAssets)
            + " | releaseWinograd=" + BoolText(releaseWinograd)
            + " | gc=" + BoolText(runGc));

        if (syncGpu)
        {
            _ops?.DebugSyncGpu();
        }
        else
        {
            LogStageTrace("cleanup skip sync gpu" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
        }

        if (releaseWinograd)
        {
            try
            {
                LogStageTrace("cleanup release winograd begin" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
                _ops?.ReleaseWinogradWorkspace();
                LogStageTrace("cleanup release winograd end" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
            }
            catch (Exception e)
            {
                LogStageTrace("cleanup release winograd failed" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label) + " | " + e.Message);
            }
        }
        else
        {
            LogStageTrace("cleanup skip release winograd" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
        }

        if (unloadUnusedAssets)
        {
            try
            {
                LogStageTrace("cleanup unload assets begin" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
                await Resources.UnloadUnusedAssets().ToUniTask(cancellationToken: ct);
                LogStageTrace("cleanup unload assets end" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
            }
            catch (Exception e)
            {
                LogStageTrace("cleanup unload assets failed" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label) + " | " + e.Message);
            }
        }
        else
        {
            LogStageTrace("cleanup skip unload assets" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
        }

        if (runGc)
        {
            try
            {
                LogStageTrace("cleanup gc begin" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                LogStageTrace("cleanup gc end" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
            }
            catch (Exception e)
            {
                LogStageTrace("cleanup gc failed" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label) + " | " + e.Message);
            }
        }
        else
        {
            LogStageTrace("cleanup skip gc" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
        }

        await UniTask.Yield();
        ct.ThrowIfCancellationRequested();
        LogStageTrace("cleanup end" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
    }

    private async UniTask EnsureLoadedAsync(CancellationToken ct)
    {
        var paths = ResolvePaths();
        var modelKey =
            paths.textParamPath + "|" +
            paths.textBinPath + "|" +
            paths.unetParamPath + "|" +
            paths.unetBinPath + "|" +
            paths.vaeParamPath + "|" +
            paths.vaeBinPath + "|" +
            paths.vaeEncoderParamPath + "|" +
            paths.vaeEncoderBinPath + "|" +
            paths.tokenizerVocabPath + "|" +
            tensorTextureFormat + "|" +
            encoderTensorTextureFormat + "|" +
            decoderTensorTextureFormat + "|" +
            BoolText(ResolveSpatialAttentionPack4()) + "|" +
            BoolText(keepRawConvWeightsForTexturePath);
        if (string.Equals(_loadedModelKey, modelKey, StringComparison.Ordinal) && _tokenizer != null)
            return;

        Release();
        EnsureRuntimeObjects();
        ApplyCommonOptions(_textRepro);
        ApplyCommonOptions(_unetRepro);
        ApplyCommonOptions(_vaeRepro);
        ApplyCommonOptions(_vaeEncoderRepro);
        ApplySpatialModelOptions(_unetRepro);
        ApplySpatialModelOptions(_vaeRepro);
        ApplySpatialModelOptions(_vaeEncoderRepro);
        _vaeRepro.TensorTextureFormat = decoderTensorTextureFormat;
        _vaeEncoderRepro.TensorTextureFormat = encoderTensorTextureFormat;

        _tokenizer = new StableDiffusionSimpleTokenizer(paths.tokenizerVocabPath);

        await LoadModelAsync(_textRepro, paths.textParamPath, paths.textBinPath, "text encoder", ct);
        await LoadModelAsync(_unetRepro, paths.unetParamPath, paths.unetBinPath, "unet", ct);
        await LoadModelAsync(_vaeRepro, paths.vaeParamPath, paths.vaeBinPath, "vae decoder", ct);
        await LoadModelAsync(_vaeEncoderRepro, paths.vaeEncoderParamPath, paths.vaeEncoderBinPath, "vae encoder", ct);
        _loadedModelKey = modelKey;
    }

    private async UniTask LoadModelAsync(NcnnRepro repro, string paramPath, string binPath, string label, CancellationToken ct)
    {
        if (repro == null)
            throw new ArgumentNullException(nameof(repro));
        if (!File.Exists(paramPath))
            throw new FileNotFoundException(label + " param not found.", paramPath);
        if (!File.Exists(binPath))
            throw new FileNotFoundException(label + " bin not found.", binPath);

        string paramText;
        string pnnxParamText = null;
        if (Application.isBatchMode)
        {
            ct.ThrowIfCancellationRequested();
            paramText = File.ReadAllText(paramPath);
            pnnxParamText = TryReadPnnxSidecarParam(paramPath);
        }
        else
        {
            paramText = await File.ReadAllTextAsync(paramPath, ct);
            pnnxParamText = await TryReadPnnxSidecarParamAsync(paramPath, ct);
        }

        using var fs = new FileStream(binPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, false);
        using var br = new NcnnBinReader(fs);
        if (Application.isBatchMode)
        {
            ct.ThrowIfCancellationRequested();
            repro.LoadModel(paramText, br, progress => LogLoadProgress(label, progress));
        }
        else
        {
            await repro.LoadModelAsync(paramText, br, progress => LogLoadProgress(label, progress), ct);
        }

        if (!string.IsNullOrWhiteSpace(pnnxParamText))
            repro.MergePnnxStringParams(pnnxParamText, overwriteExisting: false);

        LogLoadProfile(label, repro.LastLoadProfile);
    }

    private static string TryReadPnnxSidecarParam(string runtimeParamPath)
    {
        var sidecarPath = ResolvePnnxSidecarParamPath(runtimeParamPath);
        if (string.IsNullOrWhiteSpace(sidecarPath) || !File.Exists(sidecarPath))
            return null;

        try
        {
            return File.ReadAllText(sidecarPath);
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[SDInpaint] Failed to read pnnx sidecar param: " + e.Message);
            return null;
        }
    }

    private static async UniTask<string> TryReadPnnxSidecarParamAsync(string runtimeParamPath, CancellationToken ct)
    {
        var sidecarPath = ResolvePnnxSidecarParamPath(runtimeParamPath);
        if (string.IsNullOrWhiteSpace(sidecarPath) || !File.Exists(sidecarPath))
            return null;

        try
        {
            return await File.ReadAllTextAsync(sidecarPath, ct);
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[SDInpaint] Failed to read pnnx sidecar param: " + e.Message);
            return null;
        }
    }

    private static string ResolvePnnxSidecarParamPath(string runtimeParamPath)
    {
        if (string.IsNullOrWhiteSpace(runtimeParamPath))
            return null;

        var dir = Path.GetDirectoryName(runtimeParamPath);
        var fileName = Path.GetFileNameWithoutExtension(runtimeParamPath);
        var ext = Path.GetExtension(runtimeParamPath);
        if (string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(fileName) || string.IsNullOrWhiteSpace(ext))
            return null;

        return Path.Combine(dir, fileName + ".pnnx" + ext);
    }

    private ResolvedPaths ResolvePaths()
    {
        if (_resolvedPaths.HasValue)
            return _resolvedPaths.Value;

        var projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (string.IsNullOrWhiteSpace(projectRoot))
            throw new DirectoryNotFoundException("Failed to resolve project root from Application.dataPath.");

        var sdRootOverride = Environment.GetEnvironmentVariable(StableDiffusionModelRootEnvVar);
        var sdRoot = !string.IsNullOrWhiteSpace(sdRootOverride)
            ? Path.GetFullPath(Environment.ExpandEnvironmentVariables(sdRootOverride.Trim()))
            : Path.GetFullPath(Path.Combine(Application.streamingAssetsPath, stableDiffusionRootRelativePath));

        var inpaintRootOverride = Environment.GetEnvironmentVariable(InpaintModelRootEnvVar);
        var inpaintRoot = !string.IsNullOrWhiteSpace(inpaintRootOverride)
            ? Path.GetFullPath(Environment.ExpandEnvironmentVariables(inpaintRootOverride.Trim()))
            : Path.GetFullPath(Path.Combine(Application.streamingAssetsPath, inpaintingModelRootRelativePath));

        static void AddUniqueRoot(List<string> roots, string root)
        {
            if (roots == null || string.IsNullOrWhiteSpace(root))
                return;

            var fullRoot = Path.GetFullPath(root);
            for (var i = 0; i < roots.Count; i++)
            {
                if (string.Equals(roots[i], fullRoot, StringComparison.OrdinalIgnoreCase))
                    return;
            }

            roots.Add(fullRoot);
        }

        var sdRoots = new List<string>();
        AddUniqueRoot(sdRoots, sdRoot);
        if (useReferenceAssetFallback)
        {
            AddUniqueRoot(sdRoots, Path.Combine(projectRoot, "ref", "Stable-Diffusion-NCNN-main", "Windows", "Binary", "x64", "assets"));
            AddUniqueRoot(sdRoots, Path.Combine(projectRoot, "ref", "Stable-Diffusion-NCNN-main", "x86", "exe", "assets"));
            AddUniqueRoot(sdRoots, Path.Combine(projectRoot, "ref", "Stable-Diffusion-NCNN-main", "x86", "linux", "assets"));
        }

        var inpaintRoots = new List<string>();
        AddUniqueRoot(inpaintRoots, inpaintRoot);
        AddUniqueRoot(inpaintRoots, Path.Combine(inpaintRoot, "ncnn"));
        AddUniqueRoot(inpaintRoots, Path.Combine(projectRoot, "Tools", "sd15inpainting2ncnnExporter", "output", "ncnn"));
        AddUniqueRoot(inpaintRoots, Path.Combine(projectRoot, "Tools", "sd15inpainting2ncnnExporter", "output", "ncnn_scriptcheck"));

        string FindExact(IReadOnlyList<string> roots, string fileName)
        {
            if (roots == null || string.IsNullOrWhiteSpace(fileName))
                return null;

            for (var i = 0; i < roots.Count; i++)
            {
                var root = roots[i];
                if (string.IsNullOrWhiteSpace(root))
                    continue;
                var path = Path.Combine(root, fileName);
                if (File.Exists(path))
                    return path;
            }

            return null;
        }

        string FindFirst(IReadOnlyList<string> roots, params string[] fileNames)
        {
            if (fileNames == null)
                return null;

            for (var i = 0; i < fileNames.Length; i++)
            {
                var found = FindExact(roots, fileNames[i]);
                if (!string.IsNullOrWhiteSpace(found))
                    return found;
            }

            return null;
        }

        var textParamPath = FindExact(sdRoots, "FrozenCLIPEmbedder-fp16.param");
        var textBinPath = FindExact(sdRoots, "FrozenCLIPEmbedder-fp16.bin");
        var unetParamPath = FindExact(inpaintRoots, "unet.param");
        var unetBinPath = FindExact(inpaintRoots, "unet.bin");
        var vaeParamPath = FindFirst(sdRoots,
            "AutoencoderKL-512-512-fp16-opt.param",
            "AutoencoderKL-512-fp16-opt.param",
            "AutoencoderKL-base-fp16.param");
        var vaeBinPath = FindExact(sdRoots, "AutoencoderKL-fp16.bin");
        var vaeEncoderParamPath = FindExact(sdRoots, "AutoencoderKL-encoder-512-512-fp16.param");
        var vaeEncoderBinPath = FindExact(sdRoots, "AutoencoderKL-encoder-512-512-fp16.bin");
        var vocabPath = FindExact(sdRoots, "vocab.txt");

        if (string.IsNullOrWhiteSpace(textParamPath))
            throw new FileNotFoundException("FrozenCLIPEmbedder-fp16.param not found under StableDiffusion roots.", sdRoot);
        if (string.IsNullOrWhiteSpace(textBinPath))
            throw new FileNotFoundException("FrozenCLIPEmbedder-fp16.bin not found under StableDiffusion roots.", sdRoot);
        if (string.IsNullOrWhiteSpace(unetParamPath))
            throw new FileNotFoundException("SD inpainting unet.param not found under inpainting roots.", inpaintRoot);
        if (string.IsNullOrWhiteSpace(unetBinPath))
            throw new FileNotFoundException("SD inpainting unet.bin not found under inpainting roots.", inpaintRoot);
        if (string.IsNullOrWhiteSpace(vaeParamPath))
            throw new FileNotFoundException("AutoencoderKL decoder param not found under StableDiffusion roots.", sdRoot);
        if (string.IsNullOrWhiteSpace(vaeBinPath))
            throw new FileNotFoundException("AutoencoderKL-fp16.bin not found under StableDiffusion roots.", sdRoot);
        if (string.IsNullOrWhiteSpace(vaeEncoderParamPath))
            throw new FileNotFoundException("AutoencoderKL-encoder-512-512-fp16.param not found under StableDiffusion roots.", sdRoot);
        if (string.IsNullOrWhiteSpace(vaeEncoderBinPath))
            throw new FileNotFoundException("AutoencoderKL-encoder-512-512-fp16.bin not found under StableDiffusion roots.", sdRoot);
        if (string.IsNullOrWhiteSpace(vocabPath))
            throw new FileNotFoundException("vocab.txt not found under StableDiffusion roots.", sdRoot);

        _resolvedPaths = new ResolvedPaths(
            textParamPath,
            textBinPath,
            unetParamPath,
            unetBinPath,
            vaeParamPath,
            vaeBinPath,
            vaeEncoderParamPath,
            vaeEncoderBinPath,
            vocabPath);
        return _resolvedPaths.Value;
    }

    private async UniTask<float[]> EncodePromptAsync(string prompt, string label, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (_tokenizer == null)
            throw new InvalidOperationException("Tokenizer is not loaded.");

        ComputeBuffer tokenBuffer = null;
        ComputeBuffer multiplierBuffer = null;
        ComputeBuffer condBuffer = null;
        bool? previousTextTempBufferGuard = null;
        try
        {
            var tokens = new int[TokenCount];
            var multipliers = new float[TokenCount];
            _tokenizer.TokenizePrompt77(prompt, tokens, multipliers);
            LogStageTrace("text encode tokens ready"
                + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label)
                + " | tokenCount=" + tokens.Length.ToString(CultureInfo.InvariantCulture));
            tokenBuffer = NewTrackedBuffer(tokens.Length, sizeof(int), ComputeBufferType.Structured, "SDInpaint.TextTokens." + (label ?? "prompt"));
            tokenBuffer.SetData(tokens);
            multiplierBuffer = NewFloatBuffer(multipliers, "SDInpaint.TextMultipliers." + (label ?? "prompt"));
            condBuffer = NewFloatBuffer(new float[TextEmbeddingWidth], "SDInpaint.TextCondSeed." + (label ?? "prompt"));
            var tokenView = new NcnnTensorBuffer(tokenBuffer, 1, tokens.Length, 1, 1, 1, false);
            var multiplierView = new NcnnTensorBuffer(multiplierBuffer, 1, multipliers.Length, 1, 1, 1, false);
            var condView = new NcnnTensorBuffer(condBuffer, 2, TextEmbeddingWidth, 1, 1, 1, false);
            LogStageTrace("text infer begin" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
            previousTextTempBufferGuard = _textRepro.DisallowInferenceTempComputeBuffers;
            _textRepro.DisallowInferenceTempComputeBuffers = false;
            using var infer = _textRepro.InferWithMultiInputs(
                null,
                new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal)
                {
                    { TextEncoderTokenBlobName, tokenView },
                    { TextEncoderMultiplierBlobName, multiplierView },
                    { TextEncoderCondBlobName, condView }
                },
                new HashSet<string>(StringComparer.Ordinal) { TextEncoderOutputBlobName });

            LogStageTrace("text readback begin" + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label));
            var output = infer.GetBufferData(TextEncoderOutputBlobName);
            LogStageTrace("text readback end"
                + (string.IsNullOrWhiteSpace(label) ? string.Empty : " | " + label)
                + " | values=" + (output?.Length ?? 0).ToString(CultureInfo.InvariantCulture));
            await UniTask.Yield();
            return output;
        }
        finally
        {
            if (previousTextTempBufferGuard.HasValue && _textRepro != null)
                _textRepro.DisallowInferenceTempComputeBuffers = previousTextTempBufferGuard.Value;
            DisposeBuffer(tokenBuffer, "SDInpaint.TextTokens." + (label ?? "prompt"));
            DisposeBuffer(multiplierBuffer, "SDInpaint.TextMultipliers." + (label ?? "prompt"));
            DisposeBuffer(condBuffer, "SDInpaint.TextCondSeed." + (label ?? "prompt"));
        }
    }

    private async UniTask<ComputeBuffer> EncodeImageLatentsAsync(Texture source, INormalRng rng, CancellationToken ct)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (rng == null)
            throw new ArgumentNullException(nameof(rng));

        RenderTexture inputPack4 = null;
        RenderTexture stdTex = null;
        ComputeBuffer inputBuf = null;
        ComputeBuffer meanBuf = null;
        ComputeBuffer stdBuf = null;
        ComputeBuffer noiseBuf = null;
        ComputeBuffer scaledNoiseBuf = null;
        ComputeBuffer latentBuf = null;
        try
        {
            inputBuf = CreateEncoderInputBufferNcnn(source, ModelImageSize, ModelImageSize);
            inputPack4 = _vaeEncoderRepro.RentTempArray(ModelImageSize, ModelImageSize, 1, encoderTensorTextureFormat);
            _ops.FillPack4FromBufferCHW(inputBuf, ModelImageSize, ModelImageSize, 3, inputPack4);

            using (var infer = _vaeEncoderRepro.Infer(
                       inputPack4,
                       1,
                       VaeEncoderInputBlobName,
                       new HashSet<string>(StringComparer.Ordinal) { VaeEncoderMeanBlobName, VaeEncoderStdBlobName }))
            {
                meanBuf = infer.ExtractBuffer(VaeEncoderMeanBlobName);
                try
                {
                    stdBuf = infer.ExtractBuffer(VaeEncoderStdBlobName);
                }
                catch
                {
                    stdTex = infer.ExtractTexture(VaeEncoderStdBlobName);
                }
            }

            if (meanBuf == null)
                throw new InvalidOperationException("VAE encoder mean output is missing.");

            if (stdBuf == null)
            {
                if (stdTex == null)
                    throw new InvalidOperationException("VAE encoder std output is missing.");
                stdBuf = _vaeEncoderRepro.RentTempBuffer(LatentElementCount(), sizeof(float));
                _ops.Pack4ToBufferCHW(stdTex, LatentSize, LatentSize, LatentChannels, stdBuf);
            }

            noiseBuf = NewFloatBuffer(GenerateGaussian(LatentElementCount(), rng), "SDInpaint.EncodeImageLatents.noise");
            scaledNoiseBuf = NewTrackedBuffer(noiseBuf.count, sizeof(float), ComputeBufferType.Structured, "SDInpaint.EncodeImageLatents.scaledNoise");
            latentBuf = NewTrackedBuffer(noiseBuf.count, sizeof(float), ComputeBufferType.Structured, "SDInpaint.EncodeImageLatents.latent");
            _ops.BinaryOpBuf(stdBuf, noiseBuf, noiseBuf.count, 2, scaledNoiseBuf);
            _ops.BinaryOpBuf(meanBuf, scaledNoiseBuf, noiseBuf.count, 0, latentBuf);
            _ops.MulScalarInplace(latentBuf, LatentScale, latentBuf.count);
            await UniTask.Yield();
            var result = latentBuf;
            latentBuf = null;
            return result;
        }
        finally
        {
            if (inputPack4 != null)
                _vaeEncoderRepro?.ReturnTempArray(inputPack4);
            if (stdTex != null)
                _vaeEncoderRepro?.ReturnTempArray(stdTex);
            if (inputBuf != null)
                DisposeBuffer(inputBuf, "SDInpaint.EncodeImageLatents.encoderInput");
            if (meanBuf != null)
                _vaeEncoderRepro?.ReturnTempBuffer(meanBuf);
            if (stdBuf != null)
                _vaeEncoderRepro?.ReturnTempBuffer(stdBuf);
            if (noiseBuf != null)
                DisposeBuffer(noiseBuf, "SDInpaint.EncodeImageLatents.noise");
            if (scaledNoiseBuf != null)
                DisposeBuffer(scaledNoiseBuf, "SDInpaint.EncodeImageLatents.scaledNoise");
            if (latentBuf != null)
                DisposeBuffer(latentBuf, "SDInpaint.EncodeImageLatents.latent");
        }
    }

    private async UniTask<RenderTexture> EncodeImageLatentsPack4Async(Texture source, INormalRng rng, CancellationToken ct)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (rng == null)
            throw new ArgumentNullException(nameof(rng));

        RenderTexture inputPack4 = null;
        RenderTexture meanTex = null;
        RenderTexture stdTex = null;
        RenderTexture noiseTex = null;
        RenderTexture scaledNoiseTex = null;
        RenderTexture latentTex = null;
        RenderTexture scaledLatentTex = null;
        try
        {
            inputPack4 = CreateEncoderInputPack4Ncnn(source);
            if (inputPack4 == null)
                throw new InvalidOperationException("Failed to create VAE encoder pack4 input.");

            using (var infer = _vaeEncoderRepro.Infer(
                       inputPack4,
                       1,
                       VaeEncoderInputBlobName,
                       new HashSet<string>(StringComparer.Ordinal) { VaeEncoderMeanBlobName, VaeEncoderStdBlobName }))
            {
                meanTex = infer.ExtractTexture(VaeEncoderMeanBlobName);
                stdTex = infer.ExtractTexture(VaeEncoderStdBlobName);
            }

            if (meanTex == null || stdTex == null)
                throw new InvalidOperationException("VAE encoder pack4 output is missing.");

            noiseTex = CreateLatentNoisePack4(rng);
            scaledNoiseTex = _vaeEncoderRepro.RentTempArray(LatentSize, LatentSize, 1, encoderTensorTextureFormat);
            latentTex = _vaeEncoderRepro.RentTempArray(LatentSize, LatentSize, 1, encoderTensorTextureFormat);
            scaledLatentTex = _vaeEncoderRepro.RentTempArray(LatentSize, LatentSize, 1, encoderTensorTextureFormat);

            _ops.BinaryOpPack4(stdTex, noiseTex, 1, 2, scaledNoiseTex);
            _ops.BinaryOpPack4(meanTex, scaledNoiseTex, 1, 0, latentTex);
            _ops.BinaryOpScalarPack4(latentTex, LatentScale, 1, 2, scaledLatentTex);

            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();
            var result = scaledLatentTex;
            scaledLatentTex = null;
            return result;
        }
        finally
        {
            if (inputPack4 != null)
                _vaeEncoderRepro?.ReturnTempArray(inputPack4);
            if (meanTex != null)
                _vaeEncoderRepro?.ReturnTempArray(meanTex);
            if (stdTex != null)
                _vaeEncoderRepro?.ReturnTempArray(stdTex);
            if (noiseTex != null)
                _vaeEncoderRepro?.ReturnTempArray(noiseTex);
            if (scaledNoiseTex != null)
                _vaeEncoderRepro?.ReturnTempArray(scaledNoiseTex);
            if (latentTex != null)
                _vaeEncoderRepro?.ReturnTempArray(latentTex);
            if (scaledLatentTex != null)
                _vaeEncoderRepro?.ReturnTempArray(scaledLatentTex);
        }
    }

    private RenderTexture CreateEncoderInputPack4Ncnn(Texture source)
    {
        Texture2D tempTexture = null;
        try
        {
            var pixels = ReadTexturePixels32(source, out var srcW, out var srcH, out tempTexture);
            var input = CreateEncoderInputChwNcnn(pixels, srcW, srcH, ModelImageSize, ModelImageSize);
            return CreateTensorPack4Texture(_vaeEncoderRepro, input, 3, ModelImageSize, ModelImageSize, 1, 3, encoderTensorTextureFormat);
        }
        finally
        {
            if (tempTexture != null)
                DestroyImmediate(tempTexture);
        }
    }

    private RenderTexture CreateLatentNoisePack4(INormalRng rng)
    {
        return CreateTensorPack4Texture(
            _unetRepro ?? _vaeEncoderRepro,
            GenerateGaussian(LatentElementCount(), rng),
            3,
            LatentSize,
            LatentSize,
            1,
            LatentChannels,
            tensorTextureFormat);
    }

    private RenderTexture CreateMaskPack4Texture(Texture2D mask)
    {
        if (mask == null)
            return null;

        var width = mask.width;
        var height = mask.height;
        var pixels = mask.GetPixels32();
        var data = new float[pixels.Length];
        for (var y = 0; y < height; y++)
        {
            var srcRow = (height - 1 - y) * width;
            var dstRow = y * width;
            for (var x = 0; x < width; x++)
                data[dstRow + x] = SampleMaskWeight(pixels[srcRow + x]);
        }

        return CreateTensorPack4Texture(_unetRepro, data, 3, LatentSize, LatentSize, 1, 1, tensorTextureFormat);
    }

    private static RenderTexture CreateTensorPack4Texture(
        NcnnRepro owner,
        float[] data,
        int dims,
        int width,
        int height,
        int depth,
        int channels,
        RenderTextureFormat format)
    {
        if (owner == null)
            throw new ArgumentNullException(nameof(owner));
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        width = Mathf.Max(1, width);
        height = Mathf.Max(1, height);
        depth = Mathf.Max(1, depth);
        channels = Mathf.Max(1, channels);
        var packsPerDepth = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
        var sliceCount = dims == 4 ? depth * packsPerDepth : packsPerDepth;
        var rt = owner.RentTempArray(width, height, sliceCount, format);
        var textureFormat = format == RenderTextureFormat.ARGBFloat ? TextureFormat.RGBAFloat : TextureFormat.RGBAHalf;
        var wh = width * height;
        Texture2D slice = null;
        try
        {
            for (var z = 0; z < (dims == 4 ? depth : 1); z++)
            {
                for (var pack = 0; pack < packsPerDepth; pack++)
                {
                    var sliceIndex = dims == 4 ? z * packsPerDepth + pack : pack;
                    slice = new Texture2D(width, height, textureFormat, false, true)
                    {
                        wrapMode = TextureWrapMode.Clamp,
                        filterMode = FilterMode.Point
                    };
                    var pixels = new Color[wh];
                    for (var i = 0; i < wh; i++)
                    {
                        var c0 = pack * 4;
                        pixels[i] = new Color(
                            ReadTensorValue(data, dims, width, height, depth, channels, i, z, c0 + 0),
                            ReadTensorValue(data, dims, width, height, depth, channels, i, z, c0 + 1),
                            ReadTensorValue(data, dims, width, height, depth, channels, i, z, c0 + 2),
                            ReadTensorValue(data, dims, width, height, depth, channels, i, z, c0 + 3));
                    }

                    slice.SetPixels(pixels);
                    slice.Apply(false, false);
                    Graphics.CopyTexture(slice, 0, 0, rt, sliceIndex, 0);
                    DestroyImmediate(slice);
                    slice = null;
                }
            }

            return rt;
        }
        catch
        {
            owner.ReturnTempArray(rt);
            throw;
        }
        finally
        {
            if (slice != null)
                DestroyImmediate(slice);
        }
    }

    private static float ReadTensorValue(float[] data, int dims, int width, int height, int depth, int channels, int whIndex, int z, int channel)
    {
        if (channel >= channels)
            return 0f;

        var wh = width * height;
        var index = dims == 4
            ? ((channel * depth + z) * wh + whIndex)
            : (channel * wh + whIndex);
        return index >= 0 && index < data.Length ? data[index] : 0f;
    }

    private async UniTask<Texture2D> DecodeLatentsAsync(ComputeBuffer latentBuf, CancellationToken ct)
    {
        if (latentBuf == null)
            return null;

        ComputeBuffer scaledLatent = null;
        RenderTexture inputTex = null;
        RenderTexture decodedTex = null;
        RenderTexture clippedTex = null;
        RenderTexture rgbRt = null;
        try
        {
            scaledLatent = _vaeRepro.RentTempBuffer(latentBuf.count, sizeof(float));
            _ops.CopyBuf(latentBuf, scaledLatent, latentBuf.count);
            _ops.MulScalarInplace(scaledLatent, InvLatentScale, scaledLatent.count);

            inputTex = _vaeRepro.RentTempArray(LatentSize, LatentSize, 1, decoderTensorTextureFormat);
            _ops.FillPack4FromBufferCHW(scaledLatent, LatentSize, LatentSize, LatentChannels, inputTex);

            using (var infer = _vaeRepro.Infer(inputTex, 1, VaeDecoderInputBlobName, new HashSet<string>(StringComparer.Ordinal) { VaeDecoderOutputBlobName }))
            {
                decodedTex = infer.ExtractTexture(VaeDecoderOutputBlobName);
            }

            if (decodedTex == null)
                return null;

            clippedTex = _vaeRepro.RentTempArray(decodedTex.width, decodedTex.height, 1, decoderTensorTextureFormat);
            _ops.ClipPack4(decodedTex, -1f, 1f, 1, clippedTex);
            rgbRt = GetTemporaryRt(decodedTex.width, decodedTex.height, RenderTextureFormat.ARGB32, true, "SDInpaint.DecoderRgbRt");
            _ops.Pack4ToRgb01(clippedTex, rgbRt, true);
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();
            return RenderTextureToTexture2D(rgbRt, ModelImageSize, ModelImageSize);
        }
        finally
        {
            if (scaledLatent != null)
                _vaeRepro?.ReturnTempBuffer(scaledLatent);
            if (inputTex != null)
                _vaeRepro?.ReturnTempArray(inputTex);
            if (decodedTex != null)
                _vaeRepro?.ReturnTempArray(decodedTex);
            if (clippedTex != null)
                _vaeRepro?.ReturnTempArray(clippedTex);
            if (rgbRt != null)
                ReleaseTemporaryRt(rgbRt, "SDInpaint.DecoderRgbRt");
        }
    }

    private async UniTask<Texture2D> DecodeLatentsTextureAsync(RenderTexture latentTex, CancellationToken ct)
    {
        if (latentTex == null)
            return null;

        RenderTexture scaledLatentTex = null;
        RenderTexture decodedTex = null;
        RenderTexture clippedTex = null;
        RenderTexture rgbRt = null;
        try
        {
            scaledLatentTex = _vaeRepro.RentTempArray(LatentSize, LatentSize, 1, decoderTensorTextureFormat);
            _ops.BinaryOpScalarPack4(latentTex, InvLatentScale, 1, 2, scaledLatentTex);

            using (var infer = _vaeRepro.Infer(scaledLatentTex, 1, VaeDecoderInputBlobName, new HashSet<string>(StringComparer.Ordinal) { VaeDecoderOutputBlobName }))
            {
                decodedTex = infer.ExtractTexture(VaeDecoderOutputBlobName);
            }

            if (decodedTex == null)
                return null;

            clippedTex = _vaeRepro.RentTempArray(decodedTex.width, decodedTex.height, 1, decoderTensorTextureFormat);
            _ops.ClipPack4(decodedTex, -1f, 1f, 1, clippedTex);
            rgbRt = GetTemporaryRt(decodedTex.width, decodedTex.height, RenderTextureFormat.ARGB32, true, "SDInpaint.DecoderRgbRt");
            _ops.Pack4ToRgb01(clippedTex, rgbRt, true);
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();
            return RenderTextureToTexture2D(rgbRt, ModelImageSize, ModelImageSize);
        }
        finally
        {
            if (scaledLatentTex != null)
                _vaeRepro?.ReturnTempArray(scaledLatentTex);
            if (decodedTex != null)
                _vaeRepro?.ReturnTempArray(decodedTex);
            if (clippedTex != null)
                _vaeRepro?.ReturnTempArray(clippedTex);
            if (rgbRt != null)
                ReleaseTemporaryRt(rgbRt, "SDInpaint.DecoderRgbRt");
        }
    }

    private async UniTask<ComputeBuffer> RunCfgUnetAsync(
        ComputeBuffer latentsBuf,
        ComputeBuffer maskBuf,
        ComputeBuffer maskedLatentBuf,
        int timestep,
        NcnnTensorBuffer condView,
        NcnnTensorBuffer uncondView,
        float guidanceScale,
        CancellationToken ct)
    {
        RenderTexture inputTex = null;
        ComputeBuffer concatBuf = null;
        ComputeBuffer timestepBuf = null;
        ComputeBuffer condOut = null;
        ComputeBuffer uncondOut = null;
        ComputeBuffer diffBuf = null;
        ComputeBuffer scaledDiffBuf = null;
        ComputeBuffer finalBuf = null;
        List<UnetCacheBlob> unetCache = null;
        try
        {
            var plane = LatentSize * LatentSize;
            concatBuf = _unetRepro.RentTempBuffer(plane * UnetInputChannels, sizeof(float));
            _ops.CopyBufPartial(latentsBuf, 0, concatBuf, plane * LatentChannels, 0);
            _ops.CopyBufPartial(maskBuf, 0, concatBuf, plane, plane * LatentChannels);
            _ops.CopyBufPartial(maskedLatentBuf, 0, concatBuf, plane * LatentChannels, plane * (LatentChannels + 1));

            inputTex = _unetRepro.RentTempArray(LatentSize, LatentSize, Mathf.CeilToInt(UnetInputChannels / 4f), tensorTextureFormat);
            _ops.FillPack4FromBufferCHW(concatBuf, LatentSize, LatentSize, UnetInputChannels, inputTex);

            timestepBuf = NewFloatBuffer(new[] { (float)timestep }, "SDInpaint.UnetTimestep");
            var timestepView = new NcnnTensorBuffer(timestepBuf, 1, 1, 1, 1, 1, false);

            if (useOfficialUnetCache)
                unetCache = new List<UnetCacheBlob>(OfficialUnetCacheBlobNames.Length);

            condOut = RunUnetOnce(inputTex, timestepView, condView, "cond", unetCache, null);
            var cacheForUncond = unetCache != null && unetCache.Count == OfficialUnetCacheBlobNames.Length ? unetCache : null;
            uncondOut = RunUnetOnce(inputTex, timestepView, uncondView, "uncond", null, cacheForUncond);
            if (condOut == null || uncondOut == null)
                return null;

            diffBuf = _unetRepro.RentTempBuffer(condOut.count, sizeof(float));
            scaledDiffBuf = _unetRepro.RentTempBuffer(condOut.count, sizeof(float));
            finalBuf = _unetRepro.RentTempBuffer(condOut.count, sizeof(float));
            _ops.BinaryOpBuf(condOut, uncondOut, condOut.count, 1, diffBuf);
            _ops.BinaryOpScalarBuf(diffBuf, guidanceScale, diffBuf.count, 2, scaledDiffBuf);
            _ops.BinaryOpBuf(uncondOut, scaledDiffBuf, scaledDiffBuf.count, 0, finalBuf);

            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();
            return finalBuf;
        }
        finally
        {
            if (inputTex != null)
                _unetRepro?.ReturnTempArray(inputTex);
            if (concatBuf != null)
                _unetRepro?.ReturnTempBuffer(concatBuf);
            if (timestepBuf != null)
                DisposeBuffer(timestepBuf, "SDInpaint.UnetTimestep");
            if (condOut != null)
                _unetRepro?.ReturnTempBuffer(condOut);
            if (uncondOut != null)
                _unetRepro?.ReturnTempBuffer(uncondOut);
            if (diffBuf != null)
                _unetRepro?.ReturnTempBuffer(diffBuf);
            if (scaledDiffBuf != null)
                _unetRepro?.ReturnTempBuffer(scaledDiffBuf);
            ReleaseUnetCache(unetCache);
        }
    }

    private async UniTask<RenderTexture> RunCfgUnetPack4Async(
        RenderTexture latentsTex,
        RenderTexture maskTex,
        RenderTexture maskedLatentTex,
        int timestep,
        RenderTexture condTex,
        RenderTexture uncondTex,
        float guidanceScale,
        CancellationToken ct)
    {
        if (useCommandBuffer)
            return await RunCfgUnetPack4CommandBufferAsync(latentsTex, maskTex, maskedLatentTex, timestep, condTex, uncondTex, guidanceScale, ct);

        RenderTexture inputTex = null;
        RenderTexture timestepTex = null;
        RenderTexture condOutTex = null;
        RenderTexture uncondOutTex = null;
        RenderTexture finalTex = null;
        List<UnetCacheBlob> unetCache = null;
        try
        {
            if (condTex == null || uncondTex == null)
                throw new InvalidOperationException("UNet pack4 path requires prompt condition textures.");

            inputTex = BuildUnetInputPack4(latentsTex, maskTex, maskedLatentTex);
            timestepTex = CreateTensorPack4Texture(_unetRepro, new[] { (float)timestep }, 1, 1, 1, 1, 1, tensorTextureFormat);
            var shouldDumpFirstStep = enableDebugDump && !_unetDebugDumped && !string.IsNullOrWhiteSpace(_lastDumpDir);

            if (shouldDumpFirstStep)
            {
                DumpTextureRawF32Safe(Path.Combine(_lastDumpDir, "unity_unet_in0_f32.bin"), inputTex, UnetInputChannels);
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "unity_unet_in0_stats.txt"), inputTex, UnetInputChannels);
                WriteFloatArrayRawF32Safe(Path.Combine(_lastDumpDir, "unity_unet_timestep_f32.bin"), new[] { (float)timestep });
            }

            if (useOfficialUnetCache)
                unetCache = new List<UnetCacheBlob>(OfficialUnetCacheBlobNames.Length);

            condOutTex = RunUnetOnceTexture(inputTex, timestepTex, condTex, "cond", unetCache, null);
            if (condOutTex != null)
            {
                var preservedCondOutTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
                _ops.CopyPack4(condOutTex, 0, preservedCondOutTex, 0, Mathf.Max(1, Mathf.CeilToInt(LatentChannels / 4f)));
                _unetRepro.ReturnTempArray(condOutTex);
                condOutTex = preservedCondOutTex;
            }
            var cacheForUncond = unetCache != null && unetCache.Count == OfficialUnetCacheBlobNames.Length ? unetCache : null;
            uncondOutTex = RunUnetOnceTexture(inputTex, timestepTex, uncondTex, "uncond", null, cacheForUncond);
            if (uncondOutTex != null)
            {
                var preservedUncondOutTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
                _ops.CopyPack4(uncondOutTex, 0, preservedUncondOutTex, 0, Mathf.Max(1, Mathf.CeilToInt(LatentChannels / 4f)));
                _unetRepro.ReturnTempArray(uncondOutTex);
                uncondOutTex = preservedUncondOutTex;
            }
            if (condOutTex == null || uncondOutTex == null)
                return null;

            if (shouldDumpFirstStep)
            {
                DumpTextureRawF32Safe(Path.Combine(_lastDumpDir, "unity_unet_cond_out_f32.bin"), condOutTex, LatentChannels);
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "unity_unet_cond_out_stats.txt"), condOutTex, LatentChannels);
                DumpTextureRawF32Safe(Path.Combine(_lastDumpDir, "unity_unet_uncond_out_f32.bin"), uncondOutTex, LatentChannels);
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "unity_unet_uncond_out_stats.txt"), uncondOutTex, LatentChannels);
            }

            finalTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            _ops.AddPack4(condOutTex, uncondOutTex, guidanceScale, 1f - guidanceScale, 1, finalTex);

            if (shouldDumpFirstStep)
            {
                DumpTextureRawF32Safe(Path.Combine(_lastDumpDir, "unity_unet_eps_f32.bin"), finalTex, LatentChannels);
                DumpLatentTextureStatsSafe(Path.Combine(_lastDumpDir, "unity_unet_eps_stats.txt"), finalTex, LatentChannels);
                _unetDebugDumped = true;
            }

            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();
            var result = finalTex;
            finalTex = null;
            return result;
        }
        finally
        {
            if (inputTex != null)
                _unetRepro?.ReturnTempArray(inputTex);
            if (timestepTex != null)
                _unetRepro?.ReturnTempArray(timestepTex);
            if (condOutTex != null)
                _unetRepro?.ReturnTempArray(condOutTex);
            if (uncondOutTex != null)
                _unetRepro?.ReturnTempArray(uncondOutTex);
            if (finalTex != null)
                _unetRepro?.ReturnTempArray(finalTex);
            ReleaseUnetCache(unetCache);
        }
    }

    private async UniTask<RenderTexture> RunCfgUnetPack4CommandBufferAsync(
        RenderTexture latentsTex,
        RenderTexture maskTex,
        RenderTexture maskedLatentTex,
        int timestep,
        RenderTexture condTex,
        RenderTexture uncondTex,
        float guidanceScale,
        CancellationToken ct)
    {
        if (condTex == null || uncondTex == null)
            throw new InvalidOperationException("UNet command-buffer path requires prompt condition textures.");

        RenderTexture inputTex = null;
        RenderTexture timestepTex = null;
        RenderTexture outputReadbackRt = null;
        ComputeTexture condOutCmd = null;
        ComputeTexture uncondOutCmd = null;
        ComputeTexture finalCmd = null;
        try
        {
            inputTex = BuildUnetInputPack4(latentsTex, maskTex, maskedLatentTex);
            timestepTex = CreateTensorPack4Texture(_unetRepro, new[] { (float)timestep }, 1, 1, 1, 1, 1, tensorTextureFormat);

            using var cmd = new CommandBuffer { name = "SDInpaintUNetCFG" };
            if (useAsyncComputeCommandBuffer)
                cmd.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);

            var inputCmd = BindExternalTexture(cmd, inputTex, "SDInpaint.Cmd.UnetInput");
            var timestepCmd = BindExternalTexture(cmd, timestepTex, "SDInpaint.Cmd.Timestep");
            var condCmd = BindExternalTexture(cmd, condTex, "SDInpaint.Cmd.Cond");
            var uncondCmd = BindExternalTexture(cmd, uncondTex, "SDInpaint.Cmd.Uncond");

            var baseShapes = new Dictionary<string, NcnnRepro.BufferShape>(StringComparer.Ordinal)
            {
                { UnetInputBlobName, new NcnnRepro.BufferShape(3, LatentSize, LatentSize, 1, UnetInputChannels) },
                { UnetTimestepBlobName, new NcnnRepro.BufferShape(1, 1, 1, 1, 1) }
            };

            var condInputs = new Dictionary<string, ComputeTexture>(StringComparer.Ordinal)
            {
                { UnetInputBlobName, inputCmd },
                { UnetTimestepBlobName, timestepCmd },
                { UnetTextBlobName, condCmd }
            };
            var condShapes = new Dictionary<string, NcnnRepro.BufferShape>(baseShapes, StringComparer.Ordinal)
            {
                { UnetTextBlobName, new NcnnRepro.BufferShape(2, TextEmbeddingWidth, TokenCount, 1, 1) }
            };
            condOutCmd = _unetRepro.ForwardPack4(
                cmd,
                condInputs,
                condShapes,
                out _,
                new HashSet<string>(StringComparer.Ordinal) { UnetOutputBlobName },
                UnetOutputBlobName);

            var uncondInputs = new Dictionary<string, ComputeTexture>(StringComparer.Ordinal)
            {
                { UnetInputBlobName, inputCmd },
                { UnetTimestepBlobName, timestepCmd },
                { UnetTextBlobName, uncondCmd }
            };
            var uncondShapes = new Dictionary<string, NcnnRepro.BufferShape>(baseShapes, StringComparer.Ordinal)
            {
                { UnetTextBlobName, new NcnnRepro.BufferShape(2, TextEmbeddingWidth, TokenCount, 1, 1) }
            };
            uncondOutCmd = _unetRepro.ForwardPack4(
                cmd,
                uncondInputs,
                uncondShapes,
                out _,
                new HashSet<string>(StringComparer.Ordinal) { UnetOutputBlobName },
                UnetOutputBlobName);

            if (condOutCmd == null || uncondOutCmd == null)
                throw new InvalidOperationException("UNet command-buffer path produced no CFG outputs.");

            finalCmd = _unetRepro.RentTempArray(cmd, LatentSize, LatentSize, 1, tensorTextureFormat);
            _ops.AddPack4(cmd, condOutCmd, uncondOutCmd, guidanceScale, 1f - guidanceScale, 1, finalCmd);

            outputReadbackRt = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            cmd.CopyTexture(finalCmd.nameID, 0, 0, outputReadbackRt, 0, 0);

            _unetRepro.ReturnTempArray(cmd, condOutCmd);
            condOutCmd = null;
            _unetRepro.ReturnTempArray(cmd, uncondOutCmd);
            uncondOutCmd = null;
            _unetRepro.ReturnTempArray(cmd, finalCmd);
            finalCmd = null;

            if (useAsyncComputeCommandBuffer)
                Graphics.ExecuteCommandBufferAsync(cmd, ComputeQueueType.Default);
            else
                Graphics.ExecuteCommandBuffer(cmd);

            _ops.DebugSyncGpu();
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();

            var result = outputReadbackRt;
            outputReadbackRt = null;
            return result;
        }
        finally
        {
            using var cleanupCmd = new CommandBuffer { name = "SDInpaintUNetCFGCleanup" };
            if (condOutCmd != null)
                _unetRepro?.ReturnTempArray(cleanupCmd, condOutCmd);
            if (uncondOutCmd != null)
                _unetRepro?.ReturnTempArray(cleanupCmd, uncondOutCmd);
            if (finalCmd != null)
                _unetRepro?.ReturnTempArray(cleanupCmd, finalCmd);
            Graphics.ExecuteCommandBuffer(cleanupCmd);
            if (inputTex != null)
                _unetRepro?.ReturnTempArray(inputTex);
            if (timestepTex != null)
                _unetRepro?.ReturnTempArray(timestepTex);
            if (outputReadbackRt != null)
                _unetRepro?.ReturnTempArray(outputReadbackRt);
        }
    }

    private static ComputeTexture BindExternalTexture(CommandBuffer cmd, RenderTexture texture, string label)
    {
        if (cmd == null)
            throw new ArgumentNullException(nameof(cmd));
        if (texture == null)
            throw new ArgumentNullException(nameof(texture));

        var id = Shader.PropertyToID((label ?? "SDInpaint.ExternalTexture") + "." + Guid.NewGuid().ToString("N"));
        cmd.SetGlobalTexture(id, texture);
        return new ComputeTexture
        {
            nameID = id,
            width = texture.width,
            height = texture.height,
            depth = Mathf.Max(1, texture.volumeDepth),
            format = texture.format,
            trackerLabel = label
        };
    }

    private ComputeBuffer RunUnetOnce(
        RenderTexture inputTex,
        NcnnTensorBuffer timestepView,
        NcnnTensorBuffer textView,
        string dumpTag,
        List<UnetCacheBlob> captureCache,
        IReadOnlyList<UnetCacheBlob> reuseCache)
    {
        HashSet<string> pinned = null;
        var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
        {
            { UnetInputBlobName, inputTex }
        };
        var textureInputShapes = new Dictionary<string, NcnnRepro.BufferShape>(StringComparer.Ordinal)
        {
            { UnetInputBlobName, new NcnnRepro.BufferShape(3, LatentSize, LatentSize, 1, UnetInputChannels) }
        };
        if (captureCache != null)
        {
            pinned = new HashSet<string>(OfficialUnetCacheBlobNames, StringComparer.Ordinal)
            {
                UnetOutputBlobName
            };
        }

        if (reuseCache != null)
        {
            for (var i = 0; i < reuseCache.Count; i++)
            {
                var blob = reuseCache[i];
                if (blob == null || blob.texture == null || string.IsNullOrEmpty(blob.name))
                    continue;
                textureInputs[blob.name] = blob.texture;
                textureInputShapes[blob.name] = blob.shape;
            }
        }

        using var infer = _unetRepro.InferWithMultiInputs(
            textureInputs,
            new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal)
            {
                { UnetTimestepBlobName, timestepView },
                { UnetTextBlobName, textView }
            },
            pinned ?? new HashSet<string>(StringComparer.Ordinal) { UnetOutputBlobName },
            textureInputShapes);

        if (captureCache != null)
            TryCaptureOfficialUnetCache(infer, captureCache);

        var outTex = infer.ExtractTexture(UnetOutputBlobName);
        if (outTex == null)
            return null;

        try
        {
            var outBuf = _unetRepro.RentTempBuffer(LatentElementCount(), sizeof(float));
            _ops.Pack4ToBufferCHW(outTex, LatentSize, LatentSize, LatentChannels, outBuf);
            return outBuf;
        }
        finally
        {
            _unetRepro.ReturnTempArray(outTex);
        }
    }

    private RenderTexture RunUnetOnceTexture(
        RenderTexture inputTex,
        RenderTexture timestepTex,
        RenderTexture textTex,
        string dumpTag,
        List<UnetCacheBlob> captureCache,
        IReadOnlyList<UnetCacheBlob> reuseCache)
    {
        if (inputTex == null)
            throw new ArgumentNullException(nameof(inputTex));
        if (timestepTex == null)
            throw new ArgumentNullException(nameof(timestepTex));
        if (textTex == null)
            throw new ArgumentNullException(nameof(textTex));

        HashSet<string> pinned = null;
        var shouldDumpCond = enableDebugDump
            && !_unetDebugDumped
            && !string.IsNullOrWhiteSpace(_lastDumpDir)
            && string.Equals(dumpTag, "cond", StringComparison.Ordinal);
        var debugUnetBlobNames = shouldDumpCond ? ResolveDebugUnetBlobNames() : Array.Empty<string>();
        var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
        {
            { UnetInputBlobName, inputTex },
            { UnetTimestepBlobName, timestepTex },
            { UnetTextBlobName, textTex }
        };
        var textureInputShapes = new Dictionary<string, NcnnRepro.BufferShape>(StringComparer.Ordinal)
        {
            { UnetInputBlobName, new NcnnRepro.BufferShape(3, LatentSize, LatentSize, 1, UnetInputChannels) },
            { UnetTimestepBlobName, new NcnnRepro.BufferShape(1, 1, 1, 1, 1) },
            { UnetTextBlobName, new NcnnRepro.BufferShape(2, TextEmbeddingWidth, TokenCount, 1, 1) }
        };
        if (captureCache != null)
        {
            pinned = new HashSet<string>(OfficialUnetCacheBlobNames, StringComparer.Ordinal)
            {
                UnetOutputBlobName
            };
        }

        if (shouldDumpCond)
        {
            pinned ??= new HashSet<string>(StringComparer.Ordinal);
            for (var i = 0; i < debugUnetBlobNames.Length; i++)
                pinned.Add(debugUnetBlobNames[i]);
            pinned.Add(UnetOutputBlobName);
        }

        if (reuseCache != null)
        {
            for (var i = 0; i < reuseCache.Count; i++)
            {
                var blob = reuseCache[i];
                if (blob == null || blob.texture == null || string.IsNullOrEmpty(blob.name))
                    continue;
                textureInputs[blob.name] = blob.texture;
                textureInputShapes[blob.name] = blob.shape;
            }
        }

        using var infer = _unetRepro.InferWithMultiInputs(
            textureInputs,
            null,
            pinned ?? new HashSet<string>(StringComparer.Ordinal) { UnetOutputBlobName },
            textureInputShapes);

        if (captureCache != null)
            TryCaptureOfficialUnetCache(infer, captureCache);

        if (shouldDumpCond)
        {
            for (var i = 0; i < debugUnetBlobNames.Length; i++)
                TryDumpAnyBlob(infer, debugUnetBlobNames[i], Path.Combine(_lastDumpDir, "unity_unet_blob_" + debugUnetBlobNames[i] + ".txt"));
        }

        return infer.ExtractTexture(UnetOutputBlobName);
    }

    private bool TryCaptureOfficialUnetCache(NcnnRepro.InferResult infer, List<UnetCacheBlob> dst)
    {
        if (infer == null || dst == null)
            return false;

        dst.Clear();
        var captured = new List<UnetCacheBlob>(OfficialUnetCacheBlobNames.Length);
        try
        {
            for (var i = 0; i < OfficialUnetCacheBlobNames.Length; i++)
            {
                var name = OfficialUnetCacheBlobNames[i];
                if (!infer.TryGetLogicalShape(name, out var dims, out var w, out var h, out var d, out var c))
                    throw new InvalidOperationException("UNet cache blob shape not found: " + name);

                var tex = infer.ExtractTexture(name);
                if (tex == null)
                    throw new InvalidOperationException("UNet cache blob texture not found: " + name);

                captured.Add(new UnetCacheBlob
                {
                    name = name,
                    texture = tex,
                    shape = new NcnnRepro.BufferShape(dims, w, h, d, c)
                });
            }

            dst.AddRange(captured);
            return true;
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[SDInpaint] official UNet cache disabled for this step: " + e.Message);
            ReleaseUnetCache(captured);
            dst.Clear();
            return false;
        }
    }

    private RenderTexture BuildLatentMaskPack4(ComputeBuffer latentMaskBuf)
    {
        if (latentMaskBuf == null)
            return null;

        var maskTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
        _ops.FillPack4FromBufferCHW(latentMaskBuf, LatentSize, LatentSize, 1, maskTex);
        return maskTex;
    }

    private RenderTexture BuildLatentPack4(ComputeBuffer latentBuf)
    {
        if (latentBuf == null)
            return null;

        var latentTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
        _ops.FillPack4FromBufferCHW(latentBuf, LatentSize, LatentSize, LatentChannels, latentTex);
        return latentTex;
    }

    private RenderTexture BuildNoisyLatentsPack4(ComputeBuffer noiseBuf)
    {
        return BuildLatentPack4(noiseBuf);
    }

    private RenderTexture BuildNoisyLatentsPack4(ComputeBuffer cleanLatentBuf, ComputeBuffer noiseBuf, int timestep)
    {
        if (cleanLatentBuf == null || noiseBuf == null)
            return null;

        var alphaProd = GetAlphaCumprod(timestep);
        var sqrtAlpha = Mathf.Sqrt(alphaProd);
        var sqrtOneMinusAlpha = Mathf.Sqrt(Mathf.Max(0f, 1f - alphaProd));

        RenderTexture cleanTex = null;
        RenderTexture noiseTex = null;
        RenderTexture scaledCleanTex = null;
        RenderTexture scaledNoiseTex = null;
        RenderTexture outputTex = null;
        try
        {
            cleanTex = BuildLatentPack4(cleanLatentBuf);
            noiseTex = BuildLatentPack4(noiseBuf);
            if (cleanTex == null || noiseTex == null)
                return null;

            scaledCleanTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            scaledNoiseTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            outputTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            _ops.BinaryOpScalarPack4(cleanTex, sqrtAlpha, 1, 2, scaledCleanTex);
            _ops.BinaryOpScalarPack4(noiseTex, sqrtOneMinusAlpha, 1, 2, scaledNoiseTex);
            _ops.BinaryOpPack4(scaledCleanTex, scaledNoiseTex, 1, 0, outputTex);

            var result = outputTex;
            outputTex = null;
            return result;
        }
        finally
        {
            if (cleanTex != null)
                _unetRepro?.ReturnTempArray(cleanTex);
            if (noiseTex != null)
                _unetRepro?.ReturnTempArray(noiseTex);
            if (scaledCleanTex != null)
                _unetRepro?.ReturnTempArray(scaledCleanTex);
            if (scaledNoiseTex != null)
                _unetRepro?.ReturnTempArray(scaledNoiseTex);
            if (outputTex != null)
                _unetRepro?.ReturnTempArray(outputTex);
        }
    }

    private RenderTexture BuildNoisyLatentsPack4(RenderTexture cleanLatentTex, RenderTexture noiseTex, int timestep)
    {
        if (cleanLatentTex == null || noiseTex == null)
            return null;

        var alphaProd = GetAlphaCumprod(timestep);
        var sqrtAlpha = Mathf.Sqrt(alphaProd);
        var sqrtOneMinusAlpha = Mathf.Sqrt(Mathf.Max(0f, 1f - alphaProd));

        RenderTexture scaledCleanTex = null;
        RenderTexture scaledNoiseTex = null;
        RenderTexture outputTex = null;
        try
        {
            scaledCleanTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            scaledNoiseTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            outputTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            _ops.BinaryOpScalarPack4(cleanLatentTex, sqrtAlpha, 1, 2, scaledCleanTex);
            _ops.BinaryOpScalarPack4(noiseTex, sqrtOneMinusAlpha, 1, 2, scaledNoiseTex);
            _ops.BinaryOpPack4(scaledCleanTex, scaledNoiseTex, 1, 0, outputTex);

            var result = outputTex;
            outputTex = null;
            return result;
        }
        finally
        {
            if (scaledCleanTex != null)
                _unetRepro?.ReturnTempArray(scaledCleanTex);
            if (scaledNoiseTex != null)
                _unetRepro?.ReturnTempArray(scaledNoiseTex);
            if (outputTex != null)
                _unetRepro?.ReturnTempArray(outputTex);
        }
    }

    private RenderTexture BuildUnetInputPack4(RenderTexture latentsTex, RenderTexture maskTex, RenderTexture maskedLatentTex)
    {
        if (latentsTex == null)
            throw new ArgumentNullException(nameof(latentsTex));
        if (maskTex == null)
            throw new ArgumentNullException(nameof(maskTex));
        if (maskedLatentTex == null)
            throw new ArgumentNullException(nameof(maskedLatentTex));

        RenderTexture inputTex = null;
        try
        {
            inputTex = _unetRepro.RentTempArray(LatentSize, LatentSize, Mathf.CeilToInt(UnetInputChannels / 4f), tensorTextureFormat);
            _ops.BuildSdInpaintInput9Pack4(latentsTex, maskTex, maskedLatentTex, inputTex);

            var result = inputTex;
            inputTex = null;
            return result;
        }
        finally
        {
            if (inputTex != null)
                _unetRepro?.ReturnTempArray(inputTex);
        }
    }

    private RenderTexture DdimStepPack4(RenderTexture sampleTex, RenderTexture epsilonTex, int timestep, int prevTimestep)
    {
        if (sampleTex == null || epsilonTex == null)
            return null;

        var alphaProdT = GetAlphaCumprod(timestep);
        var alphaProdPrev = prevTimestep >= 0 ? GetAlphaCumprod(prevTimestep) : _finalAlphaCumprod;
        var betaProdT = Mathf.Max(0f, 1f - alphaProdT);
        var sqrtAlphaT = Mathf.Sqrt(alphaProdT);
        var sqrtAlphaPrev = Mathf.Sqrt(alphaProdPrev);
        var sqrtBetaT = Mathf.Sqrt(betaProdT);
        var sqrtOneMinusAlphaPrev = Mathf.Sqrt(Mathf.Max(0f, 1f - alphaProdPrev));

        RenderTexture scaledEpsilonTex = null;
        RenderTexture predOriginalNumeratorTex = null;
        RenderTexture predOriginalTex = null;
        RenderTexture predDirectionTex = null;
        RenderTexture scaledOriginalTex = null;
        RenderTexture outputTex = null;
        try
        {
            scaledEpsilonTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            predOriginalNumeratorTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            predOriginalTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            predDirectionTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            scaledOriginalTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);
            outputTex = _unetRepro.RentTempArray(LatentSize, LatentSize, 1, tensorTextureFormat);

            _ops.BinaryOpScalarPack4(epsilonTex, sqrtBetaT, 1, 2, scaledEpsilonTex);
            _ops.BinaryOpPack4(sampleTex, scaledEpsilonTex, 1, 1, predOriginalNumeratorTex);
            _ops.BinaryOpScalarPack4(predOriginalNumeratorTex, Mathf.Max(1e-6f, sqrtAlphaT), 1, 3, predOriginalTex);
            _ops.BinaryOpScalarPack4(epsilonTex, sqrtOneMinusAlphaPrev, 1, 2, predDirectionTex);
            _ops.BinaryOpScalarPack4(predOriginalTex, sqrtAlphaPrev, 1, 2, scaledOriginalTex);
            _ops.BinaryOpPack4(scaledOriginalTex, predDirectionTex, 1, 0, outputTex);

            var result = outputTex;
            outputTex = null;
            return result;
        }
        finally
        {
            if (scaledEpsilonTex != null)
                _unetRepro?.ReturnTempArray(scaledEpsilonTex);
            if (predOriginalNumeratorTex != null)
                _unetRepro?.ReturnTempArray(predOriginalNumeratorTex);
            if (predOriginalTex != null)
                _unetRepro?.ReturnTempArray(predOriginalTex);
            if (predDirectionTex != null)
                _unetRepro?.ReturnTempArray(predDirectionTex);
            if (scaledOriginalTex != null)
                _unetRepro?.ReturnTempArray(scaledOriginalTex);
            if (outputTex != null)
                _unetRepro?.ReturnTempArray(outputTex);
        }
    }

    private ComputeBuffer ExtractLatentPack4ToStandaloneBuffer(RenderTexture latentsTex)
    {
        if (latentsTex == null)
            return null;

        var output = NewTrackedBuffer(LatentElementCount(), sizeof(float), ComputeBufferType.Structured, "SDInpaint.ExtractLatentPack4");
        _ops.Pack4ToBufferCHW(latentsTex, LatentSize, LatentSize, LatentChannels, output);
        return output;
    }

    private void ReleaseUnetCache(IReadOnlyList<UnetCacheBlob> cache)
    {
        if (cache == null)
            return;

        for (var i = 0; i < cache.Count; i++)
        {
            var blob = cache[i];
            if (blob?.texture == null)
                continue;
            try { _unetRepro?.ReturnTempArray(blob.texture); } catch { }
            blob.texture = null;
        }
    }

    private ComputeBuffer AddNoise(ComputeBuffer cleanLatentBuf, ComputeBuffer noiseBuf, int timestep)
    {
        if (cleanLatentBuf == null || noiseBuf == null)
            return null;

        var alphaProd = GetAlphaCumprod(timestep);
        var sqrtAlpha = Mathf.Sqrt(alphaProd);
        var sqrtOneMinusAlpha = Mathf.Sqrt(Mathf.Max(0f, 1f - alphaProd));
        var scaledClean = _unetRepro.RentTempBuffer(cleanLatentBuf.count, sizeof(float));
        var scaledNoise = _unetRepro.RentTempBuffer(cleanLatentBuf.count, sizeof(float));
        var output = _unetRepro.RentTempBuffer(cleanLatentBuf.count, sizeof(float));
        try
        {
            _ops.BinaryOpScalarBuf(cleanLatentBuf, sqrtAlpha, cleanLatentBuf.count, 2, scaledClean);
            _ops.BinaryOpScalarBuf(noiseBuf, sqrtOneMinusAlpha, noiseBuf.count, 2, scaledNoise);
            _ops.BinaryOpBuf(scaledClean, scaledNoise, output.count, 0, output);
            return output;
        }
        finally
        {
            _unetRepro.ReturnTempBuffer(scaledClean);
            _unetRepro.ReturnTempBuffer(scaledNoise);
        }
    }

    private ComputeBuffer DdimStep(ComputeBuffer sampleBuf, ComputeBuffer epsilonBuf, int timestep, int prevTimestep)
    {
        if (sampleBuf == null || epsilonBuf == null)
            return null;

        var alphaProdT = GetAlphaCumprod(timestep);
        var alphaProdPrev = prevTimestep >= 0 ? GetAlphaCumprod(prevTimestep) : _finalAlphaCumprod;
        var betaProdT = Mathf.Max(0f, 1f - alphaProdT);
        var sqrtAlphaT = Mathf.Sqrt(alphaProdT);
        var sqrtAlphaPrev = Mathf.Sqrt(alphaProdPrev);
        var sqrtBetaT = Mathf.Sqrt(betaProdT);
        var sqrtOneMinusAlphaPrev = Mathf.Sqrt(Mathf.Max(0f, 1f - alphaProdPrev));

        var scaledEpsilon = _unetRepro.RentTempBuffer(sampleBuf.count, sizeof(float));
        var predOriginalNumerator = _unetRepro.RentTempBuffer(sampleBuf.count, sizeof(float));
        var predOriginal = _unetRepro.RentTempBuffer(sampleBuf.count, sizeof(float));
        var predDirection = _unetRepro.RentTempBuffer(sampleBuf.count, sizeof(float));
        var scaledOriginal = _unetRepro.RentTempBuffer(sampleBuf.count, sizeof(float));
        var output = _unetRepro.RentTempBuffer(sampleBuf.count, sizeof(float));
        try
        {
            _ops.BinaryOpScalarBuf(epsilonBuf, sqrtBetaT, epsilonBuf.count, 2, scaledEpsilon);
            _ops.BinaryOpBuf(sampleBuf, scaledEpsilon, sampleBuf.count, 1, predOriginalNumerator);
            _ops.BinaryOpScalarBuf(predOriginalNumerator, Mathf.Max(1e-6f, sqrtAlphaT), predOriginalNumerator.count, 3, predOriginal);
            _ops.BinaryOpScalarBuf(epsilonBuf, sqrtOneMinusAlphaPrev, epsilonBuf.count, 2, predDirection);
            _ops.BinaryOpScalarBuf(predOriginal, sqrtAlphaPrev, predOriginal.count, 2, scaledOriginal);
            _ops.BinaryOpBuf(scaledOriginal, predDirection, output.count, 0, output);
            return output;
        }
        finally
        {
            _unetRepro.ReturnTempBuffer(scaledEpsilon);
            _unetRepro.ReturnTempBuffer(predOriginalNumerator);
            _unetRepro.ReturnTempBuffer(predOriginal);
            _unetRepro.ReturnTempBuffer(predDirection);
            _unetRepro.ReturnTempBuffer(scaledOriginal);
        }
    }

    private void EnsureSchedulerTables()
    {
        if (_alphasCumprod != null && _alphasCumprod.Length == TrainTimestepCount)
            return;

        _alphasCumprod = new float[TrainTimestepCount];
        var sqrtBetaStart = Mathf.Sqrt(BetaStart);
        var sqrtBetaEnd = Mathf.Sqrt(BetaEnd);
        var alphaProd = 1f;
        for (var i = 0; i < TrainTimestepCount; i++)
        {
            var t = TrainTimestepCount <= 1 ? 0f : (float)i / (TrainTimestepCount - 1);
            var beta = Mathf.Pow(Mathf.Lerp(sqrtBetaStart, sqrtBetaEnd, t), 2f);
            var alpha = 1f - beta;
            alphaProd *= alpha;
            _alphasCumprod[i] = alphaProd;
        }

        _finalAlphaCumprod = SetAlphaToOne ? 1f : _alphasCumprod[0];
    }

    private int[] BuildTimesteps(int stepCount)
    {
        stepCount = Mathf.Max(1, stepCount);
        var stepRatio = TrainTimestepCount / stepCount;
        var timesteps = new int[stepCount];
        for (var i = 0; i < stepCount; i++)
            timesteps[i] = (stepCount - 1 - i) * stepRatio + StepsOffset;
        return timesteps;
    }

    private static int[] SelectTimestepsByStrength(int[] timesteps, float strength)
    {
        if (timesteps == null || timesteps.Length == 0)
            return Array.Empty<int>();

        strength = Mathf.Clamp01(strength);
        var initTimestep = Mathf.Min(Mathf.FloorToInt(timesteps.Length * strength), timesteps.Length);
        if (strength > 0f && initTimestep < 1)
            initTimestep = 1;
        var start = Mathf.Max(timesteps.Length - initTimestep, 0);
        var count = timesteps.Length - start;
        if (count <= 0)
            return Array.Empty<int>();

        var result = new int[count];
        Array.Copy(timesteps, start, result, 0, count);
        return result;
    }

    private float GetAlphaCumprod(int timestep)
    {
        timestep = Mathf.Clamp(timestep, 0, TrainTimestepCount - 1);
        return _alphasCumprod[timestep];
    }

    private static int ResolveSeed(int seed)
    {
        return seed != 0 ? seed : (Environment.TickCount & int.MaxValue);
    }

    private static int LatentElementCount()
    {
        return LatentSize * LatentSize * LatentChannels;
    }

    private static ComputeBuffer NewTrackedBuffer(int count, int stride, ComputeBufferType type, string label)
    {
        if (count <= 0 || stride <= 0)
            return null;
        var buffer = new ComputeBuffer(count, stride, type);
        NcnnGpuResourceTracker.RegisterBuffer(buffer, count, stride, label ?? "SDInpaint.StandaloneBuffer");
        return buffer;
    }

    private static ComputeBuffer NewFloatBuffer(float[] data, string label = null)
    {
        if (data == null)
            return null;
        var buffer = NewTrackedBuffer(data.Length, sizeof(float), ComputeBufferType.Structured, label ?? "SDInpaint.FloatBuffer");
        buffer.SetData(data);
        return buffer;
    }

    private ComputeBuffer CloneBufferStandalone(ComputeBuffer source, string label = null)
    {
        if (source == null)
            return null;

        var clone = NewTrackedBuffer(source.count, sizeof(float), ComputeBufferType.Structured, label ?? "SDInpaint.CloneBuffer");
        _ops.CopyBuf(source, clone, source.count);
        return clone;
    }

    private INormalRng CreateLatentNoiseRng(int seed)
    {
        return useOfficialNoise
            ? new TorchCpuNormalRng(unchecked((uint)seed))
            : new FallbackNormalRng(seed);
    }

    private static float[] GenerateGaussian(int count, INormalRng rng)
    {
        if (rng == null)
            throw new ArgumentNullException(nameof(rng));

        var data = new float[Mathf.Max(0, count)];
        if (rng is TorchCpuNormalRng torchRng && data.Length >= 16)
        {
            FillTorchCpuGaussian(data, torchRng);
            return data;
        }

        for (var i = 0; i < data.Length; i++)
            data[i] = rng.NextNormal();
        return data;
    }

    private static void FillTorchCpuGaussian(float[] data, TorchCpuNormalRng rng)
    {
        if (data == null || data.Length == 0 || rng == null)
            return;

        var block = new float[16];
        var full = data.Length - 15;
        var offset = 0;
        while (offset < full)
        {
            FillTorchCpuGaussianBlock(block, rng);
            Array.Copy(block, 0, data, offset, 16);
            offset += 16;
        }

        if ((data.Length % 16) != 0)
        {
            FillTorchCpuGaussianBlock(block, rng);
            Array.Copy(block, 0, data, data.Length - 16, 16);
        }
    }

    private static void FillTorchCpuGaussianBlock(float[] block, TorchCpuNormalRng rng)
    {
        if (block == null || block.Length < 16 || rng == null)
            return;

        for (var i = 0; i < 16; i++)
            block[i] = rng.NextUniform();

        for (var j = 0; j < 8; j++)
        {
            var u1 = 1f - block[j];
            var u2 = block[j + 8];
            var radius = Mathf.Sqrt(-2f * Mathf.Log(Mathf.Max(u1, 1e-12f)));
            var theta = 2f * Mathf.PI * u2;
            block[j] = radius * Mathf.Cos(theta);
            block[j + 8] = radius * Mathf.Sin(theta);
        }
    }

    private static int NormalizeSeed(int seed)
    {
        var mod = seed % 1000;
        if (mod < 0)
            mod += 1000;
        return mod;
    }

    private static float[] GenerateOfficialGaussian(int seed, int count)
    {
        var data = new float[Mathf.Max(0, count)];
        EnsureOpenCvNormalTables();

        var state = seed != 0 ? (ulong)(uint)seed : 0xffffffffUL;
        const float r = 3.442620f;
        const float rngFloat = 2.3283064365386962890625e-10f;
        const float floatMin = 1.17549435e-38f;

        for (var i = 0; i < data.Length; i++)
        {
            float x;
            for (;;)
            {
                var hz = unchecked((int)(uint)state);
                state = OpenCvRngNext(state);
                var iz = hz & 127;
                x = hz * OpenCvNormalWn[iz];

                var absHz = hz < 0 ? unchecked((uint)-hz) : (uint)hz;
                if (absHz < OpenCvNormalKn[iz])
                    break;

                if (iz == 0)
                {
                    float y;
                    do
                    {
                        x = (uint)state * rngFloat;
                        state = OpenCvRngNext(state);
                        y = (uint)state * rngFloat;
                        state = OpenCvRngNext(state);
                        x = (float)(-Math.Log(x + floatMin) * 0.2904764);
                        y = (float)-Math.Log(y + floatMin);
                    }
                    while (y + y < x * x);

                    x = hz > 0 ? r + x : -r - x;
                    break;
                }

                var wedgeY = (uint)state * rngFloat;
                state = OpenCvRngNext(state);
                if (OpenCvNormalFn[iz] + wedgeY * (OpenCvNormalFn[iz - 1] - OpenCvNormalFn[iz]) < (float)Math.Exp(-0.5f * x * x))
                    break;
            }

            data[i] = x;
        }

        return data;
    }

    private static void EnsureOpenCvNormalTables()
    {
        if (_openCvNormalTablesReady)
            return;

        const double m1 = 2147483648.0;
        var dn = 3.442619855899;
        var tn = dn;
        const double vn = 9.91256303526217e-3;

        var q = vn / Math.Exp(-0.5 * dn * dn);
        OpenCvNormalKn[0] = (uint)((dn / q) * m1);
        OpenCvNormalKn[1] = 0;
        OpenCvNormalWn[0] = (float)(q / m1);
        OpenCvNormalWn[127] = (float)(dn / m1);
        OpenCvNormalFn[0] = 1f;
        OpenCvNormalFn[127] = (float)Math.Exp(-0.5 * dn * dn);

        for (var i = 126; i >= 1; i--)
        {
            dn = Math.Sqrt(-2.0 * Math.Log(vn / dn + Math.Exp(-0.5 * dn * dn)));
            OpenCvNormalKn[i + 1] = (uint)((dn / tn) * m1);
            tn = dn;
            OpenCvNormalFn[i] = (float)Math.Exp(-0.5 * dn * dn);
            OpenCvNormalWn[i] = (float)(dn / m1);
        }

        _openCvNormalTablesReady = true;
    }

    private static ulong OpenCvRngNext(ulong state)
    {
        return unchecked((ulong)(uint)state * 4164903690UL + (state >> 32));
    }

    private static ComputeBuffer CreateMaskBuffer(Texture2D mask)
    {
        if (mask == null)
            return null;

        var width = mask.width;
        var height = mask.height;
        var pixels = mask.GetPixels32();
        var data = new float[pixels.Length];
        for (var y = 0; y < height; y++)
        {
            // Texture2D pixel arrays are bottom-up in Unity, while our latent CHW buffers
            // follow the same top-down convention as the encoder input path.
            var srcRow = (height - 1 - y) * width;
            var dstRow = y * width;
            for (var x = 0; x < width; x++)
                data[dstRow + x] = SampleMaskWeight(pixels[srcRow + x]);
        }

        return NewFloatBuffer(data, "SDInpaint.MaskBuffer");
    }

    private static Texture2D EnsureReadableTexture(Texture src)
    {
        if (src == null)
            return null;
        if (src is Texture2D tex)
        {
            try
            {
                var _ = tex.GetPixel(0, 0);
                return tex;
            }
            catch
            {
            }
        }
        return ReadResizedTexture(src, src.width, src.height);
    }

    private static Texture2D BuildMaskedTexture(Texture2D source, Texture2D mask)
    {
        if (source == null || mask == null || source.width != mask.width || source.height != mask.height)
            return null;

        var srcPixels = source.GetPixels32();
        var maskPixels = mask.GetPixels32();
        var pixels = new Color32[srcPixels.Length];
        for (var i = 0; i < srcPixels.Length; i++)
        {
            var alpha = SampleMaskWeight(maskPixels[i]);
            var inv = 1f - alpha;
            var s = srcPixels[i];
            // Diffusers inpainting zeros the masked area after image normalization to [-1, 1].
            // In 8-bit RGB space that "zero" corresponds to mid-gray rather than black.
            var maskedBase = 127.5f * alpha;
            pixels[i] = new Color32(
                (byte)Mathf.Clamp(Mathf.RoundToInt(s.r * inv + maskedBase), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(s.g * inv + maskedBase), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(s.b * inv + maskedBase), 0, 255),
                255);
        }

        var texture = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, false);
        texture.SetPixels32(pixels);
        texture.Apply(false, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Point;
        return texture;
    }

    private static Texture2D NormalizeInpaintMask(Texture2D sourceMask, bool blackMaskMeansInpaint)
    {
        if (sourceMask == null)
            return null;

        var sourcePixels = sourceMask.GetPixels32();
        var outputPixels = new Color32[sourcePixels.Length];
        for (var i = 0; i < sourcePixels.Length; i++)
        {
            var weight = SampleMaskWeight(sourcePixels[i]);
            if (blackMaskMeansInpaint)
                weight = 1f - weight;
            weight = weight >= 0.5f ? 1f : 0f;

            var value = (byte)Mathf.Clamp(Mathf.RoundToInt(weight * 255f), 0, 255);
            outputPixels[i] = new Color32(value, value, value, 255);
        }

        var texture = new Texture2D(sourceMask.width, sourceMask.height, TextureFormat.RGBA32, false, false);
        texture.SetPixels32(outputPixels);
        texture.Apply(false, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Point;
        return texture;
    }

    private static Texture2D CompositeWithMask(Texture2D source, Texture2D generated, Texture2D mask)
    {
        if (source == null || generated == null || mask == null)
            return null;
        if (source.width != generated.width || source.height != generated.height || source.width != mask.width || source.height != mask.height)
            return null;

        var srcPixels = source.GetPixels32();
        var genPixels = generated.GetPixels32();
        var maskPixels = mask.GetPixels32();
        var outPixels = new Color32[srcPixels.Length];
        for (var i = 0; i < outPixels.Length; i++)
        {
            var alpha = SampleMaskWeight(maskPixels[i]);
            var inv = 1f - alpha;
            var s = srcPixels[i];
            var g = genPixels[i];
            outPixels[i] = new Color32(
                (byte)Mathf.Clamp(Mathf.RoundToInt(s.r * inv + g.r * alpha), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(s.g * inv + g.g * alpha), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(s.b * inv + g.b * alpha), 0, 255),
                255);
        }

        var result = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, false);
        result.SetPixels32(outPixels);
        result.Apply(false, false);
        result.wrapMode = TextureWrapMode.Clamp;
        result.filterMode = FilterMode.Bilinear;
        return result;
    }

    private static float SampleMaskWeight(Color32 pixel)
    {
        var luminance = ((pixel.r + pixel.g + pixel.b) / 3f) / 255f;
        var alpha = pixel.a / 255f;
        return Mathf.Clamp01(luminance * alpha);
    }

    private static ComputeBuffer CreateEncoderInputBufferNcnn(Texture src, int width, int height)
    {
        if (src == null)
            throw new ArgumentNullException(nameof(src));

        Texture2D tempTexture = null;
        try
        {
            var pixels = ReadTexturePixels32(src, out var srcW, out var srcH, out tempTexture);
            var input = CreateEncoderInputChwNcnn(pixels, srcW, srcH, width, height);
            return NewFloatBuffer(input, "SDInpaint.EncoderInput");
        }
        finally
        {
            if (tempTexture != null)
                DestroyImmediate(tempTexture);
        }
    }

    private static Color32[] ReadTexturePixels32(Texture src, out int width, out int height, out Texture2D tempTexture)
    {
        tempTexture = null;
        width = src != null ? src.width : 0;
        height = src != null ? src.height : 0;
        if (src == null || width <= 0 || height <= 0)
            return Array.Empty<Color32>();

        if (src is Texture2D tex)
        {
            try
            {
                return tex.GetPixels32();
            }
            catch
            {
            }
        }

        var rt = GetTemporaryRt(width, height, RenderTextureFormat.ARGB32, false, "SDInpaint.ReadTexturePixelsRt");
        try
        {
            Graphics.Blit(src, rt);
            tempTexture = RenderTextureToTexture2D(rt, width, height);
            return tempTexture != null ? tempTexture.GetPixels32() : Array.Empty<Color32>();
        }
        finally
        {
            ReleaseTemporaryRt(rt, "SDInpaint.ReadTexturePixelsRt");
        }
    }

    private static float[] CreateEncoderInputChwNcnn(Color32[] pixelsBottomUp, int srcW, int srcH, int dstW, int dstH)
    {
        if (pixelsBottomUp == null || pixelsBottomUp.Length < srcW * srcH)
            throw new InvalidDataException("Source texture pixels are unavailable.");

        var srcBgr = new byte[srcW * srcH * 3];
        for (var y = 0; y < srcH; y++)
        {
            var srcRow = (srcH - 1 - y) * srcW;
            var dstRow = y * srcW * 3;
            for (var x = 0; x < srcW; x++)
            {
                var p = pixelsBottomUp[srcRow + x];
                var di = dstRow + x * 3;
                srcBgr[di + 0] = p.b;
                srcBgr[di + 1] = p.g;
                srcBgr[di + 2] = p.r;
            }
        }

        var dstBgr = new byte[dstW * dstH * 3];
        ResizeBilinearC3Ncnn(srcBgr, srcW, srcH, srcW * 3, dstBgr, dstW, dstH, dstW * 3);

        var wh = dstW * dstH;
        var chw = new float[wh * 3];
        const float norm = 1f / 127.5f;
        for (var i = 0; i < wh; i++)
        {
            var bi = i * 3;
            chw[i] = (dstBgr[bi + 2] - 127.5f) * norm;
            chw[wh + i] = (dstBgr[bi + 1] - 127.5f) * norm;
            chw[wh * 2 + i] = (dstBgr[bi + 0] - 127.5f) * norm;
        }

        return chw;
    }

    private static void ResizeBilinearC3Ncnn(byte[] src, int srcW, int srcH, int srcStride, byte[] dst, int dstW, int dstH, int dstStride)
    {
        if (srcW == dstW && srcH == dstH)
        {
            for (var y = 0; y < dstH; y++)
                Buffer.BlockCopy(src, y * srcStride, dst, y * dstStride, dstW * 3);
            return;
        }

        if (srcW < 2 || srcH < 2)
        {
            ResizeNearestC3(src, srcW, srcH, srcStride, dst, dstW, dstH, dstStride);
            return;
        }

        const int coefBits = 11;
        const int coefScale = 1 << coefBits;
        var scaleX = (double)srcW / dstW;
        var scaleY = (double)srcH / dstH;
        var xofs = new int[dstW];
        var yofs = new int[dstH];
        var ialpha = new short[dstW * 2];
        var ibeta = new short[dstH * 2];

        for (var dx = 0; dx < dstW; dx++)
        {
            var fx = (float)((dx + 0.5) * scaleX - 0.5);
            var sx = (int)Math.Floor(fx);
            fx -= sx;
            if (sx < 0)
            {
                sx = 0;
                fx = 0f;
            }
            if (sx >= srcW - 1)
            {
                sx = srcW - 2;
                fx = 1f;
            }

            xofs[dx] = sx * 3;
            ialpha[dx * 2 + 0] = SaturateCastShort((1f - fx) * coefScale);
            ialpha[dx * 2 + 1] = SaturateCastShort(fx * coefScale);
        }

        for (var dy = 0; dy < dstH; dy++)
        {
            var fy = (float)((dy + 0.5) * scaleY - 0.5);
            var sy = (int)Math.Floor(fy);
            fy -= sy;
            if (sy < 0)
            {
                sy = 0;
                fy = 0f;
            }
            if (sy >= srcH - 1)
            {
                sy = srcH - 2;
                fy = 1f;
            }

            yofs[dy] = sy;
            ibeta[dy * 2 + 0] = SaturateCastShort((1f - fy) * coefScale);
            ibeta[dy * 2 + 1] = SaturateCastShort(fy * coefScale);
        }

        for (var dy = 0; dy < dstH; dy++)
        {
            var sy = yofs[dy];
            var s0Base = sy * srcStride;
            var s1Base = (sy + 1) * srcStride;
            var dBase = dy * dstStride;
            var b0 = ibeta[dy * 2 + 0];
            var b1 = ibeta[dy * 2 + 1];

            for (var dx = 0; dx < dstW; dx++)
            {
                var sx = xofs[dx];
                var a0 = ialpha[dx * 2 + 0];
                var a1 = ialpha[dx * 2 + 1];
                var s0 = s0Base + sx;
                var s1 = s1Base + sx;
                var di = dBase + dx * 3;

                for (var c = 0; c < 3; c++)
                {
                    var row0 = (src[s0 + c] * a0 + src[s0 + c + 3] * a1) >> 4;
                    var row1 = (src[s1 + c] * a0 + src[s1 + c + 3] * a1) >> 4;
                    var value = (((b0 * row0) >> 16) + ((b1 * row1) >> 16) + 2) >> 2;
                    dst[di + c] = (byte)Mathf.Clamp(value, 0, 255);
                }
            }
        }
    }

    private static void ResizeNearestC3(byte[] src, int srcW, int srcH, int srcStride, byte[] dst, int dstW, int dstH, int dstStride)
    {
        for (var y = 0; y < dstH; y++)
        {
            var sy = Mathf.Clamp((int)((long)y * srcH / dstH), 0, srcH - 1);
            for (var x = 0; x < dstW; x++)
            {
                var sx = Mathf.Clamp((int)((long)x * srcW / dstW), 0, srcW - 1);
                Buffer.BlockCopy(src, sy * srcStride + sx * 3, dst, y * dstStride + x * 3, 3);
            }
        }
    }

    private static short SaturateCastShort(float value)
    {
        var rounded = (int)(value + (value >= 0f ? 0.5f : -0.5f));
        return (short)Mathf.Clamp(rounded, short.MinValue, short.MaxValue);
    }

    private static Texture2D ReadResizedTexture(Texture src, int width, int height)
    {
        if (src == null)
            return null;
        var rt = ResizeTextureBilinear(src, width, height);
        if (rt == null)
            return null;
        try
        {
            return RenderTextureToTexture2D(rt, width, height);
        }
        finally
        {
            ReleaseTemporaryRt(rt, "SDInpaint.ResizeTextureRt");
        }
    }

    private static Texture2D ReadResizedTextureNearest(Texture2D src, int width, int height)
    {
        if (src == null)
            return null;
        if (src.width == width && src.height == height)
            return src;

        var srcPixels = src.GetPixels32();
        var dstPixels = new Color32[Mathf.Max(0, width * height)];
        for (var y = 0; y < height; y++)
        {
            var sy = Mathf.Clamp((int)((long)y * src.height / height), 0, src.height - 1);
            var srcRow = sy * src.width;
            var dstRow = y * width;
            for (var x = 0; x < width; x++)
            {
                var sx = Mathf.Clamp((int)((long)x * src.width / width), 0, src.width - 1);
                dstPixels[dstRow + x] = srcPixels[srcRow + sx];
            }
        }

        var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
        texture.SetPixels32(dstPixels);
        texture.Apply(false, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Point;
        return texture;
    }

    private static RenderTexture ResizeTextureBilinear(Texture src, int width, int height)
    {
        if (src == null)
            return null;
        var rt = GetTemporaryRt(width, height, RenderTextureFormat.ARGB32, false, "SDInpaint.ResizeTextureRt");
        Graphics.Blit(src, rt);
        return rt;
    }

    private static Texture2D RenderTextureToTexture2D(RenderTexture rt, int width, int height)
    {
        var previous = RenderTexture.active;
        try
        {
            RenderTexture.active = rt;
            var tex = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            tex.Apply(false, false);
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            return tex;
        }
        finally
        {
            RenderTexture.active = previous;
        }
    }

    private static RenderTexture GetTemporaryRt(int width, int height, RenderTextureFormat format, bool enableRandomWrite, string label = null)
    {
        var desc = new RenderTextureDescriptor(width, height, format, 0)
        {
            enableRandomWrite = enableRandomWrite,
            msaaSamples = 1,
            useMipMap = false,
            autoGenerateMips = false,
            volumeDepth = 1
        };
        var rt = RenderTexture.GetTemporary(desc);
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.filterMode = FilterMode.Bilinear;
        NcnnGpuResourceTracker.RegisterTexture(rt, label ?? "SDInpaint.TempRt");
        return rt;
    }

    private static void ReleaseTemporaryRt(RenderTexture rt, string label = null)
    {
        if (rt == null)
            return;
        NcnnGpuResourceTracker.ReleaseTexture(rt, label ?? "SDInpaint.TempRt");
        try { RenderTexture.ReleaseTemporary(rt); } catch { }
    }

    private static void DisposeBuffer(ComputeBuffer buffer, string label = null)
    {
        if (buffer == null)
            return;
        NcnnGpuResourceTracker.ReleaseBuffer(buffer, label ?? "SDInpaint.StandaloneBuffer");
        try { buffer.Dispose(); } catch { }
    }

    private void ReportProgress(float value01, string message)
    {
        try
        {
            ProgressChanged?.Invoke(Mathf.Clamp01(value01), message ?? string.Empty);
        }
        catch
        {
        }
    }

    private void LogStageTrace(string message)
    {
        if (!ResolveBoolEnv(StageTraceEnvVar, false))
            return;

        try
        {
            UnityEngine.Debug.Log("[SDInpaint] Stage | " + (message ?? string.Empty));
        }
        catch
        {
        }
    }

    private static string CreateDumpDir(string prefix)
    {
        var root = Path.Combine(Path.GetTempPath(), "YanQi", "AIImage");
        Directory.CreateDirectory(root);
        var dir = Path.Combine(root, prefix + "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(dir);
        return dir;
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
        catch
        {
        }
    }

    private static void WriteAllTextSafe(string path, string text)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? string.Empty);
            File.WriteAllText(path, text ?? string.Empty);
        }
        catch
        {
        }
    }

    private static void WriteBufferStatsSafe(string path, ComputeBuffer buffer)
    {
        if (string.IsNullOrWhiteSpace(path) || buffer == null || buffer.count <= 0)
            return;

        try
        {
            var data = new float[buffer.count];
            buffer.GetData(data);
            WriteAllTextSafe(path, BuildFloatStatsReport(data));
        }
        catch
        {
        }
    }

    private static void WriteBufferRawF32Safe(string path, ComputeBuffer buffer)
    {
        if (string.IsNullOrWhiteSpace(path) || buffer == null || buffer.count <= 0)
            return;

        try
        {
            var data = new float[buffer.count];
            buffer.GetData(data);
            var bytes = new byte[data.Length * sizeof(float)];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? string.Empty);
            File.WriteAllBytes(path, bytes);
        }
        catch
        {
        }
    }

    private static void WriteFloatArrayStatsSafe(string path, float[] data)
    {
        if (string.IsNullOrWhiteSpace(path) || data == null || data.Length <= 0)
            return;

        try
        {
            WriteAllTextSafe(path, BuildFloatStatsReport(data));
        }
        catch
        {
        }
    }

    private static void WriteFloatArrayRawF32Safe(string path, float[] data)
    {
        if (string.IsNullOrWhiteSpace(path) || data == null || data.Length <= 0)
            return;

        try
        {
            var bytes = new byte[data.Length * sizeof(float)];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? string.Empty);
            File.WriteAllBytes(path, bytes);
        }
        catch
        {
        }
    }

    private void DumpTextureRawF32Safe(string path, RenderTexture texture, int channelCount)
    {
        if (string.IsNullOrWhiteSpace(path) || texture == null || channelCount <= 0 || _unetRepro == null)
            return;

        ComputeBuffer temp = null;
        try
        {
            temp = _unetRepro.RentTempBuffer(texture.width * texture.height * channelCount, sizeof(float));
            _ops.Pack4ToBufferCHW(texture, texture.width, texture.height, channelCount, temp);
            WriteBufferRawF32Safe(path, temp);
        }
        catch
        {
        }
        finally
        {
            if (temp != null)
                _unetRepro.ReturnTempBuffer(temp);
        }
    }

    private void DumpLatentTextureStatsSafe(string path, RenderTexture texture, int channelCount)
    {
        if (string.IsNullOrWhiteSpace(path) || texture == null || channelCount <= 0 || _unetRepro == null)
            return;

        ComputeBuffer temp = null;
        try
        {
            temp = _unetRepro.RentTempBuffer(texture.width * texture.height * channelCount, sizeof(float));
            _ops.Pack4ToBufferCHW(texture, texture.width, texture.height, channelCount, temp);
            WriteBufferStatsSafe(path, temp);
        }
        catch
        {
        }
        finally
        {
            if (temp != null)
                _unetRepro.ReturnTempBuffer(temp);
        }
    }

    private static string BuildFloatStatsReport(float[] data)
    {
        if (data == null || data.Length == 0)
            return "count=0";

        var min = float.PositiveInfinity;
        var max = float.NegativeInfinity;
        double sum = 0d;
        double sumAbs = 0d;
        var finite = 0;
        var nan = 0;
        var inf = 0;
        for (var i = 0; i < data.Length; i++)
        {
            var v = data[i];
            if (float.IsNaN(v))
            {
                nan++;
                continue;
            }

            if (float.IsInfinity(v))
            {
                inf++;
                continue;
            }

            finite++;
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumAbs += Math.Abs(v);
        }

        var mean = finite > 0 ? sum / finite : 0d;
        var meanAbs = finite > 0 ? sumAbs / finite : 0d;
        return string.Join(
            Environment.NewLine,
            "count=" + data.Length.ToString(CultureInfo.InvariantCulture),
            "finite=" + finite.ToString(CultureInfo.InvariantCulture),
            "nan=" + nan.ToString(CultureInfo.InvariantCulture),
            "inf=" + inf.ToString(CultureInfo.InvariantCulture),
            "min=" + (finite > 0 ? min.ToString("R", CultureInfo.InvariantCulture) : "n/a"),
            "max=" + (finite > 0 ? max.ToString("R", CultureInfo.InvariantCulture) : "n/a"),
            "mean=" + mean.ToString("R", CultureInfo.InvariantCulture),
            "mean_abs=" + meanAbs.ToString("R", CultureInfo.InvariantCulture));
    }

    private static string[] ResolveDebugUnetBlobNames()
    {
        try
        {
            var listEnv = Environment.GetEnvironmentVariable("AIIMAGE_SD_UNET_BLOBS");
            if (!string.IsNullOrWhiteSpace(listEnv))
                return ParseDebugBlobList(listEnv, UnetOutputBlobName);
        }
        catch
        {
        }

        return DebugUnetBlobNames;
    }

    private static string[] ParseDebugBlobList(string value, params string[] excludedNames)
    {
        if (string.IsNullOrWhiteSpace(value))
            return Array.Empty<string>();

        var parts = value.Split(new[] { ',', ';', ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var result = new List<string>(parts.Length);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var excluded = excludedNames != null && excludedNames.Length > 0
            ? new HashSet<string>(excludedNames, StringComparer.Ordinal)
            : null;
        for (var i = 0; i < parts.Length; i++)
        {
            var name = parts[i].Trim();
            if (string.IsNullOrEmpty(name))
                continue;
            if (excluded != null && excluded.Contains(name))
                continue;
            if (seen.Add(name))
                result.Add(name);
        }

        return result.Count > 0 ? result.ToArray() : Array.Empty<string>();
    }

    private void TryDumpAnyBlob(NcnnRepro.InferResult infer, string blobName, string path)
    {
        if (infer == null || string.IsNullOrWhiteSpace(blobName) || string.IsNullOrWhiteSpace(path))
            return;

        if (TryDumpTextureBlobByLogicalShape(infer, blobName, path))
            return;

        try
        {
            var data = infer.GetBufferData(blobName);
            WriteFloatArraySafe(path, data);
            return;
        }
        catch
        {
        }

        try
        {
            if (!infer.TryGetLogicalShape(blobName, out var dims, out var w, out var h, out var d, out var c))
                return;

            var tex = infer.GetTexture(blobName);
            if (tex == null)
                return;

            var packs = tex.volumeDepth > 0 ? tex.volumeDepth : 1;
            var channels = packs * 4;
            var physicalBuffer = NewTrackedBuffer(tex.width * tex.height * channels, sizeof(float), ComputeBufferType.Structured, "SDInpaint.DumpBlobPhysical");
            try
            {
                _ops.Pack4ToBufferCHW(tex, tex.width, tex.height, channels, physicalBuffer);
                var logicalCount = Mathf.Max(1, w) * Mathf.Max(1, h) * Mathf.Max(1, d) * Mathf.Max(1, c);
                DumpBufferToFile(path, physicalBuffer, logicalCount);
            }
            finally
            {
                DisposeBuffer(physicalBuffer, "SDInpaint.DumpBlobPhysical");
            }
        }
        catch
        {
        }
    }

    private bool TryDumpTextureBlobByLogicalShape(NcnnRepro.InferResult infer, string blobName, string path)
    {
        if (infer == null || string.IsNullOrWhiteSpace(blobName) || string.IsNullOrWhiteSpace(path))
            return false;

        try
        {
            if (!infer.TryGetLogicalShape(blobName, out var dims, out var w, out var h, out var d, out var c))
                return false;

            if (!infer.TryGetExistingTexture(blobName, out var tex) || tex == null)
                return false;

            var packs = tex.volumeDepth > 0 ? tex.volumeDepth : 1;
            var physicalChannels = dims == 4
                ? Mathf.Max(1, Mathf.CeilToInt(c / 4f)) * 4
                : Mathf.Max(1, packs) * 4;
            var physicalCount = tex.width * tex.height * Mathf.Max(1, packs) * physicalChannels;
            var logicalCount = Mathf.Max(1, w) * Mathf.Max(1, h) * Mathf.Max(1, d) * Mathf.Max(1, c);
            if (logicalCount <= 0 || logicalCount > physicalCount)
                return false;

            var physicalBuffer = NewTrackedBuffer(physicalCount, sizeof(float), ComputeBufferType.Structured, "SDInpaint.DumpBlobPhysical");
            try
            {
                if (dims == 4)
                    _ops.Pack4ToBufferCDHW(tex, tex.width, tex.height, Mathf.Max(1, d), Mathf.Max(1, c), physicalBuffer);
                else
                    _ops.Pack4ToBufferCHW(tex, tex.width, tex.height, physicalChannels, physicalBuffer);
                DumpBufferToFile(path, physicalBuffer, logicalCount);
                return true;
            }
            finally
            {
                DisposeBuffer(physicalBuffer, "SDInpaint.DumpBlobPhysical");
            }
        }
        catch
        {
            return false;
        }
    }

    private static void DumpBufferToFile(string path, ComputeBuffer buffer, int logicalCount)
    {
        if (buffer == null || string.IsNullOrWhiteSpace(path))
            return;

        var count = Mathf.Clamp(logicalCount, 0, buffer.count);
        var data = new float[buffer.count];
        buffer.GetData(data);
        if (count != data.Length)
            Array.Resize(ref data, count);
        WriteFloatArraySafe(path, data);
    }

    private static void WriteFloatArraySafe(string path, float[] data)
    {
        if (string.IsNullOrWhiteSpace(path) || data == null)
            return;

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? string.Empty);
            using var sw = new StreamWriter(path, false, Encoding.UTF8);
            for (var i = 0; i < data.Length; i++)
                sw.WriteLine(data[i].ToString("R", CultureInfo.InvariantCulture));
        }
        catch
        {
        }
    }

    private static void LogLoadProgress(string label, NcnnRepro.LoadProgress progress)
    {
        if (string.IsNullOrWhiteSpace(label))
            label = "model";
        if (progress.stage == "complete")
        {
            UnityEngine.Debug.Log("[SDInpaint] " + label + " load complete");
            return;
        }

        if (progress.stage == "layer")
        {
            if (progress.layerIndex <= 1 || progress.layerIndex == progress.layerCount || (progress.layerIndex % 50) == 0)
            {
                UnityEngine.Debug.Log("[SDInpaint] " + label + " load layer "
                    + progress.layerIndex.ToString(CultureInfo.InvariantCulture)
                    + "/" + progress.layerCount.ToString(CultureInfo.InvariantCulture)
                    + " | " + (progress.layerName ?? string.Empty)
                    + " | " + (progress.layerType ?? string.Empty));
            }
            return;
        }

        UnityEngine.Debug.Log("[SDInpaint] " + label + " load stage=" + progress.stage);
    }

    private void LogLoadProfile(string label, NcnnRepro.ModelLoadProfile profile)
    {
        if (profile == null)
            return;
        UnityEngine.Debug.Log("[SDInpaint] LoadProfile " + label
            + " | totalMs=" + profile.totalMs.ToString(CultureInfo.InvariantCulture)
            + " | parseMs=" + profile.parseParamMs.ToString(CultureInfo.InvariantCulture)
            + " | bytesRead=" + profile.totalBytesRead.ToString(CultureInfo.InvariantCulture));
    }

    private void Release()
    {
        try { _textRepro?.Dispose(); } catch { }
        try { _unetRepro?.Dispose(); } catch { }
        try { _vaeRepro?.Dispose(); } catch { }
        try { _vaeEncoderRepro?.Dispose(); } catch { }
        try { _ops?.ReleaseWinogradWorkspace(); } catch { }
        _textRepro = null;
        _unetRepro = null;
        _vaeRepro = null;
        _vaeEncoderRepro = null;
        _tokenizer = null;
        _loadedModelKey = null;
    }

    public void ReleaseRuntimeResources()
    {
        Release();
    }

    private sealed class StableDiffusionSimpleTokenizer
    {
        private readonly Dictionary<string, int> _tokenToId = new Dictionary<string, int>(StringComparer.Ordinal);

        public StableDiffusionSimpleTokenizer(string vocabPath)
        {
            if (string.IsNullOrWhiteSpace(vocabPath) || !File.Exists(vocabPath))
                throw new FileNotFoundException("Stable Diffusion vocab not found.", vocabPath);

            using var sr = new StreamReader(vocabPath);
            var index = 0;
            while (!sr.EndOfStream)
            {
                var line = sr.ReadLine() ?? string.Empty;
                if (!_tokenToId.ContainsKey(line))
                    _tokenToId.Add(line, index);
                index++;
            }

            if (_tokenToId.Count <= EndTokenId)
                throw new InvalidOperationException("Stable Diffusion vocab.txt is incomplete: " + vocabPath);
        }

        public void TokenizePrompt77(string prompt, int[] tokens77, float[] multipliers77)
        {
            if (tokens77 == null || tokens77.Length != TokenCount)
                throw new ArgumentException("tokens77 must have length " + TokenCount.ToString(CultureInfo.InvariantCulture), nameof(tokens77));
            if (multipliers77 == null || multipliers77.Length != TokenCount)
                throw new ArgumentException("multipliers77 must have length " + TokenCount.ToString(CultureInfo.InvariantCulture), nameof(multipliers77));

            for (var i = 0; i < TokenCount; i++)
            {
                tokens77[i] = EndTokenId;
                multipliers77[i] = 1f;
            }
            tokens77[0] = StartTokenId;

            var parsed = ParsePromptAttention(prompt ?? string.Empty);
            var remadeTokens = new List<int>(128);
            var remadeMultipliers = new List<float>(128);
            var lastComma = -1;
            for (var partIndex = 0; partIndex < parsed.Count; partIndex++)
            {
                var tokens = Split(parsed[partIndex].text);
                var weight = parsed[partIndex].weight;
                for (var i = 0; i < tokens.Count; i++)
                {
                    if (!_tokenToId.TryGetValue(tokens[i], out var tokenId))
                        tokenId = 0;

                    if (tokenId == 267)
                    {
                        lastComma = remadeTokens.Count;
                    }
                    else if ((Mathf.Max(remadeTokens.Count, 1) % PromptChunkTokenCount == 0)
                             && lastComma != -1
                             && remadeTokens.Count - lastComma <= 20)
                    {
                        lastComma += 1;
                        var relocTokens = remadeTokens.GetRange(lastComma, remadeTokens.Count - lastComma);
                        var relocMultipliers = remadeMultipliers.GetRange(lastComma, remadeMultipliers.Count - lastComma);
                        remadeTokens.RemoveRange(lastComma, remadeTokens.Count - lastComma);
                        remadeMultipliers.RemoveRange(lastComma, remadeMultipliers.Count - lastComma);

                        var rem = Mathf.CeilToInt(remadeTokens.Count / (float)PromptChunkTokenCount) * PromptChunkTokenCount - remadeTokens.Count;
                        for (var r = 0; r < rem; r++)
                        {
                            remadeTokens.Add(EndTokenId);
                            remadeMultipliers.Add(1f);
                        }

                        remadeTokens.AddRange(relocTokens);
                        remadeMultipliers.AddRange(relocMultipliers);
                    }

                    remadeTokens.Add(tokenId);
                    remadeMultipliers.Add(weight);
                }
            }

            var count = Mathf.Min(PromptChunkTokenCount, remadeTokens.Count);
            for (var i = 0; i < count; i++)
            {
                tokens77[i + 1] = remadeTokens[i];
                multipliers77[i + 1] = remadeMultipliers[i];
            }
        }

        private static List<(string text, float weight)> ParsePromptAttention(string text)
        {
            var result = new List<(string text, float weight)>(16);
            var round = new Stack<int>();
            var square = new Stack<int>();
            const float roundMultiplier = 1.1f;
            const float squareMultiplier = 1f / 1.1f;

            var segments = new List<string>(16);
            for (var i = 0; i < text.Length; i++)
            {
                var s = text[i].ToString();
                if (s == "(" || s == "[" || s == ")" || s == "]")
                {
                    segments.Add(s);
                }
                else
                {
                    if (segments.Count < 1)
                        segments.Add(string.Empty);
                    var last = segments[segments.Count - 1];
                    if (last == "(" || last == "[" || last == ")" || last == "]")
                        segments.Add(string.Empty);
                    segments[segments.Count - 1] += s;
                }
            }

            for (var i = 0; i < segments.Count; i++)
            {
                var segment = segments[i];
                if (segment == "(")
                {
                    round.Push(result.Count);
                }
                else if (segment == "[")
                {
                    square.Push(result.Count);
                }
                else if (segment == ")" && round.Count > 0)
                {
                    var start = round.Pop();
                    for (var p = start; p < result.Count; p++)
                        result[p] = (result[p].text, result[p].weight * roundMultiplier);
                }
                else if (segment == "]" && square.Count > 0)
                {
                    var start = square.Pop();
                    for (var p = start; p < result.Count; p++)
                        result[p] = (result[p].text, result[p].weight * squareMultiplier);
                }
                else
                {
                    result.Add((segment, 1f));
                }
            }

            while (round.Count > 0)
            {
                var start = round.Pop();
                for (var p = start; p < result.Count; p++)
                    result[p] = (result[p].text, result[p].weight * roundMultiplier);
            }

            while (square.Count > 0)
            {
                var start = square.Pop();
                for (var p = start; p < result.Count; p++)
                    result[p] = (result[p].text, result[p].weight * squareMultiplier);
            }

            for (var i = 0; i + 1 < result.Count;)
            {
                if (Math.Abs(result[i].weight - result[i + 1].weight) < 1e-6f)
                {
                    result[i] = (result[i].text + result[i + 1].text, result[i].weight);
                    result.RemoveAt(i + 1);
                }
                else
                {
                    i++;
                }
            }

            return result;
        }

        private static List<string> Split(string text)
        {
            var result = new List<string>(16);
            var src = (text ?? string.Empty) + " ";
            for (var i = 0; i < src.Length; i++)
            {
                var spacePos = src.IndexOf(' ', i);
                var commaPos = src.IndexOf(',', i);
                var pos = -1;
                if (spacePos >= 0 && commaPos >= 0)
                    pos = Mathf.Min(spacePos, commaPos);
                else if (spacePos >= 0)
                    pos = spacePos;
                else if (commaPos >= 0)
                    pos = commaPos;

                if (pos >= 0 && pos < src.Length)
                {
                    var token = src.Substring(i, pos - i);
                    var delimiter = src[pos].ToString();
                    if (token.Length > 0)
                        result.Add(token + "</w>");
                    if (delimiter != " ")
                        result.Add(delimiter + "</w>");
                    i = pos;
                }
            }

            return result;
        }
    }

    private sealed class ClipBpeTokenizer
    {
        private static readonly Regex TokenPattern = new Regex(
            @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\w\d]+",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

        private readonly Dictionary<string, int> _encoder;
        private readonly Dictionary<(string first, string second), int> _bpeRanks;
        private readonly Dictionary<string, string> _cache = new Dictionary<string, string>(StringComparer.Ordinal);
        private readonly Dictionary<byte, string> _byteEncoder;

        public ClipBpeTokenizer(string vocabJsonPath, string mergesPath)
        {
            if (string.IsNullOrWhiteSpace(vocabJsonPath) || !File.Exists(vocabJsonPath))
                throw new FileNotFoundException("Tokenizer vocab.json not found.", vocabJsonPath);
            if (string.IsNullOrWhiteSpace(mergesPath) || !File.Exists(mergesPath))
                throw new FileNotFoundException("Tokenizer merges.txt not found.", mergesPath);

            var json = File.ReadAllText(vocabJsonPath);
            _encoder = JsonConvert.DeserializeObject<Dictionary<string, int>>(json)
                       ?? throw new InvalidDataException("Failed to parse tokenizer vocab.json.");

            var merges = new List<(string first, string second)>();
            using (var sr = new StreamReader(mergesPath))
            {
                if (!sr.EndOfStream)
                    sr.ReadLine();

                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    if (string.IsNullOrWhiteSpace(line))
                        continue;
                    var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2)
                        merges.Add((parts[0], parts[1]));
                }
            }

            _bpeRanks = new Dictionary<(string first, string second), int>(merges.Count);
            for (var i = 0; i < merges.Count; i++)
                _bpeRanks[merges[i]] = i;

            _byteEncoder = BuildByteEncoder();
        }

        public int[] Tokenize(string text)
        {
            var cleaned = Clean(text);
            var tokens = new List<int>(TokenCount) { StartTokenId };

            foreach (Match match in TokenPattern.Matches(cleaned))
            {
                var token = match.Value;
                if (string.IsNullOrEmpty(token))
                    continue;

                var encoded = EncodeBytes(token);
                var bpe = ApplyBpe(encoded);
                var parts = bpe.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                for (var i = 0; i < parts.Length; i++)
                {
                    if (_encoder.TryGetValue(parts[i], out var id))
                        tokens.Add(id);
                }
            }

            tokens.Add(EndTokenId);
            if (tokens.Count > TokenCount)
            {
                tokens.RemoveRange(TokenCount, tokens.Count - TokenCount);
                tokens[TokenCount - 1] = EndTokenId;
            }

            var padded = new int[TokenCount];
            for (var i = 0; i < tokens.Count; i++)
                padded[i] = tokens[i];
            for (var i = tokens.Count; i < padded.Length; i++)
                padded[i] = EndTokenId;
            return padded;
        }

        private static string Clean(string text)
        {
            var decoded = WebUtility.HtmlDecode(text ?? string.Empty);
            decoded = decoded.Trim().ToLowerInvariant();
            decoded = Regex.Replace(decoded, "\\s+", " ");
            return decoded;
        }

        private string EncodeBytes(string token)
        {
            var bytes = Encoding.UTF8.GetBytes(token);
            var sb = new StringBuilder(bytes.Length * 2);
            for (var i = 0; i < bytes.Length; i++)
            {
                if (_byteEncoder.TryGetValue(bytes[i], out var mapped))
                    sb.Append(mapped);
            }
            return sb.ToString();
        }

        private string ApplyBpe(string token)
        {
            if (string.IsNullOrEmpty(token))
                return token;
            if (_cache.TryGetValue(token, out var cached))
                return cached;

            var word = new List<string>(token.Length);
            for (var i = 0; i < token.Length; i++)
                word.Add(token[i].ToString());
            word[word.Count - 1] += "</w>";

            while (word.Count > 1)
            {
                var bestRank = int.MaxValue;
                var bestPair = default((string first, string second));
                var found = false;
                for (var i = 0; i < word.Count - 1; i++)
                {
                    var pair = (word[i], word[i + 1]);
                    if (_bpeRanks.TryGetValue(pair, out var rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestPair = pair;
                        found = true;
                    }
                }

                if (!found)
                    break;

                var merged = new List<string>(word.Count);
                for (var i = 0; i < word.Count;)
                {
                    if (i < word.Count - 1 && word[i] == bestPair.first && word[i + 1] == bestPair.second)
                    {
                        merged.Add(bestPair.first + bestPair.second);
                        i += 2;
                    }
                    else
                    {
                        merged.Add(word[i]);
                        i++;
                    }
                }
                word = merged;
            }

            var result = string.Join(" ", word);
            _cache[token] = result;
            return result;
        }

        private static Dictionary<byte, string> BuildByteEncoder()
        {
            var bs = new List<int>();
            for (var i = 33; i <= 126; i++) bs.Add(i);
            for (var i = 161; i <= 172; i++) bs.Add(i);
            for (var i = 174; i <= 255; i++) bs.Add(i);

            var cs = new List<int>(bs);
            var n = 0;
            for (var b = 0; b < 256; b++)
            {
                if (bs.Contains(b))
                    continue;
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }

            var map = new Dictionary<byte, string>(256);
            for (var i = 0; i < bs.Count; i++)
                map[(byte)bs[i]] = char.ConvertFromUtf32(cs[i]);
            return map;
        }
    }

    private static string BoolText(bool value) => value ? "true" : "false";
}
