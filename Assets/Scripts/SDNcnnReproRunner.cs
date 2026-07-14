using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;

public struct SDNcnnReproResult
{
    public Texture2D texture;
    public string error;
    public long elapsedMs;
    public int seed;
    public bool usedInitImage;
    public bool usedMask;
    public string dumpDir;
}

public sealed class SDNcnnReproRunner : MonoBehaviour
{
    private const string ClipOutputBlobName = "cal_12";
    private const string UnetOutputBlobName = "outout";
    private const string DecoderInputBlobName = "input.1";
    private const string DecoderOutputBlobName = "815";
    private const string EncoderInputBlobName = "in0";
    private const string EncoderMeanBlobName = "out0";
    private const string EncoderStdBlobName = "out1";
    private const int TextEmbeddingWidth = 768;
    private const int PromptChunkTokenCount = 75;
    private const int PromptChunkModelTokenCount = 77;
    private const int LatentChannels = 4;
    private const float GuidanceScale = 7.5f;
    private const float Img2ImgStrength = 0.75f;
    private const float LatentScale = 0.18215f;
    private const float InvLatentScale = 1f / LatentScale;
    private const int StartTokenId = 49406;
    private const int EndTokenId = 49407;

    private static readonly string DefaultPositivePrompt =
        "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";

    private static readonly string DefaultNegativePrompt =
        "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";
    private static readonly string[] DebugUnetBlobNames =
    {
        "36",
        "37",
        "40",
        "41",
        "42",
        "43",
        "44",
        "95",
        "115",
        "139",
        "149",
        "251",
        "337",
        "425",
        "511",
        "599",
        "627",
        "711",
        "725",
        "740",
        "755",
        "772",
        "858",
        "944",
        "1032",
        "1118",
        "1194",
        "1195",
        "1196",
        "1197",
        "1198",
        "1199",
        "1200",
        "1201",
        "1202",
        "1203",
        "1204",
        "1205",
        "1206",
        "1207",
        "1208",
        "1209",
        "1210",
        "1211",
        "1212",
        "1213",
        "1214",
        "1242",
        "1243",
        "1244",
        "1245",
        "1246",
        "1266",
        "1267",
        "1268",
        "1269",
        "1270",
        "1271",
        "1272",
        "1273",
        "1274",
        "1275",
        "1276",
        "1277",
        "1278",
        "1279",
        "1280",
        "1281",
        "1282",
        "1283",
        "1284",
        "1285",
        "1286",
        "1287",
        "1288",
        "1289",
        "1290",
        "1291",
        "1292",
        "1293",
        "1294",
        "1295",
        "1296",
        "1297",
        "1298",
        "1299",
        "1300",
        "1301",
        "1302",
        "1368",
        "1369",
        "1370",
        "1371",
        "1372",
        "1373",
        "1374",
        "1375",
        "1376",
        "1377",
        "1378",
        "1379",
        "1380",
        "1381",
        "1382",
        "1383",
        "1384",
        "1385",
        "1386",
        "1387",
        "1388",
        "1390",
        "1416",
        "1417",
        "1440",
        "1441",
        "1445",
        "1448",
        "1449",
        "1450",
        "1451",
        "1453",
        "1454",
        "1455",
        "1465",
        "1470",
        "1541",
        "1542",
        "1543",
        "out0",
        "c_out_out0",
        "outout"
    };
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
        "740",
        "755",
        "772",
        "858",
        "944",
        "1032",
        "1118",
        "1204",
        "1292",
        "1378",
        "1464"
    };
    private static readonly string[] DebugClipBlobNames =
    {
        "19",
        "23",
        "24",
        "25",
        "26",
        "33",
        "34",
        "38",
        "44",
        "45",
        "49",
        "50",
        "55",
        "57",
        "out0",
        "cal_1",
        "cal_2",
        "cal_6",
        "cal_7",
        "cal_8",
        "cal_9",
        "cal_10",
        "cal_11",
        "cal_12"
    };
    private static readonly string[] DebugEncoderBlobNames =
    {
        "1",
        "4",
        "6",
        "10",
        "13",
        "15",
        "19",
        "24",
        "27",
        "31",
        "36",
        "40",
        "42",
        "47",
        "52",
        "57",
        "61",
        "63",
        "68",
        "72",
        "77",
        "81",
        "86",
        "90",
        "93",
        "94",
        "95",
        "96",
        "98",
        "99",
        "104",
        "108",
        "111",
        "112",
        "114",
        "115",
        "116"
    };

    [Serializable]
    private readonly struct ResolvedPaths
    {
        public readonly string clipParamPath;
        public readonly string clipBinPath;
        public readonly string unetBinPath;
        public readonly string decoderBinPath;
        public readonly string encoderBinPath;
        public readonly string vocabPath;
        public readonly string logSigmasPath;

        public ResolvedPaths(
            string clipParamPath,
            string clipBinPath,
            string unetBinPath,
            string decoderBinPath,
            string encoderBinPath,
            string vocabPath,
            string logSigmasPath)
        {
            this.clipParamPath = clipParamPath;
            this.clipBinPath = clipBinPath;
            this.unetBinPath = unetBinPath;
            this.decoderBinPath = decoderBinPath;
            this.encoderBinPath = encoderBinPath;
            this.vocabPath = vocabPath;
            this.logSigmasPath = logSigmasPath;
        }
    }

    private readonly struct SpatialModelInfo
    {
        public readonly string unetParamText;
        public readonly string decoderParamText;
        public readonly string encoderParamText;
        public readonly int width;
        public readonly int height;

        public SpatialModelInfo(string unetParamText, string decoderParamText, string encoderParamText, int width, int height)
        {
            this.unetParamText = unetParamText;
            this.decoderParamText = decoderParamText;
            this.encoderParamText = encoderParamText;
            this.width = width;
            this.height = height;
        }
    }

    private readonly struct PromptChunk
    {
        public readonly int[] tokens75;
        public readonly float[] multipliers75;

        public PromptChunk(int[] tokens75, float[] multipliers75)
        {
            this.tokens75 = tokens75;
            this.multipliers75 = multipliers75;
        }
    }

    public string stableDiffusionRootRelativePath = "StableDiffusion";
    public bool useReferenceAssetFallback = true;
    public bool preferOptimizedParams = true;
    public bool deterministicAncestralNoise = true;
    public bool enableDebugDump = false;
    public RenderTextureFormat tensorTextureFormat = RenderTextureFormat.ARGBFloat;
    public RenderTextureFormat decoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
    public RenderTextureFormat encoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
    public bool encoderForceBufferConvolutionAll = false;
    public bool keepRawConvWeightsForTexturePath = false;
    public bool useNcnnStyleGroupNorm = true;
    public bool useOfficialNoise = true;
    public bool useOfficialUnetCache = true;
    public bool enableMhaParallelSoftmax = true;
    public bool enableMhaQkvFusion = true;
    public bool enableLayerRuntimeProfile = false;
    public bool syncLayerRuntimeProfileGpu = false;
    public bool syncStageTimings = false;
    public bool disallowBufferAccess = false;
    public bool disallowBufferOutputs = false;
    public bool disallowBufferToTextureMaterialization = false;
    public int layerRuntimeProfileTopN = 40;

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnRepro _clipRepro;
    private NcnnRepro _unetRepro;
    private NcnnRepro _decoderRepro;
    private NcnnRepro _encoderRepro;
    private StableDiffusionSimpleTokenizer _tokenizer;
    private float[] _logSigmas;
    private ResolvedPaths? _resolvedPaths;
    private string _loadedClipKey;
    private string _loadedSpatialKey;
    private string _lastDumpDir;
    private bool _clipDebugDumped;
    private bool _unetDebugDumped;
    private bool _unetUncondDebugDumped;

    private sealed class UnetCacheBlob
    {
        public string name;
        public RenderTexture texture;
        public NcnnRepro.BufferShape shape;
    }

    public string LastDumpDir => _lastDumpDir;

    private void Awake()
    {
        EnsureRuntimeObjects();
    }

    private void OnDestroy()
    {
        Release();
    }

    public UniTask<SDNcnnReproResult> ProcessAsync(Texture initImageOrNull, CancellationToken ct)
    {
        if (initImageOrNull != null)
            return Img2ImgAsync(initImageOrNull, DefaultPositivePrompt, DefaultNegativePrompt, 256, 256, 15, 42, Img2ImgStrength, ct);
        return Txt2ImgAsync(DefaultPositivePrompt, DefaultNegativePrompt, 256, 256, 15, 42, ct);
    }

    public UniTask<SDNcnnReproResult> Txt2ImgAsync(
        string positivePrompt,
        string negativePrompt,
        int width,
        int height,
        int stepCount,
        int seed,
        CancellationToken ct)
    {
        return RunAsync(null, positivePrompt, negativePrompt, width, height, stepCount, seed, Img2ImgStrength, ct);
    }

    public UniTask<SDNcnnReproResult> Img2ImgAsync(
        Texture initImage,
        string positivePrompt,
        string negativePrompt,
        int width,
        int height,
        int stepCount,
        int seed,
        float strength,
        CancellationToken ct)
    {
        return RunAsync(initImage, positivePrompt, negativePrompt, width, height, stepCount, seed, strength, ct);
    }

    public UniTask<SDNcnnReproResult> InpaintAsync(
        Texture initImage,
        Texture maskImage,
        string positivePrompt,
        string negativePrompt,
        int width,
        int height,
        int stepCount,
        int seed,
        float strength,
        CancellationToken ct)
    {
        return RunInpaintPack4Async(initImage, maskImage, positivePrompt, negativePrompt, stepCount, seed, strength, ct);
    }

    private async UniTask<SDNcnnReproResult> RunInpaintPack4Async(
        Texture initImage,
        Texture maskImage,
        string positivePrompt,
        string negativePrompt,
        int stepCount,
        int seed,
        float strength,
        CancellationToken ct)
    {
        var runner = gameObject.AddComponent<SDInpaintingNcnnReproRunner>();
        Action<float, string> forwardProgress = ReportProgress;
        try
        {
            runner.stableDiffusionRootRelativePath = stableDiffusionRootRelativePath;
            runner.useReferenceAssetFallback = useReferenceAssetFallback;
            runner.enableDebugDump = enableDebugDump;
            runner.keepRawConvWeightsForTexturePath = keepRawConvWeightsForTexturePath;
            runner.tensorTextureFormat = tensorTextureFormat;
            runner.encoderTensorTextureFormat = encoderTensorTextureFormat;
            runner.decoderTensorTextureFormat = decoderTensorTextureFormat;
            runner.useNcnnStyleGroupNorm = useNcnnStyleGroupNorm;
            runner.useOfficialUnetCache = false;
            runner.enableMhaParallelSoftmax = enableMhaParallelSoftmax;
            runner.enableMhaQkvFusion = enableMhaQkvFusion;
            runner.enableAttentionMatMulPack4Specializations = true;
            runner.enableLayerRuntimeProfile = enableLayerRuntimeProfile;
            runner.syncLayerRuntimeProfileGpu = syncLayerRuntimeProfileGpu;
            runner.useCommandBuffer = ResolveBoolEnv("AIIMAGE_SD_USE_COMMAND_BUFFER", false);
            runner.useAsyncComputeCommandBuffer = ResolveBoolEnv("AIIMAGE_SD_USE_ASYNC_COMPUTE", runner.useAsyncComputeCommandBuffer);
            runner.disallowInferenceTempComputeBuffers = ResolveBoolEnv("AIIMAGE_SD_DISALLOW_TEMP_COMPUTE_BUFFERS", true);
            runner.disallowBufferAccess = disallowBufferAccess;
            runner.disallowBufferOutputs = disallowBufferOutputs;
            runner.disallowBufferToTextureMaterialization = disallowBufferToTextureMaterialization;
            runner.ProgressChanged += forwardProgress;

            var result = await runner.ProcessAsync(
                initImage,
                maskImage,
                positivePrompt,
                negativePrompt,
                Mathf.Max(1, stepCount),
                seed,
                Mathf.Clamp01(strength),
                GuidanceScale,
                ct);

            _lastDumpDir = result.dumpDir ?? runner.LastDumpDir;
            return new SDNcnnReproResult
            {
                texture = result.texture,
                error = result.error,
                elapsedMs = result.elapsedMs,
                seed = result.seed,
                usedInitImage = true,
                usedMask = true,
                dumpDir = _lastDumpDir
            };
        }
        finally
        {
            runner.ProgressChanged -= forwardProgress;
            UnityEngine.Object.DestroyImmediate(runner);
            ReportProgress(1f, string.Empty);
        }
    }

    private async UniTask<SDNcnnReproResult> RunAsync(
        Texture initImage,
        string positivePrompt,
        string negativePrompt,
        int width,
        int height,
        int stepCount,
        int seed,
        float strength,
        CancellationToken ct)
    {
        var totalSw = Stopwatch.StartNew();
        _lastDumpDir = null;

        SDNcnnReproResult Finish(SDNcnnReproResult result)
        {
            result.elapsedMs = totalSw.ElapsedMilliseconds;
            result.dumpDir = _lastDumpDir;
            return result;
        }

        ComputeBuffer condBuf = null;
        ComputeBuffer uncondBuf = null;
        ComputeBuffer latentBuf = null;
        var actualSeed = ResolveSeed(seed);
        try
        {
            EnsureRuntimeObjects();
            width = Mathf.Max(256, width);
            height = Mathf.Max(256, height);
            stepCount = Mathf.Max(1, stepCount);
            if ((width % 128) != 0 || (height % 128) != 0)
                return Finish(new SDNcnnReproResult { error = "Stable Diffusion width/height must be multiples of 128." });
            if (enableDebugDump)
                _lastDumpDir = CreateDumpDir("AIImage_SD_NcnnRepro");

            ReportProgress(0.02f, "Load models");
            var stageSw = Stopwatch.StartNew();
            await EnsureLoadedAsync(width, height, initImage != null, ct);
            LogStageTiming("load_models", stageSw);

            ReportProgress(0.12f, "Encode prompts");
            stageSw.Restart();
            var cond = await BuildConditioningAsync(positivePrompt ?? string.Empty, ct);
            var uncond = await BuildConditioningAsync(negativePrompt ?? string.Empty, ct);
            LogStageTiming("encode_prompts", stageSw);
            if (cond == null || cond.Length == 0 || uncond == null || uncond.Length == 0)
                return Finish(new SDNcnnReproResult { error = "Prompt conditioning failed.", seed = actualSeed, usedInitImage = initImage != null });

            condBuf = NewFloatBuffer(cond);
            uncondBuf = NewFloatBuffer(uncond);
            var condView = new NcnnTensorBuffer(condBuf, 2, TextEmbeddingWidth, cond.Length / TextEmbeddingWidth, 1, 1, false);
            var uncondView = new NcnnTensorBuffer(uncondBuf, 2, TextEmbeddingWidth, uncond.Length / TextEmbeddingWidth, 1, 1, false);

            var sigmas = BuildSigmaSchedule(stepCount);
            if (sigmas == null || sigmas.Length != stepCount + 1)
                return Finish(new SDNcnnReproResult { error = "Failed to build sigma schedule.", seed = actualSeed, usedInitImage = initImage != null });

            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                WriteAllTextSafe(Path.Combine(_lastDumpDir, "positive_prompt.txt"), positivePrompt ?? string.Empty);
                WriteAllTextSafe(Path.Combine(_lastDumpDir, "negative_prompt.txt"), negativePrompt ?? string.Empty);
                WriteFloatArray(Path.Combine(_lastDumpDir, "unity_cond.txt"), cond);
                WriteFloatArray(Path.Combine(_lastDumpDir, "unity_uncond.txt"), uncond);
                WriteFloatArray(Path.Combine(_lastDumpDir, "unity_sigmas.txt"), sigmas);
                WriteAllTextSafe(
                    Path.Combine(_lastDumpDir, "run_config.txt"),
                    "width=" + width.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "height=" + height.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "steps=" + stepCount.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "seed=" + actualSeed.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "used_init_image=" + BoolText(initImage != null) + Environment.NewLine
                    + "strength=" + strength.ToString("0.000000", CultureInfo.InvariantCulture) + Environment.NewLine
                    + "tensor_texture_format=" + tensorTextureFormat + Environment.NewLine
                    + "decoder_tensor_texture_format=" + decoderTensorTextureFormat + Environment.NewLine
                    + "encoder_tensor_texture_format=" + encoderTensorTextureFormat + Environment.NewLine
                    + "keep_raw_conv_weights_for_texture_path=" + BoolText(keepRawConvWeightsForTexturePath) + Environment.NewLine
                    + "mha_parallel_softmax=" + BoolText(ResolveMhaParallelSoftmax()) + Environment.NewLine
                    + "mha_qkv_fusion=" + BoolText(ResolveMhaQkvFusion()) + Environment.NewLine
                    + "sync_stage_timings=" + BoolText(syncStageTimings));
            }

            ReportProgress(0.22f, initImage != null ? "Encode init image" : "Init latent");
            stageSw.Restart();
            if (initImage != null)
            {
                latentBuf = await CreateImg2ImgLatentAsync(initImage, width, height, sigmas, actualSeed, Mathf.Clamp01(strength), ct);
                if (latentBuf == null)
                    return Finish(new SDNcnnReproResult { error = "img2img latent init failed.", seed = actualSeed, usedInitImage = true });
                LogStageTiming("init_img2img_latent", stageSw);
            }
            else
            {
                latentBuf = CreateTxt2ImgLatent(width, height, sigmas[0], actualSeed);
                if (latentBuf == null)
                    return Finish(new SDNcnnReproResult { error = "txt2img latent init failed.", seed = actualSeed, usedInitImage = false });
                LogStageTiming("init_txt2img_latent", stageSw);
            }

            ReportProgress(0.30f, "Sample latent");
            var samplingSigmas = initImage != null
                ? BuildImg2ImgSamplingSigmas(sigmas, stepCount, Mathf.Clamp01(strength), out var effectiveStart)
                : sigmas;
            if (samplingSigmas == null || samplingSigmas.Length < 2)
                return Finish(new SDNcnnReproResult { error = "Sampling schedule invalid.", seed = actualSeed, usedInitImage = initImage != null });

            for (var stepIndex = 0; stepIndex < samplingSigmas.Length - 1; stepIndex++)
            {
                ct.ThrowIfCancellationRequested();
                var sigma = samplingSigmas[stepIndex];
                var sigmaNext = samplingSigmas[stepIndex + 1];
                var progressBase = 0.30f;
                var progressSpan = 0.50f;
                var stepProgress = (float)stepIndex / Mathf.Max(1, samplingSigmas.Length - 1);
                ReportProgress(progressBase + progressSpan * stepProgress, "Denoise step " + (stepIndex + 1).ToString(CultureInfo.InvariantCulture) + "/" + (samplingSigmas.Length - 1).ToString(CultureInfo.InvariantCulture));
                await UniTask.Yield();

                stageSw.Restart();
                var denoisedBuf = await RunCfgDenoiserAsync(latentBuf, width, height, sigma, condView, uncondView, ct);
                LogStageTiming("denoise_step_" + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), stageSw);
                if (denoisedBuf == null)
                    return Finish(new SDNcnnReproResult { error = "UNet denoiser failed at step " + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), seed = actualSeed, usedInitImage = initImage != null });

                ComputeBuffer nextLatent = null;
                ComputeBuffer noiseBuf = null;
                try
                {
                    stageSw.Restart();
                    var sigmaUp = Mathf.Min(sigmaNext, Mathf.Sqrt(Mathf.Max(0f, sigmaNext * sigmaNext * (sigma * sigma - sigmaNext * sigmaNext) / Mathf.Max(sigma * sigma, 1e-12f))));
                    var sigmaDown = Mathf.Sqrt(Mathf.Max(0f, sigmaNext * sigmaNext - sigmaUp * sigmaUp));
                    if (sigmaUp > 0f)
                    {
                        var stepNoiseSeed = ResolveStepNoiseSeed(actualSeed, stepIndex);
                        var stepNoise = GenerateGaussianByMode(LatentElementCount(width, height), stepNoiseSeed);
                        if (initImage == null && !string.IsNullOrWhiteSpace(_lastDumpDir))
                        {
                            WriteAllTextSafe(
                                Path.Combine(_lastDumpDir, "unity_txt2img_step_noise_seed_" + (stepIndex + 1).ToString(CultureInfo.InvariantCulture) + ".txt"),
                                stepNoiseSeed.ToString(CultureInfo.InvariantCulture));
                            WriteFloatArray(
                                Path.Combine(_lastDumpDir, "unity_txt2img_step_noise_" + (stepIndex + 1).ToString(CultureInfo.InvariantCulture) + ".txt"),
                                stepNoise);
                        }
                        noiseBuf = NewFloatBuffer(stepNoise);
                    }

                    nextLatent = UpdateLatentEulerAncestral(latentBuf, denoisedBuf, sigma, sigmaDown, sigmaUp, noiseBuf, width, height);
                    LogStageTiming("update_latent_step_" + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), stageSw);
                    if (nextLatent == null)
                        return Finish(new SDNcnnReproResult { error = "Latent update failed at step " + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), seed = actualSeed, usedInitImage = initImage != null });
                    if (initImage == null && !string.IsNullOrWhiteSpace(_lastDumpDir))
                    {
                        DumpBufferToFile(
                            Path.Combine(_lastDumpDir, "unity_txt2img_latent_after_step_" + (stepIndex + 1).ToString(CultureInfo.InvariantCulture) + ".txt"),
                            nextLatent,
                            nextLatent.count);
                    }
                }
                finally
                {
                    if (noiseBuf != null)
                        DisposeBuffer(noiseBuf);
                    _unetRepro.ReturnTempBuffer(denoisedBuf);
                    _unetRepro.ReturnTempBuffer(latentBuf);
                }

                latentBuf = nextLatent;
            }

            ReportProgress(0.86f, "Decode image");
            stageSw.Restart();
            if (initImage == null && !string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_txt2img_final_latent.txt"), latentBuf, latentBuf.count);
            var finalTexture = await DecodeLatentAsync(latentBuf, width, height, ct);
            LogStageTiming("decode_latent", stageSw);
            if (finalTexture == null)
                return Finish(new SDNcnnReproResult { error = "Decoder failed.", seed = actualSeed, usedInitImage = initImage != null });

            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
                TryWriteTexturePng(finalTexture, _lastDumpDir, "final_output.png");

            ReportProgress(1f, string.Empty);
            return Finish(new SDNcnnReproResult
            {
                texture = finalTexture,
                error = null,
                seed = actualSeed,
                usedInitImage = initImage != null
            });
        }
        catch (OperationCanceledException)
        {
            return Finish(new SDNcnnReproResult { error = "Cancelled", seed = actualSeed, usedInitImage = initImage != null });
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogException(e);
            return Finish(new SDNcnnReproResult { error = e.Message, seed = actualSeed, usedInitImage = initImage != null });
        }
        finally
        {
            if (latentBuf != null)
                _unetRepro?.ReturnTempBuffer(latentBuf);
            if (condBuf != null)
                DisposeBuffer(condBuf);
            if (uncondBuf != null)
                DisposeBuffer(uncondBuf);
            ReportProgress(1f, string.Empty);
        }
    }

    [Obsolete("Legacy SD inpainting buffer path is disabled. Use RunInpaintPack4Async/SDInpaintingNcnnReproRunner instead.", true)]
    private async UniTask<SDNcnnReproResult> RunInpaintAsync(
        Texture initImage,
        Texture maskImage,
        string positivePrompt,
        string negativePrompt,
        int width,
        int height,
        int stepCount,
        int seed,
        float strength,
        CancellationToken ct)
    {
        var totalSw = Stopwatch.StartNew();
        _lastDumpDir = null;

        SDNcnnReproResult Finish(SDNcnnReproResult result)
        {
            result.elapsedMs = totalSw.ElapsedMilliseconds;
            result.dumpDir = _lastDumpDir;
            return result;
        }

        if (initImage == null)
            return Finish(new SDNcnnReproResult { error = "Inpainting requires an init image." });
        if (maskImage == null)
            return Finish(new SDNcnnReproResult { error = "Inpainting requires a mask image." });

        ComputeBuffer condBuf = null;
        ComputeBuffer uncondBuf = null;
        ComputeBuffer latentBuf = null;
        ComputeBuffer cleanLatentBuf = null;
        ComputeBuffer maskBuf = null;
        ComputeBuffer invMaskBuf = null;
        ComputeBuffer preserveNoiseBuf = null;
        Texture2D resizedMaskTex = null;
        Texture2D resizedSourceTex = null;
        var actualSeed = ResolveSeed(seed);

        try
        {
            EnsureRuntimeObjects();
            width = Mathf.Max(256, width);
            height = Mathf.Max(256, height);
            stepCount = Mathf.Max(1, stepCount);
            if ((width % 128) != 0 || (height % 128) != 0)
                return Finish(new SDNcnnReproResult { error = "Stable Diffusion width/height must be multiples of 128.", seed = actualSeed, usedInitImage = true, usedMask = true });

            if (enableDebugDump)
                _lastDumpDir = CreateDumpDir("AIImage_SD_NcnnRepro");

            ReportProgress(0.02f, "Load models");
            var stageSw = Stopwatch.StartNew();
            await EnsureLoadedAsync(width, height, true, ct);
            LogStageTiming("load_models", stageSw);

            ReportProgress(0.12f, "Encode prompts");
            stageSw.Restart();
            var cond = await BuildConditioningAsync(positivePrompt ?? string.Empty, ct);
            var uncond = await BuildConditioningAsync(negativePrompt ?? string.Empty, ct);
            LogStageTiming("encode_prompts", stageSw);
            if (cond == null || cond.Length == 0 || uncond == null || uncond.Length == 0)
                return Finish(new SDNcnnReproResult { error = "Prompt conditioning failed.", seed = actualSeed, usedInitImage = true, usedMask = true });

            condBuf = NewFloatBuffer(cond);
            uncondBuf = NewFloatBuffer(uncond);
            var condView = new NcnnTensorBuffer(condBuf, 2, TextEmbeddingWidth, cond.Length / TextEmbeddingWidth, 1, 1, false);
            var uncondView = new NcnnTensorBuffer(uncondBuf, 2, TextEmbeddingWidth, uncond.Length / TextEmbeddingWidth, 1, 1, false);

            var sigmas = BuildSigmaSchedule(stepCount);
            if (sigmas == null || sigmas.Length != stepCount + 1)
                return Finish(new SDNcnnReproResult { error = "Failed to build sigma schedule.", seed = actualSeed, usedInitImage = true, usedMask = true });

            ReportProgress(0.20f, "Prepare inpaint mask");
            stageSw.Restart();
            (maskBuf, invMaskBuf, resizedMaskTex) = CreateLatentMaskBuffers(maskImage, width, height);
            if (maskBuf == null || invMaskBuf == null || resizedMaskTex == null)
                return Finish(new SDNcnnReproResult { error = "Inpaint mask preparation failed.", seed = actualSeed, usedInitImage = true, usedMask = true });

            resizedSourceTex = ReadResizedTexture(initImage, width, height);
            if (resizedSourceTex == null)
                return Finish(new SDNcnnReproResult { error = "Inpaint source preparation failed.", seed = actualSeed, usedInitImage = true, usedMask = true });
            LogStageTiming("prepare_inpaint_inputs", stageSw);

            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                TryWriteTexturePng(resizedSourceTex, _lastDumpDir, "inpaint_source.png");
                TryWriteTexturePng(resizedMaskTex, _lastDumpDir, "inpaint_mask.png");
            }

            ReportProgress(0.24f, "Encode init image");
            stageSw.Restart();
            preserveNoiseBuf = NewFloatBuffer(GenerateInitialGaussian(LatentElementCount(width, height), actualSeed));
            cleanLatentBuf = await EncodeInitLatentAsync(initImage, width, height, actualSeed, ct);
            if (cleanLatentBuf == null)
                return Finish(new SDNcnnReproResult { error = "Inpaint latent encoding failed.", seed = actualSeed, usedInitImage = true, usedMask = true });
            LogStageTiming("encode_inpaint_source_latent", stageSw);

            var samplingSigmas = BuildImg2ImgSamplingSigmas(sigmas, stepCount, Mathf.Clamp01(strength), out var startIndex);
            if (samplingSigmas == null || samplingSigmas.Length < 2)
                return Finish(new SDNcnnReproResult { error = "Sampling schedule invalid.", seed = actualSeed, usedInitImage = true, usedMask = true });

            stageSw.Restart();
            latentBuf = BuildNoisedReferenceLatent(cleanLatentBuf, preserveNoiseBuf, samplingSigmas[0]);
            if (latentBuf == null)
                return Finish(new SDNcnnReproResult { error = "Inpaint latent init failed.", seed = actualSeed, usedInitImage = true, usedMask = true });
            LogStageTiming("init_inpaint_latent", stageSw);

            ReportProgress(0.30f, "Sample latent");
            for (var stepIndex = 0; stepIndex < samplingSigmas.Length - 1; stepIndex++)
            {
                ct.ThrowIfCancellationRequested();
                var sigma = samplingSigmas[stepIndex];
                var sigmaNext = samplingSigmas[stepIndex + 1];
                var progressBase = 0.30f;
                var progressSpan = 0.50f;
                var stepProgress = (float)stepIndex / Mathf.Max(1, samplingSigmas.Length - 1);
                ReportProgress(progressBase + progressSpan * stepProgress, "Denoise step " + (stepIndex + 1).ToString(CultureInfo.InvariantCulture) + "/" + (samplingSigmas.Length - 1).ToString(CultureInfo.InvariantCulture));
                await UniTask.Yield();

                ComputeBuffer referenceAtSigma = null;
                ComputeBuffer constrainedCurrent = null;
                ComputeBuffer denoisedBuf = null;
                ComputeBuffer nextLatent = null;
                ComputeBuffer referenceAtNext = null;
                ComputeBuffer constrainedNext = null;
                try
                {
                    stageSw.Restart();
                    referenceAtSigma = BuildNoisedReferenceLatent(cleanLatentBuf, preserveNoiseBuf, sigma);
                    constrainedCurrent = BlendLatentWithMask(latentBuf, referenceAtSigma, maskBuf, invMaskBuf);
                    denoisedBuf = await RunCfgDenoiserAsync(constrainedCurrent, width, height, sigma, condView, uncondView, ct);
                    LogStageTiming("inpaint_denoise_step_" + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), stageSw);
                    if (denoisedBuf == null)
                        return Finish(new SDNcnnReproResult { error = "UNet denoiser failed at step " + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), seed = actualSeed, usedInitImage = true, usedMask = true });

                    stageSw.Restart();
                    var sigmaUp = Mathf.Min(sigmaNext, Mathf.Sqrt(Mathf.Max(0f, sigmaNext * sigmaNext * (sigma * sigma - sigmaNext * sigmaNext) / Mathf.Max(sigma * sigma, 1e-12f))));
                    var sigmaDown = Mathf.Sqrt(Mathf.Max(0f, sigmaNext * sigmaNext - sigmaUp * sigmaUp));
                    ComputeBuffer stepNoise = null;
                    try
                    {
                        if (sigmaUp > 0f)
                            stepNoise = NewFloatBuffer(GenerateStepGaussian(LatentElementCount(width, height), actualSeed, stepIndex));
                        nextLatent = UpdateLatentEulerAncestral(constrainedCurrent, denoisedBuf, sigma, sigmaDown, sigmaUp, stepNoise, width, height);
                    }
                    finally
                    {
                        if (stepNoise != null)
                            DisposeBuffer(stepNoise);
                    }

                    if (nextLatent == null)
                        return Finish(new SDNcnnReproResult { error = "Latent update failed at step " + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), seed = actualSeed, usedInitImage = true, usedMask = true });

                    referenceAtNext = BuildNoisedReferenceLatent(cleanLatentBuf, preserveNoiseBuf, sigmaNext);
                    constrainedNext = BlendLatentWithMask(nextLatent, referenceAtNext, maskBuf, invMaskBuf);
                    LogStageTiming("inpaint_update_latent_step_" + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), stageSw);

                    _unetRepro.ReturnTempBuffer(latentBuf);
                    latentBuf = constrainedNext;
                    constrainedNext = null;
                }
                finally
                {
                    if (referenceAtSigma != null)
                        _unetRepro.ReturnTempBuffer(referenceAtSigma);
                    if (constrainedCurrent != null)
                        _unetRepro.ReturnTempBuffer(constrainedCurrent);
                    if (denoisedBuf != null)
                        _unetRepro.ReturnTempBuffer(denoisedBuf);
                    if (nextLatent != null)
                        _unetRepro.ReturnTempBuffer(nextLatent);
                    if (referenceAtNext != null)
                        _unetRepro.ReturnTempBuffer(referenceAtNext);
                    if (constrainedNext != null)
                        _unetRepro.ReturnTempBuffer(constrainedNext);
                }
            }

            ReportProgress(0.86f, "Decode image");
            stageSw.Restart();
            var decoded = await DecodeLatentAsync(latentBuf, width, height, ct);
            LogStageTiming("inpaint_decode_latent", stageSw);
            if (decoded == null)
                return Finish(new SDNcnnReproResult { error = "Decoder failed.", seed = actualSeed, usedInitImage = true, usedMask = true });

            stageSw.Restart();
            var composited = CompositeWithMask(resizedSourceTex, decoded, resizedMaskTex);
            LogStageTiming("inpaint_composite", stageSw);
            if (composited == null)
            {
                UnityEngine.Object.DestroyImmediate(decoded);
                return Finish(new SDNcnnReproResult { error = "Inpaint composite failed.", seed = actualSeed, usedInitImage = true, usedMask = true });
            }

            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                TryWriteTexturePng(decoded, _lastDumpDir, "inpaint_generated.png");
                TryWriteTexturePng(composited, _lastDumpDir, "final_output.png");
            }

            UnityEngine.Object.DestroyImmediate(decoded);
            ReportProgress(1f, string.Empty);
            return Finish(new SDNcnnReproResult
            {
                texture = composited,
                error = null,
                seed = actualSeed,
                usedInitImage = true,
                usedMask = true
            });
        }
        catch (OperationCanceledException)
        {
            return Finish(new SDNcnnReproResult { error = "Cancelled", seed = actualSeed, usedInitImage = true, usedMask = true });
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogException(e);
            return Finish(new SDNcnnReproResult { error = e.Message, seed = actualSeed, usedInitImage = true, usedMask = true });
        }
        finally
        {
            if (latentBuf != null)
                _unetRepro?.ReturnTempBuffer(latentBuf);
            if (cleanLatentBuf != null)
                _unetRepro?.ReturnTempBuffer(cleanLatentBuf);
            if (maskBuf != null)
                DisposeBuffer(maskBuf);
            if (invMaskBuf != null)
                DisposeBuffer(invMaskBuf);
            if (preserveNoiseBuf != null)
                DisposeBuffer(preserveNoiseBuf);
            if (condBuf != null)
                DisposeBuffer(condBuf);
            if (uncondBuf != null)
                DisposeBuffer(uncondBuf);
            if (resizedMaskTex != null)
                UnityEngine.Object.DestroyImmediate(resizedMaskTex);
            if (resizedSourceTex != null)
                UnityEngine.Object.DestroyImmediate(resizedSourceTex);
            ReportProgress(1f, string.Empty);
        }
    }

    private async UniTask EnsureLoadedAsync(int width, int height, bool needEncoder, CancellationToken ct)
    {
        var paths = ResolvePaths();
        if (_logSigmas == null || _logSigmas.Length != 1000)
        {
            UnityEngine.Debug.Log("[SD] Load log_sigmas | path=" + paths.logSigmasPath);
            _logSigmas = LoadFloatArray(paths.logSigmasPath, 1000);
        }

        var clipKey = paths.clipParamPath + "|" + paths.clipBinPath + "|" + paths.vocabPath + "|" + BoolText(keepRawConvWeightsForTexturePath);
        if (!string.Equals(_loadedClipKey, clipKey, StringComparison.Ordinal))
        {
            UnityEngine.Debug.Log("[SD] Load CLIP | param=" + paths.clipParamPath + " | bin=" + paths.clipBinPath);
            _clipRepro?.Dispose();
            _clipRepro = NcnnInferenceSessionFactory.Create(_ops);
            ApplyCommonOptions(_clipRepro);
            _clipDebugDumped = false;
            var clipParamText = File.ReadAllText(paths.clipParamPath);
            UnityEngine.Debug.Log("[SD] Open CLIP bin stream");
            using (var fs = new FileStream(paths.clipBinPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, false))
            using (var br = new NcnnBinReader(fs))
            {
                UnityEngine.Debug.Log("[SD] Begin CLIP LoadModel");
                _clipRepro.LoadModel(clipParamText, br, progress => LogLoadProgress("CLIP", progress));
            }
            LogLoadProfile("CLIP", _clipRepro.LastLoadProfile);

            _tokenizer = new StableDiffusionSimpleTokenizer(paths.vocabPath);
            _loadedClipKey = clipKey;
            UnityEngine.Debug.Log("[SD] CLIP loaded");
        }

        var spatialKey = width.ToString(CultureInfo.InvariantCulture) + "x" + height.ToString(CultureInfo.InvariantCulture) + "|" + BoolText(needEncoder) + "|" + tensorTextureFormat + "|" + decoderTensorTextureFormat + "|" + encoderTensorTextureFormat + "|" + BoolText(ResolveEncoderForceBufferConvolution()) + "|" + BoolText(keepRawConvWeightsForTexturePath) + "|" + paths.unetBinPath + "|" + paths.decoderBinPath + "|" + (paths.encoderBinPath ?? string.Empty);
        if (string.Equals(_loadedSpatialKey, spatialKey, StringComparison.Ordinal))
            return;

        _unetRepro?.Dispose();
        _decoderRepro?.Dispose();
        _encoderRepro?.Dispose();
        _unetRepro = NcnnInferenceSessionFactory.Create(_ops);
        _decoderRepro = NcnnInferenceSessionFactory.Create(_ops);
        _encoderRepro = needEncoder ? NcnnInferenceSessionFactory.Create(_ops) : null;
        _unetDebugDumped = false;
        _unetUncondDebugDumped = false;
        ApplyCommonOptions(_unetRepro);
        ApplyCommonOptions(_decoderRepro);
        _decoderRepro.TensorTextureFormat = decoderTensorTextureFormat;
        if (_encoderRepro != null)
        {
            ApplyCommonOptions(_encoderRepro);
            _encoderRepro.TensorTextureFormat = encoderTensorTextureFormat;
            _encoderRepro.ForceBufferConvolutionAll = ResolveEncoderForceBufferConvolution();
        }

        var modelInfo = await BuildSpatialModelInfoAsync(paths, width, height, needEncoder, ct);

        UnityEngine.Debug.Log("[SD] Load UNet | size=" + width.ToString(CultureInfo.InvariantCulture) + "x" + height.ToString(CultureInfo.InvariantCulture) + " | bin=" + paths.unetBinPath);
        UnityEngine.Debug.Log("[SD] Open UNet bin stream");
        using (var unetFs = new FileStream(paths.unetBinPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, false))
        using (var unetBr = new NcnnBinReader(unetFs))
        {
            UnityEngine.Debug.Log("[SD] Begin UNet LoadModel");
            _unetRepro.LoadModel(modelInfo.unetParamText, unetBr, progress => LogLoadProgress("UNet", progress));
        }
        LogLoadProfile("UNet", _unetRepro.LastLoadProfile);
        UnityEngine.Debug.Log("[SD] UNet loaded");

        UnityEngine.Debug.Log("[SD] Load VAE decoder | bin=" + paths.decoderBinPath);
        UnityEngine.Debug.Log("[SD] Open VAE decoder bin stream");
        using (var decoderFs = new FileStream(paths.decoderBinPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, false))
        using (var decoderBr = new NcnnBinReader(decoderFs))
        {
            UnityEngine.Debug.Log("[SD] Begin VAE decoder LoadModel");
            _decoderRepro.LoadModel(modelInfo.decoderParamText, decoderBr, progress => LogLoadProgress("VAE-Decoder", progress));
        }
        LogLoadProfile("VAE-Decoder", _decoderRepro.LastLoadProfile);
        UnityEngine.Debug.Log("[SD] VAE decoder loaded");

        if (needEncoder)
        {
            if (string.IsNullOrWhiteSpace(paths.encoderBinPath) || !File.Exists(paths.encoderBinPath))
                throw new FileNotFoundException("Autoencoder encoder bin not found. img2img requires AutoencoderKL-encoder-512-512-fp16.bin.");

            UnityEngine.Debug.Log("[SD] Load VAE encoder | bin=" + paths.encoderBinPath);
            UnityEngine.Debug.Log("[SD] Open VAE encoder bin stream");
            using (var encoderFs = new FileStream(paths.encoderBinPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, false))
            using (var encoderBr = new NcnnBinReader(encoderFs))
            {
                UnityEngine.Debug.Log("[SD] Begin VAE encoder LoadModel");
                _encoderRepro.LoadModel(modelInfo.encoderParamText, encoderBr, progress => LogLoadProgress("VAE-Encoder", progress));
            }
            LogLoadProfile("VAE-Encoder", _encoderRepro.LastLoadProfile);
            UnityEngine.Debug.Log("[SD] VAE encoder loaded");
        }

        _loadedSpatialKey = spatialKey;
    }

    private async UniTask<SpatialModelInfo> BuildSpatialModelInfoAsync(ResolvedPaths paths, int width, int height, bool needEncoder, CancellationToken ct)
    {
        var unetParamText = await LoadUnetParamTextAsync(width, height, ct);
        var decoderParamText = await LoadDecoderParamTextAsync(width, height, ct);
        string encoderParamText = null;
        if (needEncoder)
            encoderParamText = await LoadEncoderParamTextAsync(width, height, ct);
        return new SpatialModelInfo(unetParamText, decoderParamText, encoderParamText, width, height);
    }

    private async UniTask<string> LoadUnetParamTextAsync(int width, int height, CancellationToken ct)
    {
        if (preferOptimizedParams)
        {
            var exact = FindParamFile(
                "UNetModel-" + height.ToString(CultureInfo.InvariantCulture) + "-" + width.ToString(CultureInfo.InvariantCulture) + "-MHA-fp16-opt.param",
                "UNetModel-" + width.ToString(CultureInfo.InvariantCulture) + "-" + height.ToString(CultureInfo.InvariantCulture) + "-MHA-fp16-opt.param",
                width == height ? "UNetModel-" + width.ToString(CultureInfo.InvariantCulture) + "-MHA-fp16-opt.param" : null);
            if (!string.IsNullOrWhiteSpace(exact))
                return File.ReadAllText(exact);
        }

        var basePath = FindParamFile("UNetModel-base-MHA-fp16.param");
        if (string.IsNullOrWhiteSpace(basePath))
            throw new FileNotFoundException("UNet base param not found.");
        var baseText = File.ReadAllText(basePath);
        return GenerateUnetParamText(baseText, height, width);
    }

    private async UniTask<string> LoadDecoderParamTextAsync(int width, int height, CancellationToken ct)
    {
        if (preferOptimizedParams)
        {
            var exact = FindParamFile(
                "AutoencoderKL-" + height.ToString(CultureInfo.InvariantCulture) + "-" + width.ToString(CultureInfo.InvariantCulture) + "-fp16-opt.param",
                "AutoencoderKL-" + width.ToString(CultureInfo.InvariantCulture) + "-" + height.ToString(CultureInfo.InvariantCulture) + "-fp16-opt.param",
                width == height ? "AutoencoderKL-" + width.ToString(CultureInfo.InvariantCulture) + "-fp16-opt.param" : null);
            if (!string.IsNullOrWhiteSpace(exact))
                return File.ReadAllText(exact);
        }

        var basePath = FindParamFile("AutoencoderKL-base-fp16.param");
        if (string.IsNullOrWhiteSpace(basePath))
            throw new FileNotFoundException("Autoencoder decoder base param not found.");
        var baseText = File.ReadAllText(basePath);
        return GenerateDecoderParamText(baseText, height, width);
    }

    private async UniTask<string> LoadEncoderParamTextAsync(int width, int height, CancellationToken ct)
    {
        var basePath = FindParamFile("AutoencoderKL-encoder-512-512-fp16.param");
        if (string.IsNullOrWhiteSpace(basePath))
            throw new FileNotFoundException("Autoencoder encoder base param not found.");
        var baseText = File.ReadAllText(basePath);
        if (width == 512 && height == 512)
            return baseText;
        return GenerateEncoderParamText(baseText, height, width);
    }

    private async UniTask<float[]> BuildConditioningAsync(string prompt, CancellationToken ct)
    {
        if (_clipRepro == null || _tokenizer == null)
            throw new InvalidOperationException("CLIP model is not loaded.");

        var chunks = _tokenizer.TokenizePrompt(prompt ?? string.Empty);
        if (chunks.Count == 0)
            chunks.Add(StableDiffusionSimpleTokenizer.EmptyChunk());

        var rowsPerChunk = PromptChunkModelTokenCount;
        var all = new float[chunks.Count * rowsPerChunk * TextEmbeddingWidth];
        var dummyCond = new float[TextEmbeddingWidth];
        using var dummyCondBuffer = NewFloatBuffer(dummyCond);
        var dummyCondView = new NcnnTensorBuffer(dummyCondBuffer, 2, TextEmbeddingWidth, 1, 1, 1, false);

        for (var i = 0; i < chunks.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            ReportProgress(0.12f + 0.10f * ((float)i / Mathf.Max(1, chunks.Count)), "Encode prompt chunk " + (i + 1).ToString(CultureInfo.InvariantCulture) + "/" + chunks.Count.ToString(CultureInfo.InvariantCulture));

            var tokens77 = new int[PromptChunkModelTokenCount];
            var multipliers77 = new float[PromptChunkModelTokenCount];
            for (var tokenIndex = 0; tokenIndex < PromptChunkModelTokenCount; tokenIndex++)
            {
                tokens77[tokenIndex] = StartTokenId;
                multipliers77[tokenIndex] = 1f;
            }
            Array.Copy(chunks[i].tokens75, 0, tokens77, 1, PromptChunkTokenCount);
            Array.Copy(chunks[i].multipliers75, 0, multipliers77, 1, PromptChunkTokenCount);

            using var tokenBuffer = new ComputeBuffer(tokens77.Length, sizeof(int), ComputeBufferType.Structured);
            using var multiplierBuffer = NewFloatBuffer(multipliers77);
            tokenBuffer.SetData(tokens77);
            var tokenView = new NcnnTensorBuffer(tokenBuffer, 1, tokens77.Length, 1, 1, 1, false);
            var multiplierView = new NcnnTensorBuffer(multiplierBuffer, 1, multipliers77.Length, 1, 1, 1, false);

            HashSet<string> pinned = null;
            var shouldDumpClip = enableDebugDump && !_clipDebugDumped && !string.IsNullOrWhiteSpace(_lastDumpDir);
            if (shouldDumpClip)
            {
                WriteIntArray(Path.Combine(_lastDumpDir, "unity_clip_in_token.txt"), tokens77);
                WriteFloatArray(Path.Combine(_lastDumpDir, "unity_clip_in_multiplier.txt"), multipliers77);
                WriteFloatArray(Path.Combine(_lastDumpDir, "unity_clip_in_cond.txt"), dummyCond);
                pinned = new HashSet<string>(DebugClipBlobNames, StringComparer.Ordinal)
                {
                    ClipOutputBlobName
                };
            }

            using var infer = _clipRepro.InferWithMultiInputs(
                null,
                new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal)
                {
                    { "token", tokenView },
                    { "multiplier", multiplierView },
                    { "cond", dummyCondView }
                },
                pinned ?? new HashSet<string>(StringComparer.Ordinal) { ClipOutputBlobName });
            DumpLayerRuntimeProfile(_clipRepro, "layer_profile_clip_chunk_" + (i + 1).ToString(CultureInfo.InvariantCulture));

            if (shouldDumpClip)
            {
                for (var bi = 0; bi < DebugClipBlobNames.Length; bi++)
                    TryDumpAnyBlob(infer, DebugClipBlobNames[bi], Path.Combine(_lastDumpDir, "unity_clip_blob_" + DebugClipBlobNames[bi] + ".txt"), false);
                _clipDebugDumped = true;
            }

            var chunkData = infer.ReadTextureDataForOutput(ClipOutputBlobName);
            if (chunkData == null || chunkData.Length != rowsPerChunk * TextEmbeddingWidth)
                throw new InvalidOperationException("CLIP conditioning output invalid for prompt chunk " + (i + 1).ToString(CultureInfo.InvariantCulture));

            Buffer.BlockCopy(chunkData, 0, all, i * chunkData.Length * sizeof(float), chunkData.Length * sizeof(float));
            await UniTask.Yield();
        }

        if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            WriteFloatArray(Path.Combine(_lastDumpDir, "conditioning_" + SanitizeFileName(prompt) + ".txt"), all);

        return all;
    }

    private ComputeBuffer CreateTxt2ImgLatent(int width, int height, float sigma0, int seed)
    {
        var latent = _unetRepro.RentTempBuffer(LatentElementCount(width, height), sizeof(float));
        latent.SetData(GenerateInitialGaussian(latent.count, seed));
        _ops.MulScalarInplace(latent, sigma0, latent.count);
        if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_txt2img_latent_init.txt"), latent, latent.count);
        return latent;
    }

    private async UniTask<ComputeBuffer> CreateImg2ImgLatentAsync(Texture initImage, int width, int height, float[] sigmas, int seed, float strength, CancellationToken ct)
    {
        var cleanLatent = await EncodeInitLatentAsync(initImage, width, height, seed, ct);
        if (cleanLatent == null)
            return null;

        try
        {
            var totalSteps = Mathf.Max(1, sigmas.Length - 1);
            var img2ImgStepCount = Mathf.Clamp(Mathf.FloorToInt(totalSteps * strength), 1, totalSteps);
            var sigmaIndex = Mathf.Clamp((sigmas.Length - 1) - img2ImgStepCount, 0, sigmas.Length - 1);
            var sigmaKick = sigmas[sigmaIndex];
            using var kickNoise = NewFloatBuffer(GenerateInitialGaussian(LatentElementCount(width, height), seed));
            var result = BuildNoisedReferenceLatent(cleanLatent, kickNoise, sigmaKick);
            if (!string.IsNullOrWhiteSpace(_lastDumpDir) && result != null)
                DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_img2img_latent_after_sigma_reset.txt"), result, result.count);
            return result;
        }
        finally
        {
            if (cleanLatent != null)
                _unetRepro?.ReturnTempBuffer(cleanLatent);
        }
    }

    private async UniTask<ComputeBuffer> EncodeInitLatentAsync(Texture initImage, int width, int height, int seed, CancellationToken ct)
    {
        if (_encoderRepro == null)
            throw new InvalidOperationException("img2img requires encoder model.");
        if (initImage == null)
            throw new ArgumentNullException(nameof(initImage));

        RenderTexture inputPack4 = null;
        RenderTexture stdTex = null;
        ComputeBuffer meanBuf = null;
        ComputeBuffer stdBuf = null;
        ComputeBuffer inputBuf = null;
        ComputeBuffer noiseBuf = null;
        ComputeBuffer mulBuf = null;
        ComputeBuffer latentBuf = null;
        try
        {
            inputBuf = CreateEncoderInputBufferNcnn(initImage, width, height);
            inputPack4 = _encoderRepro.RentTempArray(width, height, 1, RenderTextureFormat.ARGBHalf);
            _ops.FillPack4FromBufferCHW(inputBuf, width, height, 3, inputPack4);
            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_encoder_in0.txt"), inputBuf, inputBuf.count);

            var encoderOutputs = new HashSet<string>(StringComparer.Ordinal)
            {
                EncoderMeanBlobName,
                EncoderStdBlobName
            };
            var encoderDebugBlobs = enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir)
                ? ResolveDebugEncoderBlobNames()
                : Array.Empty<string>();
            if (encoderDebugBlobs.Length > 0)
            {
                for (var i = 0; i < encoderDebugBlobs.Length; i++)
                    encoderOutputs.Add(encoderDebugBlobs[i]);
            }

            using (var infer = _encoderRepro.Infer(inputPack4, 1, EncoderInputBlobName, encoderOutputs))
            {
                DumpLayerRuntimeProfile(_encoderRepro, "layer_profile_encoder");

                if (encoderDebugBlobs.Length > 0)
                {
                    for (var i = 0; i < encoderDebugBlobs.Length; i++)
                        TryDumpAnyBlob(infer, encoderDebugBlobs[i], Path.Combine(_lastDumpDir, "unity_encoder_blob_" + encoderDebugBlobs[i] + ".txt"), true, _encoderRepro);
                }

#if UNITY_EDITOR || AIIMAGE_INFERENCE_DEBUG_ORACLE
                meanBuf = infer.ExtractBuffer(EncoderMeanBlobName);
                try
                {
                    stdBuf = infer.ExtractBuffer(EncoderStdBlobName);
                }
                catch
                {
                    stdTex = infer.ExtractTexture(EncoderStdBlobName);
                }
#else
                throw new NotSupportedException("Stable Diffusion encoder latent extraction requires legacy ComputeBuffer output; Pack4 CommandBuffer latent path is not implemented.");
#endif
            }

            if (stdBuf == null)
            {
                if (stdTex == null)
                    throw new InvalidOperationException("Encoder std output not found.");
                stdBuf = _encoderRepro.RentTempBuffer(LatentElementCount(width, height), sizeof(float));
                _ops.Pack4ToBufferCHW(stdTex, stdTex.width, stdTex.height, LatentChannels, stdBuf);
            }

            noiseBuf = NewFloatBuffer(GenerateInitialGaussian(LatentElementCount(width, height), seed));
            mulBuf = _unetRepro.RentTempBuffer(noiseBuf.count, sizeof(float));
            latentBuf = _unetRepro.RentTempBuffer(noiseBuf.count, sizeof(float));
            _ops.BinaryOpBuf(stdBuf, noiseBuf, noiseBuf.count, 2, mulBuf);
            _ops.BinaryOpBuf(meanBuf, mulBuf, noiseBuf.count, 0, latentBuf);
            _ops.MulScalarInplace(latentBuf, LatentScale, latentBuf.count);

            if (!string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_encoder_out0.txt"), meanBuf, meanBuf.count);
                DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_encoder_out1.txt"), stdBuf, stdBuf.count);
                DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_img2img_latent_after_encoder.txt"), latentBuf, latentBuf.count);
            }

            await UniTask.Yield();
            return latentBuf;
        }
        finally
        {
            if (inputPack4 != null)
                _encoderRepro?.ReturnTempArray(inputPack4);
            if (stdTex != null)
                _encoderRepro?.ReturnTempArray(stdTex);
            if (meanBuf != null)
                _encoderRepro?.ReturnTempBuffer(meanBuf);
            if (stdBuf != null)
                _encoderRepro?.ReturnTempBuffer(stdBuf);
            if (inputBuf != null)
                DisposeBuffer(inputBuf);
            if (noiseBuf != null)
                DisposeBuffer(noiseBuf);
            if (mulBuf != null)
                _unetRepro?.ReturnTempBuffer(mulBuf);
        }
    }

    private float[] BuildImg2ImgSamplingSigmas(float[] sigmas, int stepCount, float strength, out int startIndex)
    {
        var effectiveSteps = Mathf.Clamp(Mathf.FloorToInt(stepCount * strength), 1, stepCount);
        startIndex = Mathf.Clamp(stepCount - effectiveSteps, 0, stepCount);
        var result = new float[effectiveSteps + 1];
        Array.Copy(sigmas, startIndex, result, 0, result.Length);
        return result;
    }

    private ComputeBuffer BuildNoisedReferenceLatent(ComputeBuffer cleanLatentBuf, ComputeBuffer noiseBuf, float sigma)
    {
        if (cleanLatentBuf == null)
            throw new ArgumentNullException(nameof(cleanLatentBuf));
        if (noiseBuf == null)
            throw new ArgumentNullException(nameof(noiseBuf));

        var count = cleanLatentBuf.count;
        if (Mathf.Abs(sigma) <= 1e-8f)
        {
            var copy = _unetRepro.RentTempBuffer(count, sizeof(float));
            _ops.CopyBuf(cleanLatentBuf, copy, count);
            return copy;
        }

        var scaledNoise = _unetRepro.RentTempBuffer(count, sizeof(float));
        var outBuf = _unetRepro.RentTempBuffer(count, sizeof(float));
        try
        {
            _ops.BinaryOpScalarBuf(noiseBuf, sigma, count, 2, scaledNoise);
            _ops.BinaryOpBuf(cleanLatentBuf, scaledNoise, count, 0, outBuf);
            return outBuf;
        }
        finally
        {
            _unetRepro.ReturnTempBuffer(scaledNoise);
        }
    }

    private ComputeBuffer BlendLatentWithMask(ComputeBuffer generatedBuf, ComputeBuffer originalBuf, ComputeBuffer maskBuf, ComputeBuffer invMaskBuf)
    {
        if (generatedBuf == null) throw new ArgumentNullException(nameof(generatedBuf));
        if (originalBuf == null) throw new ArgumentNullException(nameof(originalBuf));
        if (maskBuf == null) throw new ArgumentNullException(nameof(maskBuf));
        if (invMaskBuf == null) throw new ArgumentNullException(nameof(invMaskBuf));

        var count = generatedBuf.count;
        var genMasked = _unetRepro.RentTempBuffer(count, sizeof(float));
        var origMasked = _unetRepro.RentTempBuffer(count, sizeof(float));
        var outBuf = _unetRepro.RentTempBuffer(count, sizeof(float));
        try
        {
            _ops.BinaryOpBuf(generatedBuf, maskBuf, count, 2, genMasked);
            _ops.BinaryOpBuf(originalBuf, invMaskBuf, count, 2, origMasked);
            _ops.BinaryOpBuf(genMasked, origMasked, count, 0, outBuf);
            return outBuf;
        }
        finally
        {
            _unetRepro.ReturnTempBuffer(genMasked);
            _unetRepro.ReturnTempBuffer(origMasked);
        }
    }

    private async UniTask<ComputeBuffer> RunCfgDenoiserAsync(ComputeBuffer latentBuf, int width, int height, float sigma, NcnnTensorBuffer condView, NcnnTensorBuffer uncondView, CancellationToken ct)
    {
        RenderTexture latentTex = null;
        ComputeBuffer condOut = null;
        ComputeBuffer uncondOut = null;
        ComputeBuffer timestepBuf = null;
        ComputeBuffer cInBuf = null;
        ComputeBuffer cOutBuf = null;
        ComputeBuffer diffBuf = null;
        ComputeBuffer scaledDiffBuf = null;
        ComputeBuffer finalBuf = null;
        List<UnetCacheBlob> unetCache = null;
        try
        {
            var latentW = Mathf.Max(1, width / 8);
            var latentH = Mathf.Max(1, height / 8);
            latentTex = _unetRepro.RentTempArray(latentW, latentH, 1, RenderTextureFormat.ARGBHalf);
            _ops.FillPack4FromBufferCHW(latentBuf, latentW, latentH, LatentChannels, latentTex);

            var t = SigmaToT(sigma);
            var cIn = 1f / Mathf.Sqrt(sigma * sigma + 1f);
            var cOut = -sigma;
            timestepBuf = NewFloatBuffer(new[] { t });
            cInBuf = NewFloatBuffer(new[] { cIn });
            cOutBuf = NewFloatBuffer(new[] { cOut });
            var timestepView = new NcnnTensorBuffer(timestepBuf, 1, 1, 1, 1, 1, false);
            var cInView = new NcnnTensorBuffer(cInBuf, 1, 1, 1, 1, 1, false);
            var cOutView = new NcnnTensorBuffer(cOutBuf, 1, 1, 1, 1, 1, false);

            if (enableDebugDump && !_unetDebugDumped && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                WriteFloatArray(Path.Combine(_lastDumpDir, "unity_unet_in1_t.txt"), new[] { t });
                WriteFloatArray(Path.Combine(_lastDumpDir, "unity_unet_c_in.txt"), new[] { cIn });
                WriteFloatArray(Path.Combine(_lastDumpDir, "unity_unet_c_out.txt"), new[] { cOut });
                DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_unet_in0.txt"), latentBuf, latentBuf.count);
            }

            if (useOfficialUnetCache)
                unetCache = new List<UnetCacheBlob>(OfficialUnetCacheBlobNames.Length);

            condOut = RunUnetOnce(latentTex, timestepView, condView, cInView, cOutView, width, height, "cond", unetCache, null);
            var cacheForUncond = unetCache != null && unetCache.Count == OfficialUnetCacheBlobNames.Length ? unetCache : null;
            uncondOut = RunUnetOnce(latentTex, timestepView, uncondView, cInView, cOutView, width, height, "uncond", null, cacheForUncond);
            if (condOut == null || uncondOut == null)
                return null;

            diffBuf = _unetRepro.RentTempBuffer(condOut.count, sizeof(float));
            scaledDiffBuf = _unetRepro.RentTempBuffer(condOut.count, sizeof(float));
            finalBuf = _unetRepro.RentTempBuffer(condOut.count, sizeof(float));
            _ops.BinaryOpBuf(condOut, uncondOut, condOut.count, 1, diffBuf);
            _ops.BinaryOpScalarBuf(diffBuf, GuidanceScale, diffBuf.count, 2, scaledDiffBuf);
            _ops.BinaryOpBuf(uncondOut, scaledDiffBuf, diffBuf.count, 0, finalBuf);

            await UniTask.Yield();
            return finalBuf;
        }
        finally
        {
            if (latentTex != null)
                _unetRepro?.ReturnTempArray(latentTex);
            if (condOut != null)
                _unetRepro?.ReturnTempBuffer(condOut);
            if (uncondOut != null)
                _unetRepro?.ReturnTempBuffer(uncondOut);
            if (timestepBuf != null)
                DisposeBuffer(timestepBuf);
            if (cInBuf != null)
                DisposeBuffer(cInBuf);
            if (cOutBuf != null)
                DisposeBuffer(cOutBuf);
            if (diffBuf != null)
                _unetRepro?.ReturnTempBuffer(diffBuf);
            if (scaledDiffBuf != null)
                _unetRepro?.ReturnTempBuffer(scaledDiffBuf);
            ReleaseUnetCache(unetCache);
        }
    }

    private ComputeBuffer RunUnetOnce(
        RenderTexture latentTex,
        NcnnTensorBuffer timestepView,
        NcnnTensorBuffer condView,
        NcnnTensorBuffer cInView,
        NcnnTensorBuffer cOutView,
        int width,
        int height,
        string dumpTag = null,
        List<UnetCacheBlob> captureCache = null,
        IReadOnlyList<UnetCacheBlob> reuseCache = null)
    {
        HashSet<string> pinned = null;
        var shouldDumpCond = enableDebugDump
            && !_unetDebugDumped
            && !string.IsNullOrWhiteSpace(_lastDumpDir)
            && string.Equals(dumpTag, "cond", StringComparison.Ordinal);
        var debugUnetBlobNames = shouldDumpCond ? ResolveDebugUnetBlobNames() : DebugUnetBlobNames;
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

        var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
        {
            { "in0", latentTex }
        };
        Dictionary<string, NcnnRepro.BufferShape> textureInputShapes = null;
        if (reuseCache != null)
        {
            textureInputShapes = new Dictionary<string, NcnnRepro.BufferShape>(StringComparer.Ordinal);
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
                { "in1", timestepView },
                { "in2", condView },
                { "c_in", cInView },
                { "c_out", cOutView }
            },
            pinned,
            textureInputShapes);
        DumpLayerRuntimeProfile(_unetRepro, "layer_profile_unet_" + SanitizeFileName(string.IsNullOrWhiteSpace(dumpTag) ? "run" : dumpTag));

        if (captureCache != null)
            TryCaptureOfficialUnetCache(infer, captureCache);

        if (shouldDumpCond)
        {
            for (var i = 0; i < debugUnetBlobNames.Length; i++)
            {
                var blobName = debugUnetBlobNames[i];
                if (!TryDumpUnetCacheBlob(captureCache, blobName, Path.Combine(_lastDumpDir, "unity_unet_blob_" + blobName + ".txt")))
                    TryDumpAnyBlob(infer, blobName, Path.Combine(_lastDumpDir, "unity_unet_blob_" + blobName + ".txt"));
            }
        }

        var outTex = infer.ExtractTexture(UnetOutputBlobName);
        if (outTex == null)
            return null;

        try
        {
            var outBuf = _unetRepro.RentTempBuffer(LatentElementCount(width, height), sizeof(float));
            _ops.Pack4ToBufferCHW(outTex, outTex.width, outTex.height, LatentChannels, outBuf);
            if (shouldDumpCond)
            {
                DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_unet_outout_cond.txt"), outBuf, outBuf.count);
                _unetDebugDumped = true;
            }
            else if (enableDebugDump
                     && !_unetUncondDebugDumped
                     && !string.IsNullOrWhiteSpace(_lastDumpDir)
                     && string.Equals(dumpTag, "uncond", StringComparison.Ordinal))
            {
                DumpBufferToFile(Path.Combine(_lastDumpDir, "unity_unet_outout_uncond.txt"), outBuf, outBuf.count);
                _unetUncondDebugDumped = true;
            }
            return outBuf;
        }
        finally
        {
            _unetRepro.ReturnTempArray(outTex);
        }
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
            UnityEngine.Debug.LogWarning("[SD] official UNet cache disabled for this step: " + e.Message);
            ReleaseUnetCache(captured);
            dst.Clear();
            return false;
        }
    }

    private bool TryDumpUnetCacheBlob(IReadOnlyList<UnetCacheBlob> cache, string blobName, string path)
    {
        if (cache == null || string.IsNullOrWhiteSpace(blobName) || string.IsNullOrWhiteSpace(path))
            return false;

        for (var i = 0; i < cache.Count; i++)
        {
            var blob = cache[i];
            if (blob == null || blob.texture == null || !string.Equals(blob.name, blobName, StringComparison.Ordinal))
                continue;

            try
            {
                var tex = blob.texture;
                var packs = tex.volumeDepth > 0 ? tex.volumeDepth : 1;
                var physicalChannels = packs * 4;
                using var physicalBuffer = new ComputeBuffer(tex.width * tex.height * physicalChannels, sizeof(float), ComputeBufferType.Structured);
                _ops.Pack4ToBufferCHW(tex, tex.width, tex.height, physicalChannels, physicalBuffer);
                var logicalCount = Mathf.Max(1, blob.shape.w) * Mathf.Max(1, blob.shape.h) * Mathf.Max(1, blob.shape.d) * Mathf.Max(1, blob.shape.c);
                DumpBufferToFile(path, physicalBuffer, logicalCount);
                return true;
            }
            catch
            {
                return false;
            }
        }

        return false;
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

    private ComputeBuffer UpdateLatentEulerAncestral(ComputeBuffer latentBuf, ComputeBuffer denoisedBuf, float sigma, float sigmaDown, float sigmaUp, ComputeBuffer noiseBuf, int width, int height)
    {
        var count = LatentElementCount(width, height);
        var diffBuf = _unetRepro.RentTempBuffer(count, sizeof(float));
        var dirBuf = _unetRepro.RentTempBuffer(count, sizeof(float));
        var deltaBuf = _unetRepro.RentTempBuffer(count, sizeof(float));
        var outBuf = _unetRepro.RentTempBuffer(count, sizeof(float));
        ComputeBuffer noiseScaled = null;
        try
        {
            _ops.BinaryOpBuf(latentBuf, denoisedBuf, count, 1, diffBuf);
            _ops.BinaryOpScalarBuf(diffBuf, sigma, count, 3, dirBuf);
            _ops.BinaryOpScalarBuf(dirBuf, sigmaDown - sigma, count, 2, deltaBuf);
            _ops.BinaryOpBuf(latentBuf, deltaBuf, count, 0, outBuf);

            if (noiseBuf != null && sigmaUp > 0f)
            {
                noiseScaled = _unetRepro.RentTempBuffer(count, sizeof(float));
                var withNoise = _unetRepro.RentTempBuffer(count, sizeof(float));
                try
                {
                    _ops.BinaryOpScalarBuf(noiseBuf, sigmaUp, count, 2, noiseScaled);
                    _ops.BinaryOpBuf(outBuf, noiseScaled, count, 0, withNoise);
                    _unetRepro.ReturnTempBuffer(outBuf);
                    outBuf = withNoise;
                    withNoise = null;
                }
                finally
                {
                    if (withNoise != null)
                        _unetRepro.ReturnTempBuffer(withNoise);
                }
            }

            return outBuf;
        }
        finally
        {
            _unetRepro.ReturnTempBuffer(diffBuf);
            _unetRepro.ReturnTempBuffer(dirBuf);
            _unetRepro.ReturnTempBuffer(deltaBuf);
            if (noiseScaled != null)
                _unetRepro.ReturnTempBuffer(noiseScaled);
        }
    }

    private async UniTask<Texture2D> DecodeLatentAsync(ComputeBuffer latentBuf, int width, int height, CancellationToken ct)
    {
        var latentW = Mathf.Max(1, width / 8);
        var latentH = Mathf.Max(1, height / 8);
        ComputeBuffer scaledLatent = null;
        RenderTexture inputTex = null;
        RenderTexture decodedTex = null;
        RenderTexture clippedTex = null;
        RenderTexture rgbRt = null;
        try
        {
            scaledLatent = _decoderRepro.RentTempBuffer(latentBuf.count, sizeof(float));
            _ops.CopyBuf(latentBuf, scaledLatent, latentBuf.count);
            _ops.MulScalarInplace(scaledLatent, InvLatentScale, scaledLatent.count);

            inputTex = _decoderRepro.RentTempArray(latentW, latentH, 1, RenderTextureFormat.ARGBHalf);
            _ops.FillPack4FromBufferCHW(scaledLatent, latentW, latentH, LatentChannels, inputTex);

            using (var infer = _decoderRepro.Infer(inputTex, 1, DecoderInputBlobName))
            {
                DumpLayerRuntimeProfile(_decoderRepro, "layer_profile_decoder");
                decodedTex = infer.ExtractTexture(DecoderOutputBlobName);
            }

            if (decodedTex == null)
                return null;

            clippedTex = _decoderRepro.RentTempArray(decodedTex.width, decodedTex.height, 1, RenderTextureFormat.ARGBHalf);
            _ops.ClipPack4(decodedTex, -1f, 1f, 1, clippedTex);
            rgbRt = GetTemporaryRt(decodedTex.width, decodedTex.height, RenderTextureFormat.ARGB32, true);
            _ops.Pack4ToRgb01(clippedTex, rgbRt, true);

            await UniTask.Yield();
            return RenderTextureToTexture2D(rgbRt, width, height);
        }
        finally
        {
            if (scaledLatent != null)
                _decoderRepro?.ReturnTempBuffer(scaledLatent);
            if (inputTex != null)
                _decoderRepro?.ReturnTempArray(inputTex);
            if (decodedTex != null)
                _decoderRepro?.ReturnTempArray(decodedTex);
            if (clippedTex != null)
                _decoderRepro?.ReturnTempArray(clippedTex);
            if (rgbRt != null)
                ReleaseTemporaryRt(rgbRt);
        }
    }

    private void EnsureRuntimeObjects()
    {
        _ops ??= new NcnnOps();
        _clipRepro ??= NcnnInferenceSessionFactory.Create(_ops);
        _unetRepro ??= NcnnInferenceSessionFactory.Create(_ops);
        _decoderRepro ??= NcnnInferenceSessionFactory.Create(_ops);
    }

    private void ApplyCommonOptions(NcnnRepro repro)
    {
        if (repro == null)
            return;
        repro.ForceBufferConvolutionAll = false;
        repro.ForceBufferBinaryOpAll = false;
        repro.ForceBufferGeluAll = false;
        repro.EnableGpuGeluBufferPath = true;
        repro.EnableConv1x1TextureConvolution = true;
        repro.EnableDepthWiseTextureConvolution = true;
        repro.EnableGeneralTextureConvolution = true;
        repro.EnableGroupNormTexturePath = true;
        repro.KeepRawConvWeightsForTexturePath = keepRawConvWeightsForTexturePath;
        repro.EnableMhaParallelSoftmax = ResolveMhaParallelSoftmax();
        repro.EnableMhaQkvFusion = ResolveMhaQkvFusion();
        repro.TensorTextureFormat = tensorTextureFormat;
        repro.UseNcnnStyleGroupNorm = useNcnnStyleGroupNorm;
        repro.LayerRuntimeProfileEnabled = enableLayerRuntimeProfile || ResolveBoolEnv("AIIMAGE_NCNN_PROFILE_LAYERS", false);
        repro.LayerRuntimeProfileSyncGpu = syncLayerRuntimeProfileGpu || ResolveBoolEnv("AIIMAGE_NCNN_PROFILE_SYNC", false);
        repro.DisallowBufferAccess = disallowBufferAccess;
        repro.DisallowBufferOutputs = disallowBufferOutputs;
        repro.DisallowBufferToTextureMaterialization = disallowBufferToTextureMaterialization;
    }

    private bool ResolveEncoderForceBufferConvolution()
    {
        return encoderForceBufferConvolutionAll || ResolveBoolEnv("AIIMAGE_SD_ENCODER_FORCE_BUFFER_CONV", false);
    }

    private bool ResolveMhaParallelSoftmax()
    {
        return ResolveBoolEnv("AIIMAGE_SD_MHA_PARALLEL_SOFTMAX", enableMhaParallelSoftmax);
    }

    private bool ResolveMhaQkvFusion()
    {
        return ResolveBoolEnv("AIIMAGE_SD_MHA_QKV_FUSION", enableMhaQkvFusion);
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

    private void Release()
    {
        try { _clipRepro?.Dispose(); } catch { }
        try { _unetRepro?.Dispose(); } catch { }
        try { _decoderRepro?.Dispose(); } catch { }
        try { _encoderRepro?.Dispose(); } catch { }
        try { _ops?.Dispose(); } catch { }
        _clipRepro = null;
        _unetRepro = null;
        _decoderRepro = null;
        _encoderRepro = null;
        _tokenizer = null;
        _loadedClipKey = null;
        _loadedSpatialKey = null;
        _logSigmas = null;
        _resolvedPaths = null;
        _ops = null;
    }

    private ResolvedPaths ResolvePaths()
    {
        if (_resolvedPaths.HasValue)
            return _resolvedPaths.Value;

        var clipParam = FindExactFile("FrozenCLIPEmbedder-fp16.param");
        var clipBin = FindExactFile("FrozenCLIPEmbedder-fp16.bin");
        var unetBin = FindExactFile("UNetModel-MHA-fp16.bin") ?? FindFileByTokens(".bin", new[] { "UNetModel" }, null, new[] { "MHA" });
        var decoderBin = FindExactFile("AutoencoderKL-fp16.bin") ?? FindFileByTokens(".bin", new[] { "AutoencoderKL" }, new[] { "encoder" }, null);
        var encoderBin = FindExactFile("AutoencoderKL-encoder-512-512-fp16.bin") ?? FindFileByTokens(".bin", new[] { "AutoencoderKL", "encoder" }, null, null);
        var vocab = FindExactFile("vocab.txt");
        var logSigmas = FindExactFile("log_sigmas.bin");

        if (string.IsNullOrWhiteSpace(clipParam))
            throw new FileNotFoundException("FrozenCLIPEmbedder-fp16.param not found under StableDiffusion search roots.");
        if (string.IsNullOrWhiteSpace(clipBin))
            throw new FileNotFoundException("FrozenCLIPEmbedder-fp16.bin not found under StableDiffusion search roots.");
        if (string.IsNullOrWhiteSpace(unetBin))
            throw new FileNotFoundException("UNetModel-MHA-fp16.bin not found under StableDiffusion search roots.");
        if (string.IsNullOrWhiteSpace(decoderBin))
            throw new FileNotFoundException("AutoencoderKL-fp16.bin not found under StableDiffusion search roots.");
        if (string.IsNullOrWhiteSpace(vocab))
            throw new FileNotFoundException("vocab.txt not found under StableDiffusion search roots.");
        if (string.IsNullOrWhiteSpace(logSigmas))
            throw new FileNotFoundException("log_sigmas.bin not found under StableDiffusion search roots.");

        _resolvedPaths = new ResolvedPaths(clipParam, clipBin, unetBin, decoderBin, encoderBin, vocab, logSigmas);
        return _resolvedPaths.Value;
    }

    private string FindParamFile(params string[] candidateNames)
    {
        for (var i = 0; i < candidateNames.Length; i++)
        {
            var candidate = candidateNames[i];
            if (string.IsNullOrWhiteSpace(candidate))
                continue;
            var exact = FindExactFile(candidate);
            if (!string.IsNullOrWhiteSpace(exact))
                return exact;
        }

        return null;
    }

    private string FindExactFile(string fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            return null;

        foreach (var root in EnumerateSearchRoots())
        {
            var path = Path.Combine(root, fileName);
            if (File.Exists(path))
                return path;
        }

        return null;
    }

    private string FindFileByTokens(string extension, string[] includeTokens, string[] excludeTokens, string[] preferredTokens)
    {
        foreach (var root in EnumerateSearchRoots())
        {
            if (!Directory.Exists(root))
                continue;

            var files = Directory.GetFiles(root, "*" + extension, SearchOption.TopDirectoryOnly);
            var matches = new List<string>(files.Length);
            for (var i = 0; i < files.Length; i++)
            {
                var name = Path.GetFileName(files[i]);
                if (includeTokens != null)
                {
                    var missing = false;
                    for (var t = 0; t < includeTokens.Length; t++)
                    {
                        if (name.IndexOf(includeTokens[t], StringComparison.OrdinalIgnoreCase) < 0)
                        {
                            missing = true;
                            break;
                        }
                    }
                    if (missing)
                        continue;
                }

                if (excludeTokens != null)
                {
                    var excluded = false;
                    for (var t = 0; t < excludeTokens.Length; t++)
                    {
                        if (name.IndexOf(excludeTokens[t], StringComparison.OrdinalIgnoreCase) >= 0)
                        {
                            excluded = true;
                            break;
                        }
                    }
                    if (excluded)
                        continue;
                }

                matches.Add(files[i]);
            }

            if (matches.Count == 0)
                continue;

            if (preferredTokens != null && preferredTokens.Length > 0)
            {
                for (var i = 0; i < matches.Count; i++)
                {
                    var name = Path.GetFileName(matches[i]);
                    var preferred = true;
                    for (var t = 0; t < preferredTokens.Length; t++)
                    {
                        if (name.IndexOf(preferredTokens[t], StringComparison.OrdinalIgnoreCase) < 0)
                        {
                            preferred = false;
                            break;
                        }
                    }

                    if (preferred)
                        return matches[i];
                }
            }

            matches.Sort((a, b) => string.CompareOrdinal(Path.GetFileName(a), Path.GetFileName(b)));
            return matches[0];
        }

        return null;
    }

    private IEnumerable<string> EnumerateSearchRoots()
    {
        var streamingRoot = Path.Combine(Application.streamingAssetsPath, stableDiffusionRootRelativePath);
        yield return streamingRoot;

        if (!useReferenceAssetFallback)
            yield break;

        var projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (string.IsNullOrWhiteSpace(projectRoot))
            yield break;

        yield return Path.Combine(projectRoot, "ref", "Stable-Diffusion-NCNN-main", "Windows", "Binary", "x64", "assets");
        yield return Path.Combine(projectRoot, "ref", "Stable-Diffusion-NCNN-main", "x86", "exe", "assets");
        yield return Path.Combine(projectRoot, "ref", "Stable-Diffusion-NCNN-main", "x86", "linux", "assets");
    }

    private static int LatentElementCount(int width, int height)
    {
        return Mathf.Max(1, width / 8) * Mathf.Max(1, height / 8) * LatentChannels;
    }

    private float[] BuildSigmaSchedule(int stepCount)
    {
        if (_logSigmas == null || _logSigmas.Length != 1000 || stepCount <= 0)
            return null;

        var sigma = new float[stepCount + 1];
        var delta = 0f - 999f / Mathf.Max(1, stepCount - 1);
        for (var i = 0; i < stepCount; i++)
        {
            var t = 999f + i * delta;
            var lowIdx = Mathf.Clamp(Mathf.FloorToInt(t), 0, 999);
            var highIdx = Mathf.Clamp(Mathf.CeilToInt(t), 0, 999);
            var w = t - lowIdx;
            sigma[i] = Mathf.Exp((1f - w) * _logSigmas[lowIdx] + w * _logSigmas[highIdx]);
        }
        sigma[stepCount] = 0f;
        return sigma;
    }

    private float SigmaToT(float sigma)
    {
        var logSigma = Mathf.Log(Mathf.Max(sigma, 1e-12f));
        var lowIdx = 0;
        for (var i = 0; i < 999; i++)
        {
            if (logSigma - _logSigmas[i] >= 0f)
                lowIdx = i;
        }

        var highIdx = Mathf.Min(lowIdx + 1, 999);
        var low = _logSigmas[lowIdx];
        var high = _logSigmas[highIdx];
        var denom = low - high;
        var w = Mathf.Abs(denom) > 1e-12f ? (low - logSigma) / denom : 0f;
        w = Mathf.Clamp01(w);
        return (1f - w) * lowIdx + w * highIdx;
    }

    private static int ResolveSeed(int seed)
    {
        if (seed != 0)
            return seed;
        return Environment.TickCount & int.MaxValue;
    }

    private int ResolveStepNoiseSeed(int baseSeed, int stepIndex)
    {
        if (deterministicAncestralNoise)
            return NormalizeSeed(baseSeed + 7919 * (stepIndex + 1));
        return NormalizeSeed(Environment.TickCount + stepIndex);
    }

    private static int NormalizeSeed(int seed)
    {
        var mod = seed % 1000;
        if (mod < 0)
            mod += 1000;
        return mod;
    }

    private float[] GenerateInitialGaussian(int count, int seed)
    {
        return GenerateGaussianByMode(count, NormalizeSeed(seed));
    }

    private float[] GenerateStepGaussian(int count, int baseSeed, int stepIndex)
    {
        return GenerateGaussianByMode(count, ResolveStepNoiseSeed(baseSeed, stepIndex));
    }

    private float[] GenerateGaussianByMode(int count, int seed)
    {
        return useOfficialNoise
            ? GenerateOfficialGaussian(count, seed)
            : GenerateGaussian(count, seed);
    }

    private static float[] GenerateGaussian(int count, int seed)
    {
        var data = new float[Mathf.Max(0, count)];
        var rng = new System.Random(seed);
        for (var i = 0; i < data.Length; i += 2)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();
            var radius = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-12)));
            var theta = 2.0 * Math.PI * u2;
            data[i] = (float)(radius * Math.Cos(theta));
            if (i + 1 < data.Length)
                data[i + 1] = (float)(radius * Math.Sin(theta));
        }

        return data;
    }

    private static float[] GenerateOfficialGaussian(int count, int seed)
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
                x = hz * _openCvNormalWn[iz];

                var absHz = hz < 0 ? unchecked((uint)-hz) : (uint)hz;
                if (absHz < _openCvNormalKn[iz])
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
                if (_openCvNormalFn[iz] + wedgeY * (_openCvNormalFn[iz - 1] - _openCvNormalFn[iz]) < (float)Math.Exp(-0.5f * x * x))
                    break;
            }

            data[i] = x;
        }

        return data;
    }

    private static readonly uint[] _openCvNormalKn = new uint[128];
    private static readonly float[] _openCvNormalWn = new float[128];
    private static readonly float[] _openCvNormalFn = new float[128];
    private static bool _openCvNormalTablesReady;

    private static void EnsureOpenCvNormalTables()
    {
        if (_openCvNormalTablesReady)
            return;

        const double m1 = 2147483648.0;
        var dn = 3.442619855899;
        var tn = dn;
        const double vn = 9.91256303526217e-3;

        var q = vn / Math.Exp(-0.5 * dn * dn);
        _openCvNormalKn[0] = (uint)((dn / q) * m1);
        _openCvNormalKn[1] = 0;
        _openCvNormalWn[0] = (float)(q / m1);
        _openCvNormalWn[127] = (float)(dn / m1);
        _openCvNormalFn[0] = 1f;
        _openCvNormalFn[127] = (float)Math.Exp(-0.5 * dn * dn);

        for (var i = 126; i >= 1; i--)
        {
            dn = Math.Sqrt(-2.0 * Math.Log(vn / dn + Math.Exp(-0.5 * dn * dn)));
            _openCvNormalKn[i + 1] = (uint)((dn / tn) * m1);
            tn = dn;
            _openCvNormalFn[i] = (float)Math.Exp(-0.5 * dn * dn);
            _openCvNormalWn[i] = (float)(dn / m1);
        }

        _openCvNormalTablesReady = true;
    }

    private static ulong OpenCvRngNext(ulong state)
    {
        return unchecked((ulong)(uint)state * 4164903690UL + (state >> 32));
    }

    private static ComputeBuffer NewFloatBuffer(float[] data)
    {
        var safe = data ?? Array.Empty<float>();
        var buffer = new ComputeBuffer(Mathf.Max(1, safe.Length), sizeof(float), ComputeBufferType.Structured);
        if (safe.Length > 0)
            buffer.SetData(safe);
        else
            buffer.SetData(new[] { 0f });
        return buffer;
    }

    private static float[] LoadFloatArray(string path, int expectedCount)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length % sizeof(float) != 0)
            throw new InvalidDataException("Float binary length invalid: " + path);
        var count = bytes.Length / sizeof(float);
        if (expectedCount > 0 && count != expectedCount)
            throw new InvalidDataException("Float binary count mismatch: " + path + " | " + count + " vs " + expectedCount);
        var data = new float[count];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return data;
    }

    private static ComputeBuffer CreateEncoderInputBufferNcnn(Texture src, int width, int height)
    {
        if (src == null)
            throw new ArgumentNullException(nameof(src));
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width));

        Texture2D tempTexture = null;
        try
        {
            var pixels = ReadTexturePixels32(src, out var srcW, out var srcH, out tempTexture);
            var input = CreateEncoderInputChwNcnn(pixels, srcW, srcH, width, height);
            return NewFloatBuffer(input);
        }
        finally
        {
            if (tempTexture != null)
                UnityEngine.Object.DestroyImmediate(tempTexture);
        }
    }

    private static Color32[] ReadTexturePixels32(Texture src, out int width, out int height, out Texture2D tempTexture)
    {
        tempTexture = null;
        width = src != null ? src.width : 0;
        height = src != null ? src.height : 0;
        if (src == null || width <= 0 || height <= 0)
            return Array.Empty<Color32>();

        if (src is Texture2D tex && tex.isReadable)
            return tex.GetPixels32();

        var rt = GetTemporaryRt(width, height, RenderTextureFormat.ARGB32, false);
        try
        {
            Graphics.Blit(src, rt);
            tempTexture = RenderTextureToTexture2D(rt, width, height);
            return tempTexture != null ? tempTexture.GetPixels32() : Array.Empty<Color32>();
        }
        finally
        {
            ReleaseTemporaryRt(rt);
        }
    }

    private static float[] CreateEncoderInputChwNcnn(Color32[] pixelsBottomUp, int srcW, int srcH, int dstW, int dstH)
    {
        if (pixelsBottomUp == null || pixelsBottomUp.Length < srcW * srcH)
            throw new InvalidDataException("Source texture pixels are unavailable.");
        if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0)
            throw new InvalidDataException("Invalid Stable Diffusion encoder input size.");

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
        if (src == null || dst == null)
            throw new ArgumentNullException(src == null ? nameof(src) : nameof(dst));
        if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0)
            throw new InvalidDataException("Invalid resize dimensions.");

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

    private static string[] ResolveDebugEncoderBlobNames()
    {
        try
        {
            var listEnv = Environment.GetEnvironmentVariable("AIIMAGE_SD_ENCODER_BLOBS");
            if (!string.IsNullOrWhiteSpace(listEnv))
                return ParseDebugEncoderBlobList(listEnv);

            var env = Environment.GetEnvironmentVariable("AIIMAGE_SD_DUMP_ENCODER_BLOBS");
            if (string.IsNullOrWhiteSpace(env))
                return Array.Empty<string>();
            env = env.Trim();
            if (string.Equals(env, "1", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "yes", StringComparison.OrdinalIgnoreCase))
                return DebugEncoderBlobNames;
        }
        catch
        {
        }

        return Array.Empty<string>();
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

    private static string[] ParseDebugEncoderBlobList(string value)
    {
        return ParseDebugBlobList(value, EncoderMeanBlobName, EncoderStdBlobName);
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

    private static RenderTexture ResizeTextureBilinear(Texture src, int width, int height)
    {
        if (src == null)
            return null;
        var rt = GetTemporaryRt(width, height, RenderTextureFormat.ARGB32, false);
        Graphics.Blit(src, rt);
        return rt;
    }

    private static Texture2D RenderTextureToTexture2D(RenderTexture rt, int width, int height)
    {
        if (rt == null)
            return null;
        var prev = RenderTexture.active;
        try
        {
            RenderTexture.active = rt;
            var tex = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            tex.Apply(false, false);
            return tex;
        }
        finally
        {
            RenderTexture.active = prev;
        }
    }

    private (ComputeBuffer maskBuf, ComputeBuffer invMaskBuf, Texture2D resizedMaskTex) CreateLatentMaskBuffers(Texture maskImage, int width, int height)
    {
        if (maskImage == null)
            return default;

        var latentW = Mathf.Max(1, width / 8);
        var latentH = Mathf.Max(1, height / 8);
        var resizedMask = ReadResizedTexture(maskImage, width, height);
        if (resizedMask == null)
            return default;

        var latentMask = ReadResizedTexture(maskImage, latentW, latentH);
        if (latentMask == null)
        {
            UnityEngine.Object.DestroyImmediate(resizedMask);
            return default;
        }

        try
        {
            var pixels = latentMask.GetPixels32();
            var count = LatentElementCount(width, height);
            var maskData = new float[count];
            var invData = new float[count];
            for (var y = 0; y < latentH; y++)
            {
                for (var x = 0; x < latentW; x++)
                {
                    var p = pixels[y * latentW + x];
                    var alpha = SampleMaskWeight(p);
                    var baseIndex = (y * latentW + x) * LatentChannels;
                    for (var c = 0; c < LatentChannels; c++)
                    {
                        maskData[baseIndex + c] = alpha;
                        invData[baseIndex + c] = 1f - alpha;
                    }
                }
            }

            return (NewFloatBuffer(maskData), NewFloatBuffer(invData), resizedMask);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(latentMask);
        }
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
            ReleaseTemporaryRt(rt);
        }
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
            var mp = maskPixels[i];
            var alpha = SampleMaskWeight(mp);
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
        return result;
    }

    private static float SampleMaskWeight(Color32 pixel)
    {
        var luminance = ((pixel.r + pixel.g + pixel.b) / 3f) / 255f;
        var alpha = pixel.a / 255f;
        return Mathf.Clamp01(luminance * alpha);
    }

    private static RenderTexture GetTemporaryRt(int width, int height, RenderTextureFormat format, bool enableRandomWrite)
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
        return rt;
    }

    private static void ReleaseTemporaryRt(RenderTexture rt)
    {
        if (rt == null)
            return;
        try { RenderTexture.ReleaseTemporary(rt); } catch { }
    }

    private static void DisposeBuffer(ComputeBuffer buffer)
    {
        if (buffer == null)
            return;
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

    private void LogStageTiming(string stage, Stopwatch stopwatch)
    {
        if (syncStageTimings)
            SyncGpuForStageTiming();
        LogStageTiming(stage, stopwatch != null ? stopwatch.ElapsedMilliseconds : 0L);
    }

    private void LogStageTiming(string stage, long elapsedMs)
    {
        if (string.IsNullOrWhiteSpace(stage))
            stage = "stage";

        var line = stage + "\t" + elapsedMs.ToString(CultureInfo.InvariantCulture);
        UnityEngine.Debug.Log("[SD] timing " + line);
        if (string.IsNullOrWhiteSpace(_lastDumpDir))
            return;

        try
        {
            Directory.CreateDirectory(_lastDumpDir);
            File.AppendAllText(Path.Combine(_lastDumpDir, "stage_timings.tsv"), line + Environment.NewLine);
        }
        catch
        {
        }
    }

    private void SyncGpuForStageTiming()
    {
        try
        {
            _ops?.DebugSyncGpu();
        }
        catch
        {
        }
    }

    private void DumpLayerRuntimeProfile(NcnnRepro repro, string prefix)
    {
        if (repro == null || repro.LastRuntimeProfile == null)
            return;

        var text = repro.FormatLastLayerRuntimeProfile(Mathf.Max(1, layerRuntimeProfileTopN));
        if (string.IsNullOrWhiteSpace(text))
            return;

        var profile = repro.LastRuntimeProfile;
        var safePrefix = SanitizeFileName(string.IsNullOrWhiteSpace(prefix) ? "layer_profile" : prefix);
        var fileName = safePrefix + "_" + profile.inferenceIndex.ToString("000", CultureInfo.InvariantCulture) + ".tsv";
        UnityEngine.Debug.Log("[SD] layer profile " + fileName + " | totalMs=" + profile.totalMs.ToString("0.###", CultureInfo.InvariantCulture) + " | syncGpu=" + (profile.syncGpu ? "1" : "0"));

        if (string.IsNullOrWhiteSpace(_lastDumpDir))
            return;

        try
        {
            Directory.CreateDirectory(_lastDumpDir);
            File.WriteAllText(Path.Combine(_lastDumpDir, fileName), text);
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
            UnityEngine.Debug.Log("[SD] " + label + " load complete");
            return;
        }

        if (progress.stage == "layer")
        {
            if (progress.layerIndex <= 1 || progress.layerIndex == progress.layerCount || (progress.layerIndex % 50) == 0)
            {
                UnityEngine.Debug.Log("[SD] " + label + " load layer " + progress.layerIndex.ToString(CultureInfo.InvariantCulture) + "/" + progress.layerCount.ToString(CultureInfo.InvariantCulture) + " | " + (progress.layerName ?? "") + " | " + (progress.layerType ?? ""));
            }

            return;
        }

        UnityEngine.Debug.Log("[SD] " + label + " load stage=" + progress.stage + " | " + progress.layerIndex.ToString(CultureInfo.InvariantCulture) + "/" + progress.layerCount.ToString(CultureInfo.InvariantCulture));
    }

    private static void LogLoadProfile(string label, NcnnRepro.ModelLoadProfile profile)
    {
        if (profile == null)
            return;
        if (string.IsNullOrWhiteSpace(label))
            label = "model";

        var items = new List<KeyValuePair<string, NcnnRepro.LayerTypeLoadProfile>>(profile.layerTypes);
        items.Sort((a, b) => b.Value.totalMs.CompareTo(a.Value.totalMs));
        var top = new List<string>();
        for (var i = 0; i < Math.Min(6, items.Count); i++)
        {
            var item = items[i];
            top.Add(item.Key
                + ":" + item.Value.totalMs.ToString(CultureInfo.InvariantCulture) + "ms"
                + " read=" + item.Value.readMs.ToString(CultureInfo.InvariantCulture)
                + " upload=" + item.Value.uploadMs.ToString(CultureInfo.InvariantCulture)
                + " pack=" + item.Value.packMs.ToString(CultureInfo.InvariantCulture)
                + " count=" + item.Value.count.ToString(CultureInfo.InvariantCulture));
        }

        UnityEngine.Debug.Log("[SD] LoadProfile " + label
            + " | totalMs=" + profile.totalMs.ToString(CultureInfo.InvariantCulture)
            + " | releaseMs=" + profile.releaseMs.ToString(CultureInfo.InvariantCulture)
            + " | parseMs=" + profile.parseParamMs.ToString(CultureInfo.InvariantCulture)
            + " | buildBlobUseMs=" + profile.buildBlobUseCountMs.ToString(CultureInfo.InvariantCulture)
            + " | bytesRead=" + profile.totalBytesRead.ToString(CultureInfo.InvariantCulture)
            + " | top=" + string.Join(" ; ", top));
    }

    private static string GenerateUnetParamText(string baseText, int height, int width)
    {
        var lines = baseText.Replace("\r\n", "\n").Split('\n');
        var sb = new StringBuilder(baseText.Length + 512);
        var reshapeCount = 0;
        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            if (line.StartsWith("Reshape", StringComparison.Ordinal))
            {
                switch (reshapeCount)
                {
                    case 0: line = line.Substring(0, line.Length - 4) + (width * height / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 1: line = line.Substring(0, line.Length - 7) + (width / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 2: line = line.Substring(0, line.Length - 4) + (width * height / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 3: line = line.Substring(0, line.Length - 7) + (width / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 4: line = line.Substring(0, line.Length - 4) + (width * height / 2 / 2 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 5: line = line.Substring(0, line.Length - 7) + (width / 2 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 2 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 6: line = line.Substring(0, line.Length - 4) + (width * height / 2 / 2 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 7: line = line.Substring(0, line.Length - 7) + (width / 2 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 2 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 8: line = line.Substring(0, line.Length - 3) + (width * height / 4 / 4 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 9: line = line.Substring(0, line.Length - 7) + (width / 4 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 4 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 10: line = line.Substring(0, line.Length - 3) + (width * height / 4 / 4 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 11: line = line.Substring(0, line.Length - 7) + (width / 4 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 4 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 12: line = line.Substring(0, line.Length - 2) + (width * height / 8 / 8 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 13: line = line.Substring(0, line.Length - 5) + (width / 8 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 14: line = line.Substring(0, line.Length - 3) + (width * height / 4 / 4 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 15: line = line.Substring(0, line.Length - 7) + (width / 4 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 4 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 16: line = line.Substring(0, line.Length - 3) + (width * height / 4 / 4 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 17: line = line.Substring(0, line.Length - 7) + (width / 4 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 4 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 18: line = line.Substring(0, line.Length - 3) + (width * height / 4 / 4 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 19: line = line.Substring(0, line.Length - 7) + (width / 4 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 4 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 20: line = line.Substring(0, line.Length - 4) + (width * height / 2 / 2 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 21: line = line.Substring(0, line.Length - 7) + (width / 2 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 2 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 22: line = line.Substring(0, line.Length - 4) + (width * height / 2 / 2 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 23: line = line.Substring(0, line.Length - 7) + (width / 2 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 2 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 24: line = line.Substring(0, line.Length - 4) + (width * height / 2 / 2 / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 25: line = line.Substring(0, line.Length - 7) + (width / 2 / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 2 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 26: line = line.Substring(0, line.Length - 4) + (width * height / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 27: line = line.Substring(0, line.Length - 7) + (width / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 28: line = line.Substring(0, line.Length - 4) + (width * height / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 29: line = line.Substring(0, line.Length - 7) + (width / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 30: line = line.Substring(0, line.Length - 4) + (width * height / 8 / 8).ToString(CultureInfo.InvariantCulture); break;
                    case 31: line = line.Substring(0, line.Length - 7) + (width / 8).ToString(CultureInfo.InvariantCulture) + " 2=" + (height / 8).ToString(CultureInfo.InvariantCulture); break;
                }

                reshapeCount++;
            }

            sb.AppendLine(line);
        }

        return sb.ToString();
    }

    private static string GenerateDecoderParamText(string baseText, int height, int width)
    {
        var lines = baseText.Replace("\r\n", "\n").Split('\n');
        var sb = new StringBuilder(baseText.Length + 256);
        var reshapeCount = 0;
        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            if (line.StartsWith("Reshape", StringComparison.Ordinal))
            {
                if (reshapeCount < 3)
                    line = line.Substring(0, line.Length - 12) + "0=" + (width * height / 8 / 8).ToString(CultureInfo.InvariantCulture) + " 1=512";
                else
                    line = line.Substring(0, line.Length - 15) + "0=" + (width / 8).ToString(CultureInfo.InvariantCulture) + " 1=" + (height / 8).ToString(CultureInfo.InvariantCulture) + " 2=512";
                reshapeCount++;
            }

            sb.AppendLine(line);
        }

        return sb.ToString();
    }

    private static string GenerateEncoderParamText(string baseText, int height, int width)
    {
        var lines = baseText.Replace("\r\n", "\n").Split('\n');
        var sb = new StringBuilder(baseText.Length + 128);
        var reshapeCount = 0;
        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            if (line.StartsWith("Reshape", StringComparison.Ordinal))
            {
                switch (reshapeCount)
                {
                    case 0:
                        line = line.Substring(0, line.Length - 12) + "0=" + (width * height / 8 / 8).ToString(CultureInfo.InvariantCulture) + " 1=512";
                        break;
                    case 1:
                        line = line.Substring(0, line.Length - 15) + "0=" + (width / 8).ToString(CultureInfo.InvariantCulture) + " 1=" + (height / 8).ToString(CultureInfo.InvariantCulture) + " 2=512";
                        break;
                }

                reshapeCount++;
            }

            sb.AppendLine(line);
        }

        return sb.ToString();
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
            var bytes = texture.EncodeToPNG();
            File.WriteAllBytes(Path.Combine(dir, fileName), bytes);
        }
        catch
        {
        }
    }

    private static void WriteFloatArray(string path, float[] data)
    {
        if (string.IsNullOrWhiteSpace(path) || data == null)
            return;
        var sb = new StringBuilder(data.Length * 12);
        for (var i = 0; i < data.Length; i++)
        {
            if (i > 0)
                sb.Append('\n');
            sb.Append(data[i].ToString("R", CultureInfo.InvariantCulture));
        }
        WriteAllTextSafe(path, sb.ToString());
    }

    private static void WriteIntArray(string path, int[] data)
    {
        if (string.IsNullOrWhiteSpace(path) || data == null)
            return;
        var sb = new StringBuilder(data.Length * 8);
        for (var i = 0; i < data.Length; i++)
        {
            if (i > 0)
                sb.Append('\n');
            sb.Append(data[i].ToString(CultureInfo.InvariantCulture));
        }
        WriteAllTextSafe(path, sb.ToString());
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
        WriteFloatArray(path, data);
    }

    private void TryDumpAnyBlob(NcnnRepro.InferResult infer, string blobName, string path, bool useUnetOwner = true, NcnnRepro ownerOverride = null)
    {
        if (infer == null || string.IsNullOrWhiteSpace(blobName) || string.IsNullOrWhiteSpace(path))
            return;

        try
        {
            var data = infer.ReadTextureDataForOutput(blobName);
            WriteFloatArray(path, data);
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
            using var physicalBuffer = new ComputeBuffer(tex.width * tex.height * channels, sizeof(float), ComputeBufferType.Structured);
            _ops.Pack4ToBufferCHW(tex, tex.width, tex.height, channels, physicalBuffer);
            var logicalCount = Mathf.Max(1, w) * Mathf.Max(1, h) * Mathf.Max(1, d) * Mathf.Max(1, c);
            DumpBufferToFile(path, physicalBuffer, logicalCount);
        }
        catch
        {
        }
    }

    private static void WriteAllTextSafe(string path, string content)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllText(path, content ?? string.Empty);
        }
        catch
        {
        }
    }

    private static string SanitizeFileName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "empty";
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(value.Length);
        for (var i = 0; i < value.Length; i++)
        {
            var ch = value[i];
            var replace = false;
            for (var j = 0; j < invalid.Length; j++)
            {
                if (ch == invalid[j])
                {
                    replace = true;
                    break;
                }
            }
            sb.Append(replace ? '_' : ch);
        }
        return sb.ToString();
    }

    private static string BoolText(bool value) => value ? "true" : "false";

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
        }

        public List<PromptChunk> TokenizePrompt(string prompt)
        {
            var parsed = ParsePromptAttention(prompt ?? string.Empty);
            var tokenized = new List<List<int>>(parsed.Count);
            for (var i = 0; i < parsed.Count; i++)
            {
                var parts = Split(parsed[i].text);
                var ids = new List<int>(parts.Count);
                for (var j = 0; j < parts.Count; j++)
                {
                    if (!_tokenToId.TryGetValue(parts[j], out var id))
                        id = 0;
                    ids.Add(id);
                }
                tokenized.Add(ids);
            }

            var remadeTokens = new List<int>(256);
            var multipliers = new List<float>(256);
            var lastComma = -1;
            for (var it = 0; it < tokenized.Count; it++)
            {
                var tokens = tokenized[it];
                var weight = parsed[it].weight;
                for (var i = 0; i < tokens.Count; i++)
                {
                    var token = tokens[i];
                    if (token == 267)
                    {
                        lastComma = remadeTokens.Count;
                    }
                    else if ((Mathf.Max(remadeTokens.Count, 1) % PromptChunkTokenCount == 0)
                             && lastComma != -1
                             && remadeTokens.Count - lastComma <= 20)
                    {
                        lastComma += 1;
                        var relocTokens = remadeTokens.GetRange(lastComma, remadeTokens.Count - lastComma);
                        var relocMults = multipliers.GetRange(lastComma, multipliers.Count - lastComma);
                        remadeTokens.RemoveRange(lastComma, remadeTokens.Count - lastComma);
                        multipliers.RemoveRange(lastComma, multipliers.Count - lastComma);
                        var rem = Mathf.CeilToInt(remadeTokens.Count / (float)PromptChunkTokenCount) * PromptChunkTokenCount - remadeTokens.Count;
                        for (var r = 0; r < rem; r++)
                        {
                            remadeTokens.Add(EndTokenId);
                            multipliers.Add(1f);
                        }
                        remadeTokens.AddRange(relocTokens);
                        multipliers.AddRange(relocMults);
                    }

                    remadeTokens.Add(token);
                    multipliers.Add(weight);
                }
            }

            var promptTargetLength = Mathf.CeilToInt(Mathf.Max(remadeTokens.Count, 1) / (float)PromptChunkTokenCount) * PromptChunkTokenCount;
            var tokensToAdd = promptTargetLength - remadeTokens.Count;
            for (var i = 0; i < tokensToAdd; i++)
            {
                remadeTokens.Add(EndTokenId);
                multipliers.Add(1f);
            }

            var chunks = new List<PromptChunk>(Mathf.Max(1, remadeTokens.Count / PromptChunkTokenCount));
            for (var offset = 0; offset < remadeTokens.Count; offset += PromptChunkTokenCount)
            {
                var tokens75 = new int[PromptChunkTokenCount];
                var mult75 = new float[PromptChunkTokenCount];
                for (var i = 0; i < PromptChunkTokenCount; i++)
                {
                    var srcIndex = offset + i;
                    tokens75[i] = srcIndex < remadeTokens.Count ? remadeTokens[srcIndex] : EndTokenId;
                    mult75[i] = srcIndex < multipliers.Count ? multipliers[srcIndex] : 1f;
                }
                chunks.Add(new PromptChunk(tokens75, mult75));
            }

            if (chunks.Count == 0)
                chunks.Add(EmptyChunk());
            return chunks;
        }

        public static PromptChunk EmptyChunk()
        {
            var tokens = new int[PromptChunkTokenCount];
            var mult = new float[PromptChunkTokenCount];
            for (var i = 0; i < PromptChunkTokenCount; i++)
            {
                tokens[i] = EndTokenId;
                mult[i] = 1f;
            }
            return new PromptChunk(tokens, mult);
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
}
