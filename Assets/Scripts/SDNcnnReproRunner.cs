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
    public bool enableTempPool = false;
    public int maxPooledPerShape = 0;
    public bool deterministicAncestralNoise = true;
    public bool enableDebugDump = false;

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
            await EnsureLoadedAsync(width, height, initImage != null, ct);

            ReportProgress(0.12f, "Encode prompts");
            var cond = await BuildConditioningAsync(positivePrompt ?? string.Empty, ct);
            var uncond = await BuildConditioningAsync(negativePrompt ?? string.Empty, ct);
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
                WriteAllTextSafe(
                    Path.Combine(_lastDumpDir, "run_config.txt"),
                    "width=" + width.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "height=" + height.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "steps=" + stepCount.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "seed=" + actualSeed.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                    + "used_init_image=" + BoolText(initImage != null) + Environment.NewLine
                    + "strength=" + strength.ToString("0.000000", CultureInfo.InvariantCulture));
            }

            ReportProgress(0.22f, initImage != null ? "Encode init image" : "Init latent");
            if (initImage != null)
            {
                latentBuf = await CreateImg2ImgLatentAsync(initImage, width, height, sigmas, actualSeed, Mathf.Clamp01(strength), ct);
                if (latentBuf == null)
                    return Finish(new SDNcnnReproResult { error = "img2img latent init failed.", seed = actualSeed, usedInitImage = true });
            }
            else
            {
                latentBuf = CreateTxt2ImgLatent(width, height, sigmas[0], actualSeed);
                if (latentBuf == null)
                    return Finish(new SDNcnnReproResult { error = "txt2img latent init failed.", seed = actualSeed, usedInitImage = false });
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

                var denoisedBuf = await RunCfgDenoiserAsync(latentBuf, width, height, sigma, condView, uncondView, ct);
                if (denoisedBuf == null)
                    return Finish(new SDNcnnReproResult { error = "UNet denoiser failed at step " + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), seed = actualSeed, usedInitImage = initImage != null });

                ComputeBuffer nextLatent = null;
                ComputeBuffer noiseBuf = null;
                try
                {
                    var sigmaUp = Mathf.Min(sigmaNext, Mathf.Sqrt(Mathf.Max(0f, sigmaNext * sigmaNext * (sigma * sigma - sigmaNext * sigmaNext) / Mathf.Max(sigma * sigma, 1e-12f))));
                    var sigmaDown = Mathf.Sqrt(Mathf.Max(0f, sigmaNext * sigmaNext - sigmaUp * sigmaUp));
                    if (sigmaUp > 0f)
                        noiseBuf = NewFloatBuffer(GenerateGaussian(LatentElementCount(width, height), ResolveStepNoiseSeed(actualSeed, stepIndex)));

                    nextLatent = UpdateLatentEulerAncestral(latentBuf, denoisedBuf, sigma, sigmaDown, sigmaUp, noiseBuf, width, height);
                    if (nextLatent == null)
                        return Finish(new SDNcnnReproResult { error = "Latent update failed at step " + (stepIndex + 1).ToString(CultureInfo.InvariantCulture), seed = actualSeed, usedInitImage = initImage != null });
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
            var finalTexture = await DecodeLatentAsync(latentBuf, width, height, ct);
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

    private async UniTask EnsureLoadedAsync(int width, int height, bool needEncoder, CancellationToken ct)
    {
        var paths = ResolvePaths();
        if (_logSigmas == null || _logSigmas.Length != 1000)
        {
            UnityEngine.Debug.Log("[SD] Load log_sigmas | path=" + paths.logSigmasPath);
            _logSigmas = LoadFloatArray(paths.logSigmasPath, 1000);
        }

        var clipKey = paths.clipParamPath + "|" + paths.clipBinPath + "|" + paths.vocabPath;
        if (!string.Equals(_loadedClipKey, clipKey, StringComparison.Ordinal))
        {
            UnityEngine.Debug.Log("[SD] Load CLIP | param=" + paths.clipParamPath + " | bin=" + paths.clipBinPath);
            _clipRepro?.Dispose();
            _clipRepro = new NcnnRepro(_ops);
            ApplyCommonOptions(_clipRepro);
            var clipParamText = File.ReadAllText(paths.clipParamPath);
            UnityEngine.Debug.Log("[SD] Open CLIP bin stream");
            using (var fs = new FileStream(paths.clipBinPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, false))
            using (var br = new NcnnBinReader(fs))
            {
                UnityEngine.Debug.Log("[SD] Begin CLIP LoadModel");
                _clipRepro.LoadModel(clipParamText, br, progress => LogLoadProgress("CLIP", progress));
            }

            _tokenizer = new StableDiffusionSimpleTokenizer(paths.vocabPath);
            _loadedClipKey = clipKey;
            UnityEngine.Debug.Log("[SD] CLIP loaded");
        }

        var spatialKey = width.ToString(CultureInfo.InvariantCulture) + "x" + height.ToString(CultureInfo.InvariantCulture) + "|" + BoolText(needEncoder) + "|" + paths.unetBinPath + "|" + paths.decoderBinPath + "|" + (paths.encoderBinPath ?? string.Empty);
        if (string.Equals(_loadedSpatialKey, spatialKey, StringComparison.Ordinal))
            return;

        _unetRepro?.Dispose();
        _decoderRepro?.Dispose();
        _encoderRepro?.Dispose();
        _unetRepro = new NcnnRepro(_ops);
        _decoderRepro = new NcnnRepro(_ops);
        _encoderRepro = needEncoder ? new NcnnRepro(_ops) : null;
        ApplyCommonOptions(_unetRepro);
        ApplyCommonOptions(_decoderRepro);
        if (_encoderRepro != null)
            ApplyCommonOptions(_encoderRepro);

        var modelInfo = await BuildSpatialModelInfoAsync(paths, width, height, needEncoder, ct);

        UnityEngine.Debug.Log("[SD] Load UNet | size=" + width.ToString(CultureInfo.InvariantCulture) + "x" + height.ToString(CultureInfo.InvariantCulture) + " | bin=" + paths.unetBinPath);
        UnityEngine.Debug.Log("[SD] Open UNet bin stream");
        using (var unetFs = new FileStream(paths.unetBinPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, false))
        using (var unetBr = new NcnnBinReader(unetFs))
        {
            UnityEngine.Debug.Log("[SD] Begin UNet LoadModel");
            _unetRepro.LoadModel(modelInfo.unetParamText, unetBr, progress => LogLoadProgress("UNet", progress));
        }
        UnityEngine.Debug.Log("[SD] UNet loaded");

        UnityEngine.Debug.Log("[SD] Load VAE decoder | bin=" + paths.decoderBinPath);
        UnityEngine.Debug.Log("[SD] Open VAE decoder bin stream");
        using (var decoderFs = new FileStream(paths.decoderBinPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, false))
        using (var decoderBr = new NcnnBinReader(decoderFs))
        {
            UnityEngine.Debug.Log("[SD] Begin VAE decoder LoadModel");
            _decoderRepro.LoadModel(modelInfo.decoderParamText, decoderBr, progress => LogLoadProgress("VAE-Decoder", progress));
        }
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
            tokens77[0] = StartTokenId;
            multipliers77[0] = 1f;
            Array.Copy(chunks[i].tokens75, 0, tokens77, 1, PromptChunkTokenCount);
            Array.Copy(chunks[i].multipliers75, 0, multipliers77, 1, PromptChunkTokenCount);
            tokens77[PromptChunkModelTokenCount - 1] = EndTokenId;
            multipliers77[PromptChunkModelTokenCount - 1] = 1f;

            using var tokenBuffer = new ComputeBuffer(tokens77.Length, sizeof(int), ComputeBufferType.Structured);
            using var multiplierBuffer = NewFloatBuffer(multipliers77);
            tokenBuffer.SetData(tokens77);
            var tokenView = new NcnnTensorBuffer(tokenBuffer, 1, tokens77.Length, 1, 1, 1, false);
            var multiplierView = new NcnnTensorBuffer(multiplierBuffer, 1, multipliers77.Length, 1, 1, 1, false);

            using var infer = _clipRepro.InferWithMultiInputs(
                null,
                new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal)
                {
                    { "token", tokenView },
                    { "multiplier", multiplierView },
                    { "cond", dummyCondView }
                },
                new HashSet<string>(StringComparer.Ordinal) { ClipOutputBlobName });

            var chunkData = infer.GetBufferData(ClipOutputBlobName);
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
        latent.SetData(GenerateGaussian(latent.count, NormalizeSeed(seed)));
        _ops.MulScalarInplace(latent, sigma0, latent.count);
        return latent;
    }

    private async UniTask<ComputeBuffer> CreateImg2ImgLatentAsync(Texture initImage, int width, int height, float[] sigmas, int seed, float strength, CancellationToken ct)
    {
        if (_encoderRepro == null)
            throw new InvalidOperationException("img2img requires encoder model.");
        if (initImage == null)
            throw new ArgumentNullException(nameof(initImage));

        RenderTexture resized = null;
        RenderTexture inputPack4 = null;
        RenderTexture stdTex = null;
        ComputeBuffer meanBuf = null;
        ComputeBuffer stdBuf = null;
        ComputeBuffer noiseBuf = null;
        ComputeBuffer mulBuf = null;
        ComputeBuffer latentBuf = null;
        try
        {
            resized = ResizeTextureBilinear(initImage, width, height);
            inputPack4 = _encoderRepro.RentTempArray(width, height, 1, RenderTextureFormat.ARGBHalf);
            _ops.PackRgbToPack4Gfpgan(resized, 0, 0, 1f, 1f, inputPack4);

            using (var infer = _encoderRepro.Infer(inputPack4, 1, EncoderInputBlobName, new HashSet<string>(StringComparer.Ordinal)
            {
                EncoderMeanBlobName,
                EncoderStdBlobName
            }))
            {
                meanBuf = infer.ExtractBuffer(EncoderMeanBlobName);
                try
                {
                    stdBuf = infer.ExtractBuffer(EncoderStdBlobName);
                }
                catch
                {
                    stdTex = infer.ExtractTexture(EncoderStdBlobName);
                }
            }

            if (stdBuf == null)
            {
                if (stdTex == null)
                    throw new InvalidOperationException("Encoder std output not found.");
                stdBuf = _encoderRepro.RentTempBuffer(LatentElementCount(width, height), sizeof(float));
                _ops.Pack4ToBufferCHW(stdTex, stdTex.width, stdTex.height, LatentChannels, stdBuf);
            }

            noiseBuf = NewFloatBuffer(GenerateGaussian(LatentElementCount(width, height), NormalizeSeed(seed)));
            mulBuf = _unetRepro.RentTempBuffer(noiseBuf.count, sizeof(float));
            latentBuf = _unetRepro.RentTempBuffer(noiseBuf.count, sizeof(float));
            _ops.BinaryOpBuf(stdBuf, noiseBuf, noiseBuf.count, 2, mulBuf);
            _ops.BinaryOpBuf(meanBuf, mulBuf, noiseBuf.count, 0, latentBuf);
            _ops.MulScalarInplace(latentBuf, LatentScale, latentBuf.count);

            var totalSteps = Mathf.Max(1, sigmas.Length - 1);
            var img2ImgStepCount = Mathf.Clamp(Mathf.FloorToInt(totalSteps * strength), 1, totalSteps);
            var sigmaIndex = Mathf.Clamp((sigmas.Length - 1) - img2ImgStepCount, 0, sigmas.Length - 1);
            var sigmaKick = sigmas[sigmaIndex];
            if (sigmaKick > 0f)
            {
                using var kickNoise = NewFloatBuffer(GenerateGaussian(LatentElementCount(width, height), NormalizeSeed(seed)));
                var kickScaled = _unetRepro.RentTempBuffer(kickNoise.count, sizeof(float));
                var kicked = _unetRepro.RentTempBuffer(kickNoise.count, sizeof(float));
                try
                {
                    _ops.BinaryOpScalarBuf(kickNoise, sigmaKick, kickNoise.count, 2, kickScaled);
                    _ops.BinaryOpBuf(latentBuf, kickScaled, kickNoise.count, 0, kicked);
                    _unetRepro.ReturnTempBuffer(latentBuf);
                    latentBuf = kicked;
                    kicked = null;
                }
                finally
                {
                    _unetRepro.ReturnTempBuffer(kickScaled);
                    if (kicked != null)
                        _unetRepro.ReturnTempBuffer(kicked);
                }
            }

            await UniTask.Yield();
            return latentBuf;
        }
        finally
        {
            if (resized != null)
                ReleaseTemporaryRt(resized);
            if (inputPack4 != null)
                _encoderRepro?.ReturnTempArray(inputPack4);
            if (stdTex != null)
                _encoderRepro?.ReturnTempArray(stdTex);
            if (meanBuf != null)
                _encoderRepro?.ReturnTempBuffer(meanBuf);
            if (stdBuf != null)
                _encoderRepro?.ReturnTempBuffer(stdBuf);
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

            condOut = RunUnetOnce(latentTex, timestepView, condView, cInView, cOutView, width, height);
            uncondOut = RunUnetOnce(latentTex, timestepView, uncondView, cInView, cOutView, width, height);
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
        }
    }

    private ComputeBuffer RunUnetOnce(RenderTexture latentTex, NcnnTensorBuffer timestepView, NcnnTensorBuffer condView, NcnnTensorBuffer cInView, NcnnTensorBuffer cOutView, int width, int height)
    {
        using var infer = _unetRepro.InferWithMultiInputs(
            new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
            {
                { "in0", latentTex }
            },
            new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal)
            {
                { "in1", timestepView },
                { "in2", condView },
                { "c_in", cInView },
                { "c_out", cOutView }
            });

        var outTex = infer.ExtractTexture(UnetOutputBlobName);
        if (outTex == null)
            return null;

        try
        {
            var outBuf = _unetRepro.RentTempBuffer(LatentElementCount(width, height), sizeof(float));
            _ops.Pack4ToBufferCHW(outTex, outTex.width, outTex.height, LatentChannels, outBuf);
            return outBuf;
        }
        finally
        {
            _unetRepro.ReturnTempArray(outTex);
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
                decodedTex = infer.ExtractTexture(DecoderOutputBlobName);
            }

            if (decodedTex == null)
                return null;

            clippedTex = _decoderRepro.RentTempArray(decodedTex.width, decodedTex.height, 1, RenderTextureFormat.ARGBHalf);
            _ops.ClipPack4(decodedTex, -1f, 1f, 1, clippedTex);
            rgbRt = GetTemporaryRt(decodedTex.width, decodedTex.height, RenderTextureFormat.ARGB32, false);
            _ops.Pack4ToRgb01(clippedTex, rgbRt);

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
        _clipRepro ??= new NcnnRepro(_ops);
        _unetRepro ??= new NcnnRepro(_ops);
        _decoderRepro ??= new NcnnRepro(_ops);
    }

    private void ApplyCommonOptions(NcnnRepro repro)
    {
        if (repro == null)
            return;
        repro.EnableTempPool = enableTempPool;
        repro.MaxPooledPerShape = maxPooledPerShape;
        repro.ForceBufferConvolutionAll = false;
        repro.ForceBufferBinaryOpAll = false;
        repro.ForceBufferGeluAll = false;
        repro.EnableConv1x1TextureConvolution = true;
        repro.EnableDepthWiseTextureConvolution = true;
        repro.TensorTextureFormat = RenderTextureFormat.ARGBHalf;
    }

    private void Release()
    {
        try { _clipRepro?.Dispose(); } catch { }
        try { _unetRepro?.Dispose(); } catch { }
        try { _decoderRepro?.Dispose(); } catch { }
        try { _encoderRepro?.Dispose(); } catch { }
        _clipRepro = null;
        _unetRepro = null;
        _decoderRepro = null;
        _encoderRepro = null;
        _tokenizer = null;
        _loadedClipKey = null;
        _loadedSpatialKey = null;
        _logSigmas = null;
        _resolvedPaths = null;
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

    private static RenderTexture GetTemporaryRt(int width, int height, RenderTextureFormat format, bool enableRandomWrite)
    {
        var rt = RenderTexture.GetTemporary(width, height, 0, format, RenderTextureReadWrite.Default);
        rt.enableRandomWrite = enableRandomWrite;
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.filterMode = FilterMode.Bilinear;
        rt.Create();
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
