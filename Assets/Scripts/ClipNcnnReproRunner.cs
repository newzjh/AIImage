using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Collections.Generic;
using System.Text;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;

public struct ClipLabelScore
{
    public string label;
    public string prompt;
    public float probability;
    public float similarity;
}

public struct ClipClassificationResult
{
    public string bestLabel;
    public float bestProbability;
    public ClipLabelScore[] scores;
    public float[] imageEmbedding;
    public string error;
    public long elapsedMs;
}

public sealed class ClipNcnnReproRunner : MonoBehaviour
{
    public enum ClipModelLevel
    {
        B,
        BLT,
        S0,
        S1,
        S2
    }

    [Serializable]
    public sealed class ClipLabelDefinition
    {
        public string label;
        public string prompt;
    }

    [Serializable]
    private sealed class ClipTextEmbeddingCache
    {
        public string modelKey;
        public ClipTextEmbeddingCacheEntry[] entries;
    }

    [Serializable]
    private sealed class ClipTextEmbeddingCacheEntry
    {
        public string label;
        public string prompt;
        public float[] embedding;
    }

    private const string InputBlobName = "in0";
    private const string OutputBlobName = "out0";
    private const int EmbeddingSize = 512;
    private const string TextEmbeddingCacheSuffix = ".label_embeddings.json";
    private const string ExtraDebugImageBlobEnvVar = "AIIMAGE_CLIP_EXTRA_DEBUG_IMAGE_BLOBS";
    private static readonly string[] DebugTextBlobNames = { "1", "3", "6", "7", "17", "18", "19", "22", "23", "24", "26", "27", "28", "29", "30", "57", "84", "111", "138", "165", "192", "219", "246", "273", "300", "320", "327", "out0" };
    private static readonly string[] DebugImageBlobNames = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "341", "343", "348", "349", "350", "351", "352", "353", "354", "355", "356", "357", "358", "359", "360", "361", "363", "366", "367", "368", "369", "370", "371", "372", "373", "376", "377", "378", "379", "380", "381", "382", "383", "384", "385", "386", "387", "388", "389", "390", "391", "392", "393", "396", "397", "398", "399", "414", "434", "449", "469", "484", "504", "519", "541", "out0" };
    private static readonly string[] DebugImageCompareLayers = { "convdw_253", "convdw_254", "conv_41", "conv_42", "convdw_255", "convdw_256", "conv_43", "conv_44", "convdw_257", "convdw_258", "conv_45", "conv_46", "convdw_259", "convrelu_0", "convsigmoid_3", "conv_49", "convdw_260", "convdw_261", "conv_50", "conv_51", "convdw_294", "convdw_295", "conv_84", "conv_85", "convdw_296", "convdw_297", "conv_86", "conv_87", "convdw_298", "convdw_299", "conv_88", "conv_89", "convdw_300" };

    public ClipModelLevel modelLevel = ClipModelLevel.S0;
    public string clipRootRelativePath = "Clip";
    public bool enableTempPool = true;
    public int maxPooledPerShape = 4;
    public bool enableDebugDump = false;
    public bool forceFullRenderTexturePath = true;
    public bool useCommandBuffer = false;
    public bool useAsyncComputeCommandBuffer = true;
    public bool enableGeneralTextureConvolution = true;
    public bool enableAttentionMatMulPack4Specializations = true;
    public bool disallowBufferAccess = false;
    public bool disallowBufferOutputs = false;
    public bool disallowBufferToTextureMaterialization = false;
    public bool logAllLayerHeartbeats = false;
    public bool logAllLayerOutputs = false;
    public bool logAllBufferMaterialize = false;
    public bool enableLayerRuntimeProfile = false;
    public bool layerRuntimeProfileSyncGpu = false;
    public bool forwardReproDebugLogToUnity = false;
    public ClipLabelDefinition[] labelDefinitions =
    {
        new ClipLabelDefinition { label = "Portrait", prompt = "a portrait photo" },
        new ClipLabelDefinition { label = "Landscape", prompt = "a landscape photo" },
        new ClipLabelDefinition { label = "Night", prompt = "a night photo" },
        new ClipLabelDefinition { label = "Food", prompt = "a photo of food" },
        new ClipLabelDefinition { label = "Pet", prompt = "a photo of a pet" },
        new ClipLabelDefinition { label = "Architecture", prompt = "an architecture photo" },
        new ClipLabelDefinition { label = "Document", prompt = "a photo of a document" },
        new ClipLabelDefinition { label = "Group", prompt = "a group photo" },
        new ClipLabelDefinition { label = "Photo", prompt = "a photo" }
    };

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnRepro _imageRepro;
    private NcnnRepro _textRepro;
    private NcnnRepro _projectionRepro;
    private MobileClipSimpleTokenizer _tokenizer;
    private string _loadedModelKey;
    private string _lastTextEmbeddingSource;
    private ClipLabelScore[] _cachedTextScores;
    private float[][] _cachedTextEmbeddings;
    private string _lastDumpDir;
    private string _lastLayerRuntimeProfileText;
    private List<string> _imageCompareLines;

    public string LastDumpDir => _lastDumpDir;
    public string ClassificationCacheSignature => BuildClassificationCacheSignature();

    private void Awake()
    {
        EnsureRuntimeObjects();
    }

    private void OnDestroy()
    {
        Release();
    }

    public async UniTask<ClipClassificationResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (src == null)
            return default;

        var totalSw = Stopwatch.StartNew();
        _lastDumpDir = null;
        _lastLayerRuntimeProfileText = null;

        ClipClassificationResult Finish(ClipClassificationResult result)
        {
            result.elapsedMs = totalSw.ElapsedMilliseconds;
            return result;
        }

        RenderTexture resized = null;
        RenderTexture inputPack4 = null;
        long loadMs = 0;
        long imageInferMs = 0;
        long scoreMs = 0;
        try
        {
            EnsureRuntimeObjects();
            ApplyReproOptions();
            if (enableDebugDump && string.IsNullOrWhiteSpace(_lastDumpDir))
                _lastDumpDir = CreateDumpDir();
            var ensureSw = Stopwatch.StartNew();
            await EnsureLoaded(ct);
            ensureSw.Stop();
            loadMs = ensureSw.ElapsedMilliseconds;
            if (_cachedTextEmbeddings == null || _cachedTextEmbeddings.Length == 0)
                return Finish(new ClipClassificationResult { error = "CLIP text embeddings unavailable" });

            var targetSize = ResolveInputSize();
            ReportProgress(0.72f, "Prepare input");
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();

            resized = ResizeTextureBilinear(src, targetSize, targetSize);
            if (resized == null)
                return Finish(new ClipClassificationResult { error = "Resize input failed" });

            ReportProgress(0.84f, "Encode image");
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();

            float[] imageEmbedding;
            var imageInferSw = Stopwatch.StartNew();
            if (useCommandBuffer)
            {
                imageEmbedding = await EncodeImageWithCommandBufferAsync(resized, targetSize, ct);
            }
            else
            {
                inputPack4 = _imageRepro.RentTempArray(targetSize, targetSize, 1, RenderTextureFormat.ARGBHalf);
                _ops.PackRgbToPack4(resized, 0, 0, 1f, 1f, inputPack4, true);
                if (enableDebugDump && !ShouldAvoidInferenceBufferReadback() && !string.IsNullOrWhiteSpace(_lastDumpDir))
                    DumpPack4TextureLogical(inputPack4, targetSize, targetSize, 3, Path.Combine(_lastDumpDir, "image_blob_in0.txt"));

                System.Collections.Generic.HashSet<string> pinnedImage = null;
                var debugImageBlobNames = GetDebugImageBlobNames();
                if (enableDebugDump)
                    pinnedImage = new System.Collections.Generic.HashSet<string>(debugImageBlobNames, StringComparer.Ordinal);

                using var infer = _imageRepro.Infer(inputPack4, 1, InputBlobName, pinnedImage);
                if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                {
                    for (var i = 0; i < debugImageBlobNames.Length; i++)
                        TryDumpAnyBlob(infer, debugImageBlobNames[i], Path.Combine(_lastDumpDir, "image_blob_" + debugImageBlobNames[i] + ".txt"));
                }
                if (ShouldUseTextureReadbackForImageOutput(infer))
                    _imageRepro?.Ops?.DebugSyncGpu();
                imageEmbedding = await ReadImageEmbeddingAsync(infer, ct);
            }
            imageInferSw.Stop();
            imageInferMs = imageInferSw.ElapsedMilliseconds;
            if (enableLayerRuntimeProfile)
            {
                _lastLayerRuntimeProfileText = _imageRepro?.FormatLastLayerRuntimeProfile(256);
                if (!string.IsNullOrWhiteSpace(_lastLayerRuntimeProfileText))
                {
                    try
                    {
                        UnityEngine.Debug.Log("[LAYER-PROFILE] CLIP(image)\n" + _lastLayerRuntimeProfileText);
                    }
                    catch
                    {
                    }
                }
            }

            if (imageEmbedding == null || imageEmbedding.Length != EmbeddingSize)
                return Finish(new ClipClassificationResult { error = "Image embedding missing or invalid" });
            NormalizeInPlace(imageEmbedding);
            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpVector(_lastDumpDir, "image_embedding.txt", imageEmbedding);

            ReportProgress(0.96f, "Score labels");
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();

            var scoreSw = Stopwatch.StartNew();
            var scores = new ClipLabelScore[_cachedTextEmbeddings.Length];
            var logits = new float[_cachedTextEmbeddings.Length];
            for (var i = 0; i < _cachedTextEmbeddings.Length; i++)
            {
                var sim = Dot(imageEmbedding, _cachedTextEmbeddings[i]) * 100f;
                logits[i] = sim;
                scores[i] = _cachedTextScores[i];
                scores[i].similarity = sim;
            }

            var probs = Softmax(logits);
            var bestIndex = 0;
            for (var i = 0; i < scores.Length; i++)
            {
                scores[i].probability = probs[i];
                if (scores[i].probability > scores[bestIndex].probability)
                    bestIndex = i;
            }

            Array.Sort(scores, (a, b) => b.probability.CompareTo(a.probability));
            scoreSw.Stop();
            scoreMs = scoreSw.ElapsedMilliseconds;

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                DumpScores(_lastDumpDir, modelLevel, scores);
                if (_imageCompareLines != null && _imageCompareLines.Count > 0)
                    File.WriteAllLines(Path.Combine(_lastDumpDir, "image_conv_compare.txt"), _imageCompareLines);
                if (!string.IsNullOrWhiteSpace(_lastLayerRuntimeProfileText))
                    File.WriteAllText(Path.Combine(_lastDumpDir, "layer_runtime_profile.tsv"), _lastLayerRuntimeProfileText);
            }

            UnityEngine.Debug.Log("[CLIP] Run | model=" + ResolveModelKey()
                + " | textSource=" + (_lastTextEmbeddingSource ?? "none")
                + " | imagePath=" + (useCommandBuffer ? (useAsyncComputeCommandBuffer ? "cmd-async" : "cmd-sync") : "immediate")
                + " | loadMs=" + loadMs
                + " | imageInferMs=" + imageInferMs
                + " | scoreMs=" + scoreMs
                + " | totalMs=" + totalSw.ElapsedMilliseconds
                + " | best=" + (scores.Length > 0 ? scores[0].label : ""));

            ReportProgress(1f, string.Empty);
            return Finish(new ClipClassificationResult
            {
                bestLabel = scores.Length > 0 ? scores[0].label : null,
                bestProbability = scores.Length > 0 ? scores[0].probability : 0f,
                scores = scores,
                imageEmbedding = (float[])imageEmbedding.Clone()
            });
        }
        catch (OperationCanceledException)
        {
            return Finish(new ClipClassificationResult { error = "Cancelled" });
        }
        catch (Exception e)
        {
            if (ShouldForwardReproDebugLogToUnity())
                UnityEngine.Debug.LogError("[CLIP] " + e);
            return Finish(new ClipClassificationResult { error = FormatExceptionSummary(e) });
        }
        finally
        {
            if (resized != null)
                ReleaseTemporaryRt(resized);
            if (inputPack4 != null)
                _imageRepro?.ReturnTempArray(inputPack4);
            ReportProgress(1f, string.Empty);
        }
    }

    private void EnsureRuntimeObjects()
    {
        _ops ??= new NcnnOps();
        _imageRepro ??= new NcnnRepro(_ops);
    }

    private void EnsureTextRuntimeObjects()
    {
        _ops ??= new NcnnOps();
        _textRepro ??= new NcnnRepro(_ops);
        _projectionRepro ??= new NcnnRepro(_ops);
    }

    private void EnsureTokenizer(string clipRoot)
    {
        if (_tokenizer != null)
            return;

        _tokenizer = new MobileClipSimpleTokenizer(
            Path.Combine(clipRoot, "vocab.txt"),
            Path.Combine(clipRoot, "bpe_simple_vocab_16e6.txt"));
    }

    private void ReleaseTextRuntime()
    {
        try { _textRepro?.Dispose(); } catch { }
        try { _projectionRepro?.Dispose(); } catch { }
        _textRepro = null;
        _projectionRepro = null;
        _tokenizer = null;
    }

    private void ApplyReproOptions()
    {
        ApplyOptions(_imageRepro);
        ApplyOptions(_textRepro);
        ApplyOptions(_projectionRepro);
        var imageUsesTexturePath = forceFullRenderTexturePath || useCommandBuffer;
        var strictImageReference = !imageUsesTexturePath;
        var strictTextReference = !forceFullRenderTexturePath && modelLevel == ClipModelLevel.S0;
        var strictProjectionReference = false;
        var allowGeneralTextureConvolution = enableGeneralTextureConvolution || imageUsesTexturePath;
        var allowAttentionMatMulPack4 = enableAttentionMatMulPack4Specializations || imageUsesTexturePath;
        var avoidInferenceBufferReadback = ShouldAvoidInferenceBufferReadback();
        var captureImageDebugLog = enableDebugDump;
        var enableAnyDebugLogSink = captureImageDebugLog || ShouldForwardReproDebugLogToUnity();
        if (_imageRepro != null)
        {
            ConfigureClipRepro(_imageRepro, strictImageReference, allowGeneralTextureConvolution, allowAttentionMatMulPack4);
            _imageRepro.DebugCompareTextureLayers = enableDebugDump && !avoidInferenceBufferReadback
                ? new HashSet<string>(DebugImageCompareLayers, StringComparer.Ordinal)
                : null;
            _imageCompareLines = captureImageDebugLog ? new List<string>() : null;
            _imageRepro.DebugLog = enableAnyDebugLogSink ? CreateReproDebugLogSink("image", captureImageDebugLog) : null;
        }
        if (_textRepro != null)
        {
            ConfigureClipRepro(_textRepro, strictTextReference, allowGeneralTextureConvolution, allowAttentionMatMulPack4);
            _textRepro.DebugLog = enableAnyDebugLogSink ? CreateReproDebugLogSink("text", captureForDump: false) : null;
        }
        if (_projectionRepro != null)
        {
            ConfigureClipRepro(_projectionRepro, strictProjectionReference, allowGeneralTextureConvolution, allowAttentionMatMulPack4);
            _projectionRepro.DebugLog = enableAnyDebugLogSink ? CreateReproDebugLogSink("projection", captureForDump: false) : null;
        }
    }

    private void ApplyOptions(NcnnRepro repro)
    {
        if (repro == null)
            return;
        repro.EnableTempPool = enableTempPool;
        repro.MaxPooledPerShape = maxPooledPerShape;
        repro.ForceBufferConvolutionAll = false;
        repro.DisallowBufferAccess = disallowBufferAccess;
        repro.DisallowBufferOutputs = disallowBufferOutputs;
        repro.DisallowBufferToTextureMaterialization = disallowBufferToTextureMaterialization;
        repro.DisallowInferenceTempComputeBuffers = disallowBufferAccess
            || disallowBufferOutputs
            || disallowBufferToTextureMaterialization;
        repro.DebugLogAllLayerHeartbeats = logAllLayerHeartbeats;
        repro.DebugLogAllLayerOutputs = logAllLayerOutputs;
        repro.DebugLogAllBufferMaterialize = logAllBufferMaterialize;
        repro.LayerRuntimeProfileEnabled = enableLayerRuntimeProfile;
        repro.LayerRuntimeProfileSyncGpu = layerRuntimeProfileSyncGpu;
    }

    private static void ConfigureClipRepro(
        NcnnRepro repro,
        bool strictReference,
        bool allowGeneralTextureConvolution,
        bool allowAttentionMatMulPack4)
    {
        if (repro == null)
            return;

        repro.ForceBufferConvolutionAll = strictReference;
        repro.ForceBufferBinaryOpAll = strictReference;
        repro.ForceBufferGeluAll = strictReference;
        repro.EnableDepthWiseTextureConvolution = !strictReference;
        repro.EnableConv1x1TextureConvolution = !strictReference;
        repro.EnableGeneralTextureConvolution = !strictReference && allowGeneralTextureConvolution;
        repro.EnableAttentionMatMulPack4Specializations = !strictReference && allowAttentionMatMulPack4;
        repro.EnableVistaTailPack4Specializations = false;
    }

    private bool ShouldForwardReproDebugLogToUnity()
    {
        return forwardReproDebugLogToUnity
            || logAllLayerHeartbeats
            || logAllLayerOutputs
            || logAllBufferMaterialize
            || disallowBufferAccess
            || disallowBufferOutputs
            || disallowBufferToTextureMaterialization;
    }

    private Action<string> CreateReproDebugLogSink(string stage, bool captureForDump)
    {
        return line =>
        {
            if (string.IsNullOrWhiteSpace(line))
                return;

            if (captureForDump)
                _imageCompareLines?.Add(line);

            if (ShouldForwardReproDebugLogToUnity())
                UnityEngine.Debug.Log("[CLIP-REPRO][" + stage + "] " + line);
        };
    }

    private static string FormatExceptionSummary(Exception e)
    {
        if (e == null)
            return null;

        var sb = new StringBuilder();
        var current = e;
        while (current != null)
        {
            if (sb.Length > 0)
                sb.Append(" --> ");
            sb.Append(current.GetType().Name);
            if (!string.IsNullOrWhiteSpace(current.Message))
                sb.Append(": ").Append(current.Message);
            current = current.InnerException;
        }
        return sb.ToString();
    }

    private async UniTask EnsureLoaded(CancellationToken ct)
    {
        var modelKey = ResolveModelKey();
        if (string.Equals(_loadedModelKey, modelKey, StringComparison.Ordinal)
            && _cachedTextEmbeddings != null
            && _cachedTextEmbeddings.Length > 0
            && _imageRepro?.Model != null)
            return;

        _cachedTextEmbeddings = null;
        _cachedTextScores = null;
        _loadedModelKey = null;
        _lastTextEmbeddingSource = null;

        var clipRoot = Path.Combine(Application.streamingAssetsPath, clipRootRelativePath);
        var modelRoot = Path.Combine(clipRoot, modelKey);

        var warmupSw = Stopwatch.StartNew();
        long imageLoadMs = 0;
        long textLoadMs = 0;
        long projectionLoadMs = 0;
        long buildTextMs = 0;
        NcnnRepro.ModelLoadProfile imageProfile = null;
        NcnnRepro.ModelLoadProfile textProfile = null;
        NcnnRepro.ModelLoadProfile projectionProfile = null;

        ReportProgress(0.02f, "Warm up CLIP");
        await UniTask.Yield();
        ct.ThrowIfCancellationRequested();

        var stageSw = Stopwatch.StartNew();
        await LoadReproModelAsync(
            _imageRepro,
            Path.Combine(modelRoot, "image_encoder.ncnn.param"),
            Path.Combine(modelRoot, "image_encoder.ncnn.bin"),
            "image encoder",
            0.04f,
            0.36f,
            ct);
        stageSw.Stop();
        imageLoadMs = stageSw.ElapsedMilliseconds;
        imageProfile = _imageRepro?.LastLoadProfile;

        ct.ThrowIfCancellationRequested();
        if (TryLoadTextEmbeddingCache(clipRoot, modelKey, out var cacheSource))
        {
            _lastTextEmbeddingSource = cacheSource;
            TryWriteTextEmbeddingCache(modelKey);
            ReportProgress(0.50f, "Load label cache");
            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpCachedTextEmbeddings();
            ReleaseTextRuntime();
        }
        else
        {
            EnsureTextRuntimeObjects();
            ApplyReproOptions();
            EnsureTokenizer(clipRoot);

            stageSw.Restart();
            await LoadReproModelAsync(
                _textRepro,
                Path.Combine(modelRoot, "text_encoder.ncnn.param"),
                Path.Combine(modelRoot, "text_encoder.ncnn.bin"),
                "text encoder",
                0.36f,
                0.50f,
                ct);
            stageSw.Stop();
            textLoadMs = stageSw.ElapsedMilliseconds;
            textProfile = _textRepro?.LastLoadProfile;

            stageSw.Restart();
            await LoadReproModelAsync(
                _projectionRepro,
                Path.Combine(modelRoot, "projection_layer.ncnn.param"),
                Path.Combine(modelRoot, "projection_layer.ncnn.bin"),
                "projection",
                0.50f,
                0.56f,
                ct);
            stageSw.Stop();
            projectionLoadMs = stageSw.ElapsedMilliseconds;
            projectionProfile = _projectionRepro?.LastLoadProfile;

            stageSw.Restart();
            await BuildTextEmbeddingsAsync(0.56f, 0.06f, ct);
            stageSw.Stop();
            buildTextMs = stageSw.ElapsedMilliseconds;

            _lastTextEmbeddingSource = "computed";
            TryWriteTextEmbeddingCache(modelKey);
            ReleaseTextRuntime();
        }

        _loadedModelKey = modelKey;
        warmupSw.Stop();
        UnityEngine.Debug.Log("[CLIP] Warmup | model=" + modelKey
            + " | textSource=" + (_lastTextEmbeddingSource ?? "none")
            + " | imageLoadMs=" + imageLoadMs
            + " | textLoadMs=" + textLoadMs
            + " | projectionLoadMs=" + projectionLoadMs
            + " | buildTextMs=" + buildTextMs
            + " | totalMs=" + warmupSw.ElapsedMilliseconds);
        LogLoadProfile("image encoder", imageProfile);
        if (textProfile != null)
            LogLoadProfile("text encoder", textProfile);
        if (projectionProfile != null)
            LogLoadProfile("projection", projectionProfile);
        ReportProgress(0.62f, "CLIP warmup ready");
    }

    private async UniTask BuildTextEmbeddingsAsync(float progressStart, float progressSpan, CancellationToken ct)
    {
        if (labelDefinitions == null || labelDefinitions.Length < 8)
            throw new InvalidOperationException("CLIP labelDefinitions must contain at least 8 labels.");

        _cachedTextScores = new ClipLabelScore[labelDefinitions.Length];
        _cachedTextEmbeddings = new float[labelDefinitions.Length][];
        for (var i = 0; i < labelDefinitions.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            var def = labelDefinitions[i];
            if (def == null || string.IsNullOrWhiteSpace(def.label) || string.IsNullOrWhiteSpace(def.prompt))
                throw new InvalidOperationException("CLIP label definition is incomplete at index " + i);

            ReportProgress(progressStart + progressSpan * ((float)i / Math.Max(1, labelDefinitions.Length)), "Encode text " + (i + 1) + "/" + labelDefinitions.Length);
            await UniTask.Yield();
            _cachedTextScores[i] = new ClipLabelScore { label = def.label.Trim(), prompt = def.prompt.Trim() };
            _cachedTextEmbeddings[i] = await EncodeTextAsync(def.prompt.Trim(), ct);
            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpVector(_lastDumpDir, "text_embedding_" + SanitizeFileName(def.label.Trim()) + ".txt", _cachedTextEmbeddings[i]);
        }

        ReportProgress(progressStart + progressSpan, "Text embeddings ready");
    }

    private async UniTask<float[]> EncodeTextAsync(string prompt, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var tokens = _tokenizer.Tokenize(prompt);

        using var tokenBuffer = new ComputeBuffer(tokens.Length, sizeof(int), ComputeBufferType.Structured);
        tokenBuffer.SetData(tokens);
        var tokenView = new NcnnTensorBuffer(tokenBuffer, 1, tokens.Length, 1, 1, 1, false);
        System.Collections.Generic.HashSet<string> pinned = null;
        if (enableDebugDump)
            pinned = new System.Collections.Generic.HashSet<string>(DebugTextBlobNames, StringComparer.Ordinal);

        using var infer = _textRepro.InferWithMultiInputs(null, new System.Collections.Generic.Dictionary<string, NcnnTensorBuffer>
        {
            { InputBlobName, tokenView }
        }, pinned);

        if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
        {
            for (var i = 0; i < DebugTextBlobNames.Length; i++)
                TryDumpAnyBlob(infer, DebugTextBlobNames[i], Path.Combine(_lastDumpDir, "text_blob_" + DebugTextBlobNames[i] + ".txt"));
        }

        var textOutput = infer.ReadTextureDataForOutput(OutputBlobName);
        if (textOutput == null || textOutput.Length != EmbeddingSize * tokens.Length)
            throw new InvalidOperationException("Text encoder output invalid for prompt: " + prompt);

        var eotIndex = Array.IndexOf(tokens, MobileClipSimpleTokenizer.EndTokenId);
        if (eotIndex < 0)
            eotIndex = Math.Max(0, tokens.Length - 1);

        var row = new float[EmbeddingSize];
        Array.Copy(textOutput, eotIndex * EmbeddingSize, row, 0, EmbeddingSize);

        using var rowBuffer = new ComputeBuffer(row.Length, sizeof(float), ComputeBufferType.Structured);
        rowBuffer.SetData(row);
        var rowView = new NcnnTensorBuffer(rowBuffer, 2, EmbeddingSize, 1, 1, 1, false);
        using var projInfer = _projectionRepro.InferWithMultiInputs(null, new System.Collections.Generic.Dictionary<string, NcnnTensorBuffer>
        {
            { InputBlobName, rowView }
        });

        var embedding = projInfer.ReadTextureDataForOutput(OutputBlobName);
        if (embedding == null || embedding.Length != EmbeddingSize)
            throw new InvalidOperationException("Projection output invalid for prompt: " + prompt);
        NormalizeInPlace(embedding);
        await UniTask.Yield();
        return embedding;
    }

    private async UniTask LoadReproModelAsync(
        NcnnRepro repro,
        string paramPath,
        string binPath,
        string modelLabel,
        float progressStart,
        float progressEnd,
        CancellationToken ct)
    {
        if (repro == null)
            throw new ArgumentNullException(nameof(repro));
        if (!File.Exists(paramPath))
            throw new FileNotFoundException("Model param not found", paramPath);
        if (!File.Exists(binPath))
            throw new FileNotFoundException("Model bin not found", binPath);

        ReportProgress(progressStart, "Read " + modelLabel);
        var paramText = File.ReadAllText(paramPath);
        ct.ThrowIfCancellationRequested();

        ReportProgress(Mathf.Min(progressEnd, progressStart + 0.002f), "Prepare " + modelLabel);
        await UniTask.Yield();

        using var fs = new FileStream(binPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, false);
        using var br = new NcnnBinReader(fs);
        await repro.LoadModelAsync(paramText, br, progress => ReportLoadProgress(modelLabel, progressStart, progressEnd, progress), ct);

        ReportProgress(progressEnd, modelLabel + " ready");
    }

    private void Release()
    {
        try { _imageRepro?.Dispose(); } catch { }
        _imageRepro = null;
        ReleaseTextRuntime();
        try { _ops?.Dispose(); } catch { }
        _ops = null;
        _cachedTextEmbeddings = null;
        _cachedTextScores = null;
        _loadedModelKey = null;
        _lastTextEmbeddingSource = null;
    }

    private string ResolveModelKey()
    {
        switch (modelLevel)
        {
            case ClipModelLevel.B:
                return "mobileclip_b_export";
            case ClipModelLevel.BLT:
                return "mobileclip_blt_export";
            case ClipModelLevel.S0:
                return "mobileclip_s0_export";
            case ClipModelLevel.S2:
                return "mobileclip_s2_export";
            default:
                return "mobileclip_s1_export";
        }
    }

    private int ResolveInputSize()
    {
        return modelLevel == ClipModelLevel.B || modelLevel == ClipModelLevel.BLT ? 224 : 256;
    }

    private bool TryLoadTextEmbeddingCache(string clipRoot, string modelKey, out string source)
    {
        source = null;
        if (string.IsNullOrWhiteSpace(clipRoot) || string.IsNullOrWhiteSpace(modelKey))
            return false;
        if (IsTextCacheDisabledByEnv())
            return false;

        var candidates = new[]
        {
            new KeyValuePair<string, string>(Path.Combine(clipRoot, modelKey + TextEmbeddingCacheSuffix), "streaming-cache"),
            new KeyValuePair<string, string>(Path.Combine(Application.persistentDataPath, clipRootRelativePath, modelKey + TextEmbeddingCacheSuffix), "persistent-cache")
        };

        for (var i = 0; i < candidates.Length; i++)
        {
            var path = candidates[i].Key;
            if (!File.Exists(path))
                continue;

            try
            {
                var cache = JsonUtility.FromJson<ClipTextEmbeddingCache>(File.ReadAllText(path));
                if (TryApplyTextEmbeddingCache(modelKey, cache))
                {
                    source = candidates[i].Value;
                    return true;
                }
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning("[CLIP] Failed to load text embedding cache: " + path + " | " + e.Message);
            }
        }

        if (TryLoadTextEmbeddingLogCache(modelKey))
        {
            source = "log-dump-cache";
            return true;
        }

        return false;
    }

    private static bool IsTextCacheDisabledByEnv()
    {
        try
        {
            var env = Environment.GetEnvironmentVariable("AIIMAGE_CLIP_DISABLE_TEXT_CACHE");
            return string.Equals(env, "1", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "yes", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private bool TryApplyTextEmbeddingCache(string modelKey, ClipTextEmbeddingCache cache)
    {
        if (cache == null)
            return false;
        if (!string.IsNullOrWhiteSpace(cache.modelKey)
            && !string.Equals(cache.modelKey, modelKey, StringComparison.Ordinal))
            return false;
        if (cache.entries == null || labelDefinitions == null || cache.entries.Length != labelDefinitions.Length)
            return false;

        var scores = new ClipLabelScore[cache.entries.Length];
        var embeddings = new float[cache.entries.Length][];
        for (var i = 0; i < cache.entries.Length; i++)
        {
            var def = labelDefinitions[i];
            var entry = cache.entries[i];
            if (def == null || entry == null)
                return false;

            var label = def.label?.Trim();
            var prompt = def.prompt?.Trim();
            if (string.IsNullOrWhiteSpace(label) || string.IsNullOrWhiteSpace(prompt))
                return false;
            if (!string.Equals(entry.label?.Trim(), label, StringComparison.Ordinal))
                return false;
            if (!string.Equals(entry.prompt?.Trim(), prompt, StringComparison.Ordinal))
                return false;
            if (entry.embedding == null || entry.embedding.Length != EmbeddingSize)
                return false;

            var normalized = (float[])entry.embedding.Clone();
            NormalizeInPlace(normalized);
            scores[i] = new ClipLabelScore { label = label, prompt = prompt };
            embeddings[i] = normalized;
        }

        _cachedTextScores = scores;
        _cachedTextEmbeddings = embeddings;
        return true;
    }

    private void DumpCachedTextEmbeddings()
    {
        if (string.IsNullOrWhiteSpace(_lastDumpDir) || _cachedTextScores == null || _cachedTextEmbeddings == null)
            return;

        for (var i = 0; i < Mathf.Min(_cachedTextScores.Length, _cachedTextEmbeddings.Length); i++)
        {
            if (_cachedTextEmbeddings[i] == null)
                continue;
            DumpVector(_lastDumpDir, "text_embedding_" + SanitizeFileName(_cachedTextScores[i].label) + ".txt", _cachedTextEmbeddings[i]);
        }
    }

    private bool TryLoadTextEmbeddingLogCache(string modelKey)
    {
        var root = Path.Combine(Application.dataPath, "..", "Logs", "ClipNcnnRepro");
        if (!Directory.Exists(root))
            return false;

        var expectedModel = modelLevel == ClipModelLevel.S0 ? "model=S0" : "model=S1";
        var dirs = Directory.GetDirectories(root);
        Array.Sort(dirs, StringComparer.Ordinal);
        for (var i = dirs.Length - 1; i >= 0; i--)
        {
            var scorePath = Path.Combine(dirs[i], "scores.txt");
            if (!File.Exists(scorePath))
                continue;

            string firstLine;
            try
            {
                using var sr = new StreamReader(scorePath);
                firstLine = sr.ReadLine();
            }
            catch
            {
                continue;
            }

            if (!string.Equals(firstLine, expectedModel, StringComparison.Ordinal))
                continue;

            var promptByLabel = new Dictionary<string, string>(StringComparer.Ordinal);
            try
            {
                foreach (var line in File.ReadLines(scorePath))
                {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("model=", StringComparison.Ordinal))
                        continue;

                    var parts = line.Split('\t');
                    if (parts.Length >= 4)
                    {
                        var label = parts[0]?.Trim();
                        var prompt = parts[3]?.Trim();
                        if (!string.IsNullOrWhiteSpace(label) && !string.IsNullOrWhiteSpace(prompt))
                            promptByLabel[label] = prompt;
                    }
                }
            }
            catch
            {
                continue;
            }

            var entries = new ClipTextEmbeddingCacheEntry[labelDefinitions.Length];
            var valid = true;
            for (var j = 0; j < labelDefinitions.Length; j++)
            {
                var def = labelDefinitions[j];
                if (def == null || string.IsNullOrWhiteSpace(def.label) || string.IsNullOrWhiteSpace(def.prompt))
                {
                    valid = false;
                    break;
                }

                if (!promptByLabel.TryGetValue(def.label.Trim(), out var promptFromDump))
                {
                    valid = false;
                    break;
                }

                var dumpPath = Path.Combine(dirs[i], "text_embedding_" + SanitizeFileName(def.label.Trim()) + ".txt");
                var embedding = TryReadEmbeddingDump(dumpPath);
                if (embedding == null || embedding.Length != EmbeddingSize)
                {
                    valid = false;
                    break;
                }

                entries[j] = new ClipTextEmbeddingCacheEntry
                {
                    label = def.label.Trim(),
                    prompt = promptFromDump,
                    embedding = embedding
                };
            }

            if (!valid)
                continue;

            var cache = new ClipTextEmbeddingCache
            {
                modelKey = modelKey,
                entries = entries
            };
            return TryApplyTextEmbeddingCache(modelKey, cache);
        }

        return false;
    }

    private static float[] TryReadEmbeddingDump(string path)
    {
        if (!File.Exists(path))
            return null;

        var values = new List<float>(EmbeddingSize);
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var tab = line.IndexOf('\t');
            if (tab <= 0 || tab >= line.Length - 1)
                continue;

            if (float.TryParse(line.Substring(tab + 1), NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                values.Add(v);
        }

        return values.Count == EmbeddingSize ? values.ToArray() : null;
    }

    private void TryWriteTextEmbeddingCache(string modelKey)
    {
        if (string.IsNullOrWhiteSpace(modelKey) || _cachedTextScores == null || _cachedTextEmbeddings == null)
            return;
        if (_cachedTextScores.Length != _cachedTextEmbeddings.Length || labelDefinitions == null || labelDefinitions.Length != _cachedTextScores.Length)
            return;

        try
        {
            var entries = new ClipTextEmbeddingCacheEntry[_cachedTextScores.Length];
            for (var i = 0; i < entries.Length; i++)
            {
                if (_cachedTextEmbeddings[i] == null || _cachedTextEmbeddings[i].Length != EmbeddingSize)
                    return;

                entries[i] = new ClipTextEmbeddingCacheEntry
                {
                    label = _cachedTextScores[i].label,
                    prompt = _cachedTextScores[i].prompt,
                    embedding = (float[])_cachedTextEmbeddings[i].Clone()
                };
            }

            var cache = new ClipTextEmbeddingCache
            {
                modelKey = modelKey,
                entries = entries
            };

            var dir = Path.Combine(Application.persistentDataPath, clipRootRelativePath);
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, modelKey + TextEmbeddingCacheSuffix);
            File.WriteAllText(path, JsonUtility.ToJson(cache, true));
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[CLIP] Failed to write text embedding cache: " + e.Message);
        }
    }

    private void ReportLoadProgress(string modelLabel, float progressStart, float progressEnd, NcnnRepro.LoadProgress progress)
    {
        var progress01 = Mathf.Lerp(progressStart, progressEnd, Mathf.Clamp01(progress.progress01));
        string text;
        switch (progress.stage)
        {
            case "release":
                text = "Reset " + modelLabel;
                break;
            case "parse":
                text = "Parse " + modelLabel;
                break;
            case "build-blobs":
                text = "Build " + modelLabel + " graph";
                break;
            case "layer":
                text = "Load " + modelLabel + " " + progress.layerIndex + "/" + progress.layerCount;
                if (!string.IsNullOrWhiteSpace(progress.layerType))
                    text += " (" + progress.layerType + ")";
                break;
            default:
                text = "Load " + modelLabel;
                break;
        }

        ReportProgress(progress01, text);
    }

    private static void LogLoadProfile(string label, NcnnRepro.ModelLoadProfile profile)
    {
        if (profile == null)
            return;

        var items = new List<KeyValuePair<string, NcnnRepro.LayerTypeLoadProfile>>(profile.layerTypes);
        items.Sort((a, b) => b.Value.totalMs.CompareTo(a.Value.totalMs));
        var top = new List<string>();
        for (var i = 0; i < Math.Min(4, items.Count); i++)
        {
            var item = items[i];
            top.Add(item.Key
                + ":" + item.Value.totalMs + "ms"
                + " read=" + item.Value.readMs
                + " upload=" + item.Value.uploadMs
                + " pack=" + item.Value.packMs
                + " count=" + item.Value.count);
        }

        UnityEngine.Debug.Log("[CLIP] LoadProfile " + label
            + " | totalMs=" + profile.totalMs
            + " | releaseMs=" + profile.releaseMs
            + " | parseMs=" + profile.parseParamMs
            + " | buildBlobUseMs=" + profile.buildBlobUseCountMs
            + " | bytesRead=" + profile.totalBytesRead
            + " | top=" + string.Join(" ; ", top));
    }

    private static RenderTexture ResizeTextureBilinear(Texture src, int width, int height)
    {
        if (src == null)
            return null;
        var rt = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default);
        var prev = RenderTexture.active;
        try
        {
            Graphics.Blit(src, rt);
        }
        finally
        {
            RenderTexture.active = prev;
        }
        return rt;
    }

    private static void ReleaseTemporaryRt(RenderTexture rt)
    {
        if (rt == null)
            return;
        RenderTexture.ReleaseTemporary(rt);
    }

    private static float Dot(float[] a, float[] b)
    {
        var total = 0f;
        var n = Math.Min(a?.Length ?? 0, b?.Length ?? 0);
        for (var i = 0; i < n; i++)
            total += a[i] * b[i];
        return total;
    }

    private static void NormalizeInPlace(float[] v)
    {
        if (v == null || v.Length == 0)
            return;
        var sum = 0f;
        for (var i = 0; i < v.Length; i++)
            sum += v[i] * v[i];
        var norm = Mathf.Sqrt(Mathf.Max(sum, 1e-12f));
        for (var i = 0; i < v.Length; i++)
            v[i] /= norm;
    }

    private static float[] Softmax(float[] logits)
    {
        if (logits == null || logits.Length == 0)
            return Array.Empty<float>();

        var max = logits[0];
        for (var i = 1; i < logits.Length; i++)
            max = Mathf.Max(max, logits[i]);

        var sum = 0f;
        var probs = new float[logits.Length];
        for (var i = 0; i < logits.Length; i++)
        {
            probs[i] = Mathf.Exp(logits[i] - max);
            sum += probs[i];
        }

        var inv = sum > 1e-12f ? 1f / sum : 0f;
        for (var i = 0; i < probs.Length; i++)
            probs[i] *= inv;
        return probs;
    }

    private static string CreateDumpDir()
    {
        var root = Path.Combine(Application.dataPath, "..", "Logs", "ClipNcnnRepro");
        Directory.CreateDirectory(root);
        var dir = Path.Combine(root, DateTime.Now.ToString("yyyyMMdd-HHmmss-fff", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void DumpScores(string dumpDir, ClipModelLevel level, ClipLabelScore[] scores)
    {
        if (string.IsNullOrWhiteSpace(dumpDir) || scores == null)
            return;

        using var sw = new StreamWriter(Path.Combine(dumpDir, "scores.txt"), false);
        sw.WriteLine("model=" + level);
        for (var i = 0; i < scores.Length; i++)
        {
            sw.WriteLine(
                scores[i].label + "\t"
                + scores[i].probability.ToString("0.000000", CultureInfo.InvariantCulture) + "\t"
                + scores[i].similarity.ToString("0.000000", CultureInfo.InvariantCulture) + "\t"
                + scores[i].prompt);
        }
    }

    private static void DumpVector(string dumpDir, string fileName, float[] values)
    {
        if (string.IsNullOrWhiteSpace(dumpDir) || string.IsNullOrWhiteSpace(fileName) || values == null)
            return;

        using var sw = new StreamWriter(Path.Combine(dumpDir, fileName), false);
        var finiteCount = 0;
        var nanCount = 0;
        var infCount = 0;
        var min = float.PositiveInfinity;
        var max = float.NegativeInfinity;
        for (var i = 0; i < values.Length; i++)
        {
            var v = values[i];
            if (float.IsNaN(v))
            {
                nanCount++;
                continue;
            }
            if (float.IsInfinity(v))
            {
                infCount++;
                continue;
            }
            finiteCount++;
            min = Mathf.Min(min, v);
            max = Mathf.Max(max, v);
        }

        sw.WriteLine("count=" + values.Length);
        sw.WriteLine("finite=" + finiteCount);
        sw.WriteLine("nan=" + nanCount);
        sw.WriteLine("inf=" + infCount);
        sw.WriteLine("min=" + (finiteCount > 0 ? min.ToString("R", CultureInfo.InvariantCulture) : "NaN"));
        sw.WriteLine("max=" + (finiteCount > 0 ? max.ToString("R", CultureInfo.InvariantCulture) : "NaN"));
        for (var i = 0; i < values.Length; i++)
            sw.WriteLine(i.ToString(CultureInfo.InvariantCulture) + "\t" + values[i].ToString("R", CultureInfo.InvariantCulture));
    }

    private static string SanitizeFileName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "unnamed";
        var s = value;
        foreach (var ch in Path.GetInvalidFileNameChars())
            s = s.Replace(ch, '_');
        return s.Replace(' ', '_');
    }

    private void DumpPack4TextureLogical(RenderTexture texture, int width, int height, int channels, string path)
    {
        if (ShouldAvoidInferenceBufferReadback())
            return;
        if (texture == null || width <= 0 || height <= 0 || channels <= 0 || string.IsNullOrWhiteSpace(path))
            return;

        var physicalChannels = Mathf.Max(4, ((channels + 3) / 4) * 4);
        var physicalCount = width * height * physicalChannels;
        using var physicalBuffer = new ComputeBuffer(physicalCount, sizeof(float), ComputeBufferType.Structured);
        _ops.Pack4ToBufferCHW(texture, width, height, physicalChannels, physicalBuffer);
        var physical = new float[physicalCount];
        physicalBuffer.GetData(physical);
        var logicalCount = width * height * channels;
        var logical = new float[logicalCount];
        Array.Copy(physical, logical, logicalCount);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);
        DumpVector(dir, Path.GetFileName(path), logical);
    }

    private void TryDumpAnyBlob(NcnnRepro.InferResult infer, string blobName, string path)
    {
        if (infer == null || string.IsNullOrWhiteSpace(blobName) || string.IsNullOrWhiteSpace(path))
            return;
        try
        {
            float[] values = null;
            if (ShouldAvoidInferenceBufferReadback())
            {
                if (!infer.TryGetExistingTextureData(blobName, out values) || values == null)
                    return;
            }
            else
            {
                try
                {
                    values = infer.ReadTextureDataForOutput(blobName);
                }
                catch
                {
                    if (!infer.TryGetExistingTextureData(blobName, out values) || values == null)
                    {
                        if (infer.TryGetLogicalShape(blobName, out _, out var w, out var h, out var d, out var c))
                        {
                            var texture = infer.GetTexture(blobName);
                            if (texture != null)
                            {
                                var packs = texture.volumeDepth > 0 ? texture.volumeDepth : 1;
                                var physicalChannels = packs * 4;
                                var physicalCount = texture.width * texture.height * physicalChannels;
                                using var physicalBuffer = new ComputeBuffer(physicalCount, sizeof(float), ComputeBufferType.Structured);
                                _ops.Pack4ToBufferCHW(texture, texture.width, texture.height, physicalChannels, physicalBuffer);
                                var physical = new float[physicalCount];
                                physicalBuffer.GetData(physical);
                                var logicalCount = w * h * d * c;
                                values = new float[Mathf.Min(logicalCount, physicalCount)];
                                Array.Copy(physical, values, values.Length);
                            }
                        }
                    }
                }
            }
            if (values == null)
                return;
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(dir))
                Directory.CreateDirectory(dir);
            DumpVector(dir, Path.GetFileName(path), values);
        }
        catch
        {
        }
    }

    private bool ShouldAvoidInferenceBufferReadback()
    {
        return disallowBufferAccess || disallowBufferOutputs || disallowBufferToTextureMaterialization;
    }

    private string BuildClassificationCacheSignature()
    {
        var sb = new StringBuilder();
        sb.Append("model=").Append(ResolveModelKey());
        sb.Append("|size=").Append(ResolveInputSize().ToString(CultureInfo.InvariantCulture));
        if (labelDefinitions != null)
        {
            for (var i = 0; i < labelDefinitions.Length; i++)
            {
                var def = labelDefinitions[i];
                sb.Append("|");
                sb.Append(def?.label?.Trim() ?? string.Empty);
                sb.Append("=>");
                sb.Append(def?.prompt?.Trim() ?? string.Empty);
            }
        }
        return sb.ToString();
    }

    private async UniTask<float[]> ReadImageEmbeddingAsync(NcnnRepro.InferResult infer, CancellationToken ct)
    {
        if (infer == null)
            return null;

        if (ShouldUseTextureReadbackForImageOutput(infer))
        {
            var texture = infer.GetTexture(OutputBlobName);
            if (texture != null)
            {
                var values = await ReadWidthVectorTextureAsync(texture, EmbeddingSize, ct);
                if (values != null && values.Length == EmbeddingSize)
                    return values;
            }
        }

        return infer.ReadTextureDataForOutput(OutputBlobName);
    }

    private async UniTask<float[]> EncodeImageWithCommandBufferAsync(Texture resizedInput, int targetSize, CancellationToken ct)
    {
        if (resizedInput == null)
            return null;

        using var cmd = new CommandBuffer { name = "ClipImageEncoder" };
        if (useAsyncComputeCommandBuffer)
            cmd.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);

        var inputCmd = _imageRepro.RentTempArray(cmd, targetSize, targetSize, 1, RenderTextureFormat.ARGBHalf);
        var outputCmd = default(ComputeTexture);
        RenderTexture outputReadbackRt = null;
        try
        {
            _ops.PackRgbToPack4(cmd, resizedInput, 0, 0, 1f, 1f, inputCmd, true);
            outputCmd = _imageRepro.ForwardPack4(cmd, inputCmd, 1, InputBlobName);
            if (outputCmd == null)
                throw new InvalidOperationException("CLIP command-buffer image encoder produced no output texture.");

            outputReadbackRt = outputCmd.dimension == TextureDimension.Tex2D
                ? _imageRepro.RentTempMat(outputCmd.width, outputCmd.height, outputCmd.format)
                : _imageRepro.RentTempArray(outputCmd.width, outputCmd.height, outputCmd.depth, outputCmd.format);
            cmd.CopyTexture(outputCmd.nameID, 0, 0, outputReadbackRt, 0, 0);

            _imageRepro.ReturnTempArray(cmd, outputCmd);
            outputCmd = null;
            _imageRepro.ReturnTempArray(cmd, inputCmd);
            inputCmd = null;

            if (useAsyncComputeCommandBuffer)
                Graphics.ExecuteCommandBufferAsync(cmd, ComputeQueueType.Default);
            else
                Graphics.ExecuteCommandBuffer(cmd);

            _imageRepro?.Ops?.DebugSyncGpu();
            ct.ThrowIfCancellationRequested();

            var values = await ReadWidthVectorTextureAsync(outputReadbackRt, EmbeddingSize, ct);
            if (values == null || values.Length != EmbeddingSize)
                throw new InvalidOperationException("CLIP command-buffer image encoder readback returned unexpected width: " + (values == null ? 0 : values.Length));
            return values;
        }
        finally
        {
            if (outputCmd != null)
                _imageRepro?.ReturnTempArray(cmd, outputCmd);
            if (inputCmd != null)
                _imageRepro?.ReturnTempArray(cmd, inputCmd);
            if (outputReadbackRt != null)
                _imageRepro?.ReturnTempArray(outputReadbackRt);
        }
    }

    private static string[] GetDebugImageBlobNames()
    {
        var extra = Environment.GetEnvironmentVariable(ExtraDebugImageBlobEnvVar);
        if (string.IsNullOrWhiteSpace(extra))
            return DebugImageBlobNames;

        var merged = new List<string>(DebugImageBlobNames.Length + 8);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < DebugImageBlobNames.Length; i++)
        {
            var name = DebugImageBlobNames[i];
            if (!string.IsNullOrWhiteSpace(name) && seen.Add(name))
                merged.Add(name);
        }

        var tokens = extra.Split(new[] { ',', ';', '\r', '\n', '\t', ' ' }, StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i < tokens.Length; i++)
        {
            var name = tokens[i]?.Trim();
            if (!string.IsNullOrWhiteSpace(name) && seen.Add(name))
                merged.Add(name);
        }

        return merged.ToArray();
    }

    private bool ShouldUseTextureReadbackForImageOutput(NcnnRepro.InferResult infer)
    {
        if (infer == null)
            return false;
        if (!forceFullRenderTexturePath && !disallowBufferAccess)
            return false;
        return infer.TryGetLogicalShape(OutputBlobName, out var dims, out var w, out var h, out var d, out var c)
            && dims == 1
            && w == EmbeddingSize
            && h == 1
            && d == 1
            && c == 1;
    }

    private static UniTask<float[]> ReadWidthVectorTextureAsync(RenderTexture texture, int width, CancellationToken ct)
    {
        if (texture == null || width <= 0)
            return UniTask.FromResult<float[]>(null);

        var prevActive = RenderTexture.active;
        Texture2D readbackTex = null;
        try
        {
            ct.ThrowIfCancellationRequested();
            if (Application.isBatchMode)
            {
                UnityEngine.Debug.Log("[CLIP] image embedding readback start | tex="
                    + texture.width + "x" + texture.height + "x" + Mathf.Max(1, texture.volumeDepth)
                    + " | requestedWidth=" + width);
            }

            Graphics.SetRenderTarget(texture, 0, CubemapFace.Unknown, 0);
            var readbackFormat = texture.format switch
            {
                RenderTextureFormat.RFloat => TextureFormat.RFloat,
                RenderTextureFormat.RHalf => TextureFormat.RHalf,
                RenderTextureFormat.ARGBFloat => TextureFormat.RGBAFloat,
                _ => TextureFormat.RGBAHalf
            };
            readbackTex = new Texture2D(width, 1, readbackFormat, false, true);
            readbackTex.ReadPixels(new Rect(0, 0, width, 1), 0, 0, false);
            readbackTex.Apply(false, false);

            var values = new float[width];
            if (readbackFormat == TextureFormat.RFloat)
            {
                var rawFloat = readbackTex.GetRawTextureData<float>();
                if (rawFloat.Length < width)
                    throw new InvalidOperationException("Readback payload too small for CLIP RFloat embedding: " + rawFloat.Length);
                for (var i = 0; i < width; i++)
                    values[i] = rawFloat[i];
            }
            else if (readbackFormat == TextureFormat.RHalf)
            {
                var rawHalf = readbackTex.GetRawTextureData<ushort>();
                if (rawHalf.Length < width)
                    throw new InvalidOperationException("Readback payload too small for CLIP RHalf embedding: " + rawHalf.Length);
                for (var i = 0; i < width; i++)
                    values[i] = HalfBitsToFloat(rawHalf[i]);
            }
            else if (readbackFormat == TextureFormat.RGBAFloat)
            {
                var rawFloat4 = readbackTex.GetRawTextureData<float>();
                if (rawFloat4.Length < width * 4)
                    throw new InvalidOperationException("Readback payload too small for CLIP RGBAFloat embedding: " + rawFloat4.Length);
                for (var i = 0; i < width; i++)
                    values[i] = rawFloat4[i * 4];
            }
            else
            {
                var rawHalf4 = readbackTex.GetRawTextureData<ushort>();
                if (rawHalf4.Length < width * 4)
                    throw new InvalidOperationException("Readback payload too small for CLIP RGBAHalf embedding: " + rawHalf4.Length);
                for (var i = 0; i < width; i++)
                    values[i] = HalfBitsToFloat(rawHalf4[i * 4]);
            }

            if (Application.isBatchMode)
                UnityEngine.Debug.Log("[CLIP] image embedding readback done | values=" + values.Length);

            return UniTask.FromResult(values);
        }
        finally
        {
            RenderTexture.active = prevActive;
            if (readbackTex != null)
                UnityEngine.Object.DestroyImmediate(readbackTex);
        }
    }

    private static float HalfBitsToFloat(ushort bits)
    {
        var sign = (bits >> 15) & 0x1;
        var exponent = (bits >> 10) & 0x1F;
        var mantissa = bits & 0x03FF;

        if (exponent == 0)
        {
            if (mantissa == 0)
                return sign == 0 ? 0f : -0f;

            var value = mantissa / 1024f;
            value *= Mathf.Pow(2f, -14f);
            return sign == 0 ? value : -value;
        }

        if (exponent == 31)
        {
            if (mantissa == 0)
                return sign == 0 ? float.PositiveInfinity : float.NegativeInfinity;
            return float.NaN;
        }

        var normalized = 1f + mantissa / 1024f;
        normalized *= Mathf.Pow(2f, exponent - 15f);
        return sign == 0 ? normalized : -normalized;
    }

    private void ReportProgress(float progress01, string text)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(progress01), text ?? string.Empty); } catch { }
    }
}
