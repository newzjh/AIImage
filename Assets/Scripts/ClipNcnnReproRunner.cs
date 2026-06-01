using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
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
    public string error;
    public long elapsedMs;
}

public sealed class ClipNcnnReproRunner : MonoBehaviour
{
    public enum ClipModelLevel
    {
        S0,
        S1
    }

    [Serializable]
    public sealed class ClipLabelDefinition
    {
        public string label;
        public string prompt;
    }

    private const string InputBlobName = "in0";
    private const string OutputBlobName = "out0";
    private const int EmbeddingSize = 512;

    public ClipModelLevel modelLevel = ClipModelLevel.S1;
    public string clipRootRelativePath = "Clip";
    public bool enableTempPool = true;
    public int maxPooledPerShape = 4;
    public bool enableDebugDump = false;
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
    private NcnnRepro4 _imageRepro;
    private NcnnRepro4 _textRepro;
    private NcnnRepro4 _projectionRepro;
    private MobileClipSimpleTokenizer _tokenizer;
    private string _loadedModelKey;
    private ClipLabelScore[] _cachedTextScores;
    private float[][] _cachedTextEmbeddings;
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

    public async UniTask<ClipClassificationResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (src == null)
            return default;

        var totalSw = Stopwatch.StartNew();
        _lastDumpDir = null;

        ClipClassificationResult Finish(ClipClassificationResult result)
        {
            result.elapsedMs = totalSw.ElapsedMilliseconds;
            return result;
        }

        RenderTexture resized = null;
        RenderTexture inputPack4 = null;
        try
        {
            EnsureRuntimeObjects();
            ApplyReproOptions();
            await EnsureLoaded(ct);
            if (_cachedTextEmbeddings == null || _cachedTextEmbeddings.Length == 0)
                return Finish(new ClipClassificationResult { error = "CLIP text embeddings unavailable" });

            var targetSize = 256;
            ReportProgress(0.08f, "Prepare input");
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();

            resized = ResizeTextureBilinear(src, targetSize, targetSize);
            if (resized == null)
                return Finish(new ClipClassificationResult { error = "Resize input failed" });

            inputPack4 = _imageRepro.RentTempArray(targetSize, targetSize, 1, RenderTextureFormat.ARGBHalf);
            _ops.PackRgbToPack4(resized, 0, 0, 1f, 1f, inputPack4);

            ReportProgress(0.35f, "Encode image");
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();

            float[] imageEmbedding;
            using (var infer = _imageRepro.Infer(inputPack4, 1, InputBlobName))
            {
                imageEmbedding = infer.GetBufferData(OutputBlobName);
            }

            if (imageEmbedding == null || imageEmbedding.Length != EmbeddingSize)
                return Finish(new ClipClassificationResult { error = "Image embedding missing or invalid" });
            NormalizeInPlace(imageEmbedding);

            ReportProgress(0.72f, "Score labels");
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();

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

            if (enableDebugDump)
            {
                _lastDumpDir = CreateDumpDir();
                DumpScores(_lastDumpDir, modelLevel, scores);
            }

            ReportProgress(1f, string.Empty);
            return Finish(new ClipClassificationResult
            {
                bestLabel = scores.Length > 0 ? scores[0].label : null,
                bestProbability = scores.Length > 0 ? scores[0].probability : 0f,
                scores = scores
            });
        }
        catch (OperationCanceledException)
        {
            return Finish(new ClipClassificationResult { error = "Cancelled" });
        }
        catch (Exception e)
        {
            return Finish(new ClipClassificationResult { error = e.Message });
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
        _imageRepro ??= new NcnnRepro4(_ops);
        _textRepro ??= new NcnnRepro4(_ops);
        _projectionRepro ??= new NcnnRepro4(_ops);
    }

    private void ApplyReproOptions()
    {
        ApplyOptions(_imageRepro);
        ApplyOptions(_textRepro);
        ApplyOptions(_projectionRepro);
    }

    private void ApplyOptions(NcnnRepro4 repro)
    {
        if (repro == null)
            return;
        repro.EnableTempPool = enableTempPool;
        repro.MaxPooledPerShape = maxPooledPerShape;
    }

    private async UniTask EnsureLoaded(CancellationToken ct)
    {
        var modelKey = modelLevel == ClipModelLevel.S0 ? "mobileclip_s0_export" : "mobileclip_s1_export";
        if (string.Equals(_loadedModelKey, modelKey, StringComparison.Ordinal) && _cachedTextEmbeddings != null)
            return;

        var clipRoot = Path.Combine(Application.streamingAssetsPath, clipRootRelativePath);
        var modelRoot = Path.Combine(clipRoot, modelKey);
        var vocabPath = Path.Combine(clipRoot, "vocab.txt");
        var bpePath = Path.Combine(clipRoot, "bpe_simple_vocab_16e6.txt");

        _tokenizer = new MobileClipSimpleTokenizer(vocabPath, bpePath);
        LoadModel(_imageRepro, Path.Combine(modelRoot, "image_encoder.ncnn.param"), Path.Combine(modelRoot, "image_encoder.ncnn.bin"));
        LoadModel(_textRepro, Path.Combine(modelRoot, "text_encoder.ncnn.param"), Path.Combine(modelRoot, "text_encoder.ncnn.bin"));
        LoadModel(_projectionRepro, Path.Combine(modelRoot, "projection_layer.ncnn.param"), Path.Combine(modelRoot, "projection_layer.ncnn.bin"));

        ReportProgress(0.12f, "Encode text labels");
        await BuildTextEmbeddingsAsync(ct);
        _loadedModelKey = modelKey;
    }

    private async UniTask BuildTextEmbeddingsAsync(CancellationToken ct)
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

            ReportProgress(0.12f + 0.18f * ((float)i / Math.Max(1, labelDefinitions.Length)), "Encode text " + (i + 1) + "/" + labelDefinitions.Length);
            await UniTask.Yield();
            _cachedTextScores[i] = new ClipLabelScore { label = def.label.Trim(), prompt = def.prompt.Trim() };
            _cachedTextEmbeddings[i] = await EncodeTextAsync(def.prompt.Trim(), ct);
        }
    }

    private async UniTask<float[]> EncodeTextAsync(string prompt, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var tokens = _tokenizer.Tokenize(prompt);

        using var tokenBuffer = new ComputeBuffer(tokens.Length, sizeof(int), ComputeBufferType.Structured);
        tokenBuffer.SetData(tokens);
        var tokenView = new NcnnTensorBuffer(tokenBuffer, 1, tokens.Length, 1, 1, 1, false);
        using var infer = _textRepro.InferWithMultiInputs(null, new System.Collections.Generic.Dictionary<string, NcnnTensorBuffer>
        {
            { InputBlobName, tokenView }
        });

        var textOutput = infer.GetBufferData(OutputBlobName);
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

        var embedding = projInfer.GetBufferData(OutputBlobName);
        if (embedding == null || embedding.Length != EmbeddingSize)
            throw new InvalidOperationException("Projection output invalid for prompt: " + prompt);
        NormalizeInPlace(embedding);
        await UniTask.Yield();
        return embedding;
    }

    private static void LoadModel(NcnnRepro4 repro, string paramPath, string binPath)
    {
        if (repro == null)
            throw new ArgumentNullException(nameof(repro));
        if (!File.Exists(paramPath))
            throw new FileNotFoundException("Model param not found", paramPath);
        if (!File.Exists(binPath))
            throw new FileNotFoundException("Model bin not found", binPath);

        var paramText = File.ReadAllText(paramPath);
        using var fs = File.OpenRead(binPath);
        using var br = new NcnnBinReader(fs);
        repro.LoadModel(paramText, br);
    }

    private void Release()
    {
        try { _imageRepro?.Dispose(); } catch { }
        try { _textRepro?.Dispose(); } catch { }
        try { _projectionRepro?.Dispose(); } catch { }
        _imageRepro = null;
        _textRepro = null;
        _projectionRepro = null;
        _cachedTextEmbeddings = null;
        _cachedTextScores = null;
        _loadedModelKey = null;
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

    private void ReportProgress(float progress01, string text)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(progress01), text ?? string.Empty); } catch { }
    }
}
