using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Collections.Generic;
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
    private static readonly string[] DebugTextBlobNames = { "1", "3", "6", "7", "17", "18", "19", "22", "23", "24", "26", "27", "28", "29", "30", "57", "84", "111", "138", "165", "192", "219", "246", "273", "300", "320", "327", "out0" };
    private static readonly string[] DebugImageBlobNames = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "341", "343", "348", "349", "350", "351", "352", "353", "354", "355", "356", "357", "358", "359", "360", "361", "363", "366", "367", "368", "369", "370", "371", "372", "373", "376", "377", "378", "379", "380", "381", "382", "383", "384", "385", "386", "387", "388", "389", "390", "391", "392", "393", "396", "397", "398", "399", "414", "434", "449", "469", "484", "504", "519", "541", "out0" };
    private static readonly string[] DebugImageCompareLayers = { "convdw_253", "convdw_254", "conv_41", "conv_42", "convdw_255", "convdw_256", "conv_43", "conv_44", "convdw_257", "convdw_258", "conv_45", "conv_46", "convdw_259", "convrelu_0", "convsigmoid_3", "conv_49", "convdw_260", "convdw_261", "conv_50", "conv_51", "convdw_294", "convdw_295", "conv_84", "conv_85", "convdw_296", "convdw_297", "conv_86", "conv_87", "convdw_298", "convdw_299", "conv_88", "conv_89", "convdw_300" };

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
    private List<string> _imageCompareLines;

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
            if (enableDebugDump && string.IsNullOrWhiteSpace(_lastDumpDir))
                _lastDumpDir = CreateDumpDir();
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
            _ops.PackRgbToPack4(resized, 0, 0, 1f, 1f, inputPack4, true);
            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpPack4TextureLogical(inputPack4, targetSize, targetSize, 3, Path.Combine(_lastDumpDir, "image_blob_in0.txt"));

            ReportProgress(0.35f, "Encode image");
            await UniTask.Yield();
            ct.ThrowIfCancellationRequested();

            float[] imageEmbedding;
            System.Collections.Generic.HashSet<string> pinnedImage = null;
            if (enableDebugDump)
                pinnedImage = new System.Collections.Generic.HashSet<string>(DebugImageBlobNames, StringComparer.Ordinal);

            using (var infer = _imageRepro.Infer(inputPack4, 1, InputBlobName, pinnedImage))
            {
                if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                {
                    for (var i = 0; i < DebugImageBlobNames.Length; i++)
                        TryDumpAnyBlob(infer, DebugImageBlobNames[i], Path.Combine(_lastDumpDir, "image_blob_" + DebugImageBlobNames[i] + ".txt"));
                }
                imageEmbedding = infer.GetBufferData(OutputBlobName);
            }

            if (imageEmbedding == null || imageEmbedding.Length != EmbeddingSize)
                return Finish(new ClipClassificationResult { error = "Image embedding missing or invalid" });
            NormalizeInPlace(imageEmbedding);
            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpVector(_lastDumpDir, "image_embedding.txt", imageEmbedding);

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

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                DumpScores(_lastDumpDir, modelLevel, scores);
                if (_imageCompareLines != null && _imageCompareLines.Count > 0)
                    File.WriteAllLines(Path.Combine(_lastDumpDir, "image_conv_compare.txt"), _imageCompareLines);
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
        if (_imageRepro != null)
        {
            ConfigureClipRepro(_imageRepro, true);
            if (enableDebugDump)
            {
                _imageRepro.DebugCompareTextureLayers = new HashSet<string>(DebugImageCompareLayers, StringComparer.Ordinal);
                _imageCompareLines = new List<string>();
                _imageRepro.DebugLog = line =>
                {
                    if (!string.IsNullOrWhiteSpace(line))
                        _imageCompareLines.Add(line);
                };
            }
            else
            {
                _imageRepro.DebugCompareTextureLayers = null;
                _imageRepro.DebugLog = null;
                _imageCompareLines = null;
            }
        }
        if (_textRepro != null)
        {
            ConfigureClipRepro(_textRepro, modelLevel == ClipModelLevel.S0);
        }
        if (_projectionRepro != null)
        {
            ConfigureClipRepro(_projectionRepro, false);
        }
    }

    private void ApplyOptions(NcnnRepro4 repro)
    {
        if (repro == null)
            return;
        repro.EnableTempPool = enableTempPool;
        repro.MaxPooledPerShape = maxPooledPerShape;
        repro.ForceBufferConvolutionAll = false;
    }

    private static void ConfigureClipRepro(NcnnRepro4 repro, bool strictReference)
    {
        if (repro == null)
            return;

        repro.EnableWinograd23 = false;
        repro.ForceBufferConvolutionAll = strictReference;
        repro.ForceBufferBinaryOpAll = strictReference;
        repro.ForceBufferGeluAll = strictReference;
        repro.EnableDepthWiseTextureConvolution = !strictReference;
        repro.EnableConv1x1TextureConvolution = !strictReference;
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
            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                DumpVector(_lastDumpDir, "text_embedding_" + SanitizeFileName(def.label.Trim()) + ".txt", _cachedTextEmbeddings[i]);
        }
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

    private void TryDumpAnyBlob(NcnnRepro4.InferResult infer, string blobName, string path)
    {
        if (infer == null || string.IsNullOrWhiteSpace(blobName) || string.IsNullOrWhiteSpace(path))
            return;
        try
        {
            float[] values = null;
            try
            {
                values = infer.GetBufferData(blobName);
            }
            catch
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

    private void ReportProgress(float progress01, string text)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(progress01), text ?? string.Empty); } catch { }
    }
}
