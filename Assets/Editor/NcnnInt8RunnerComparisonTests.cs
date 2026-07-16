#if UNITY_EDITOR
using System;
using System.Collections;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AIImage.Inference.Core;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

[Serializable]
public sealed class NcnnInt8RunnerComparisonReport
{
    public string reportVersion = "aiimage.int8-selective-runner-comparison/v1";
    public string quantizationVersion = "aiimage.int8-selective/v1";
    public string executionBackend = "runner-texture-path";
    public string mattingInputPath;
    public string yoloInputPath;
    public string clipInputPath;
    public NcnnInt8RunnerComparisonCase[] cases;
}

[Serializable]
public sealed class NcnnInt8RunnerComparisonCase
{
    public string runner;
    public string comparedOutput;
    public string fp32Summary;
    public string fp16Summary;
    public string int8SelectiveSummary;
    public long fp32ElapsedMs;
    public long fp16ElapsedMs;
    public long int8SelectiveElapsedMs;
    public NcnnInt8RunnerGpuStats fp32GpuStats;
    public NcnnInt8RunnerGpuStats fp16GpuStats;
    public NcnnInt8RunnerGpuStats int8SelectiveGpuStats;
    public NcnnInt8RunnerPeakRtBufferComparison peakRtBufferComparison;
    public NcnnInt8RunnerPackageWeightComparison packageWeightComparison;
    public NcnnInt8RunnerEffectiveWeightComparison effectiveWeightComparison;
    public NcnnInt8RunnerQuantizationCoverage int8SelectiveCoverage;
    public NcnnInt8RunnerError fp16VsFp32;
    public NcnnInt8RunnerError int8SelectiveVsFp32;
    public float fp16CoverageDelta;
    public float int8SelectiveCoverageDelta;
    public float fp16CosineDistance;
    public float int8SelectiveCosineDistance;
    public string status;
}

[Serializable]
public sealed class NcnnInt8RunnerGpuStats
{
    public long peakBufferBytes;
    public long peakTextureBytes;
    public long peakTotalBytes;
    public long peakTemporaryTextureBytes;
    public int peakBufferCount;
    public int peakTextureCount;
}

[Serializable]
public sealed class NcnnInt8RunnerPeakRtBufferComparison
{
    public NcnnInt8RunnerGpuStats fp32;
    public NcnnInt8RunnerGpuStats fp16;
    public NcnnInt8RunnerGpuStats int8Selective;
}

[Serializable]
public sealed class NcnnInt8RunnerPackageWeightComparison
{
    public string measurement = "param+bin+manifest asset bytes currently packaged";
    public long fp32;
    public long fp16;
    public long int8Selective;
}

[Serializable]
public sealed class NcnnInt8RunnerEffectiveWeightComparison
{
    public string measurement = "estimated layer weight bytes: FP32/FP16 dense weights vs selected INT8 packed weights plus per-output scales; unselected layers remain FP32";
    public long fp32;
    public long fp16;
    public long int8Selective;
}

[Serializable]
public sealed class NcnnInt8RunnerQuantizationCoverage
{
    public string manifest;
    public int weightedLayerCount;
    public int int8LayerCount;
    public int floatOverrideLayerCount;
    public int unselectedLayerCount;
    public long totalWeightElements;
    public long int8WeightElements;
    public float int8WeightCoverage;
}

[Serializable]
public sealed class NcnnInt8RunnerError
{
    public float maxAbsoluteError;
    public float meanAbsoluteError;
    public float rootMeanSquareError;
}

public sealed class NcnnInt8RunnerComparisonTests
{
    private static readonly string Input02Path = Path.Combine(Directory.GetCurrentDirectory(), "documents", "ClipCompareInput", "02.png");
    private static readonly string Input03Path = Path.Combine(Directory.GetCurrentDirectory(), "documents", "ClipCompareInput", "03.jpg");
    private static readonly string OutputPath = Path.Combine(Directory.GetCurrentDirectory(), "output", "int8-selective-runner-comparison.json");

    [UnityTest]
    public IEnumerator MattingYoloClip_Runners_Int8Selective_StayWithinFp16Fp32Envelope()
    {
        var report = new NcnnInt8RunnerComparisonReport
        {
            mattingInputPath = Input03Path,
            yoloInputPath = Input03Path,
            clipInputPath = Input02Path,
            cases = new NcnnInt8RunnerComparisonCase[3]
        };

        var mattingTask = RunMattingComparisonAsync().AsTask();
        yield return WaitForTask(mattingTask);
        report.cases[0] = mattingTask.Result;

        var yoloTask = RunYoloComparisonAsync().AsTask();
        yield return WaitForTask(yoloTask);
        report.cases[1] = yoloTask.Result;

        var clipTask = RunClipComparisonAsync().AsTask();
        yield return WaitForTask(clipTask);
        report.cases[2] = clipTask.Result;

        Directory.CreateDirectory(Path.GetDirectoryName(OutputPath));
        File.WriteAllText(OutputPath, JsonUtility.ToJson(report, true) + "\n");
        Debug.Log("[INT8SelectiveRunnerComparison] wrote " + OutputPath);

        Assert.That(report.cases[0].int8SelectiveVsFp32.meanAbsoluteError, Is.LessThanOrEqualTo(0.035f), "Matting INT8 matte MAE");
        Assert.That(report.cases[0].int8SelectiveVsFp32.rootMeanSquareError, Is.LessThanOrEqualTo(0.06f), "Matting INT8 matte RMSE");
        Assert.That(report.cases[1].status, Is.EqualTo("ok"), "YOLO INT8 person count");
        Assert.That(report.cases[1].int8SelectiveVsFp32.meanAbsoluteError, Is.LessThanOrEqualTo(0.08f), "YOLO INT8 mask MAE");
        Assert.That(Mathf.Abs(report.cases[1].int8SelectiveCoverageDelta), Is.LessThanOrEqualTo(0.08f), "YOLO INT8 mask coverage delta");
        Assert.That(report.cases[2].status, Is.EqualTo("ok"), "CLIP INT8 best label");
        Assert.That(report.cases[2].int8SelectiveVsFp32.meanAbsoluteError, Is.LessThanOrEqualTo(0.03f), "CLIP INT8 embedding MAE");
        Assert.That(report.cases[2].int8SelectiveCosineDistance, Is.LessThanOrEqualTo(0.02f), "CLIP INT8 embedding cosine distance");
    }

    private static async UniTask<NcnnInt8RunnerComparisonCase> RunMattingComparisonAsync()
    {
        var input = LoadTexture(Input03Path);
        MattingRun fp32 = null;
        MattingRun fp16 = null;
        MattingRun int8 = null;
        try
        {
            fp32 = await RunMattingAsync(input, NcnnPrecisionMode.FP32);
            fp16 = await RunMattingAsync(input, NcnnPrecisionMode.FP16);
            int8 = await RunMattingAsync(input, NcnnPrecisionMode.INT8Selective);

            var result = new NcnnInt8RunnerComparisonCase
            {
                runner = "matting",
                comparedOutput = "matte.r",
                fp32Summary = DescribeTexture(fp32.result.matte),
                fp16Summary = DescribeTexture(fp16.result.matte),
                int8SelectiveSummary = DescribeTexture(int8.result.matte),
                fp32ElapsedMs = fp32.result.elapsedMs,
                fp16ElapsedMs = fp16.result.elapsedMs,
                int8SelectiveElapsedMs = int8.result.elapsedMs,
                fp32GpuStats = fp32.stats,
                fp16GpuStats = fp16.stats,
                int8SelectiveGpuStats = int8.stats,
                peakRtBufferComparison = CreatePeakComparison(fp32.stats, fp16.stats, int8.stats),
                packageWeightComparison = CreateMattingPackageWeightComparison(),
                effectiveWeightComparison = CreateEffectiveWeightComparison(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Matting", "matting.param"),
                    "matting.int8.model.json"),
                int8SelectiveCoverage = CreateQuantizationCoverage(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Matting", "matting.param"),
                    "matting.int8.model.json"),
                fp16VsFp32 = CompareTextureChannel(fp32.result.matte, fp16.result.matte, 0),
                int8SelectiveVsFp32 = CompareTextureChannel(fp32.result.matte, int8.result.matte, 0),
                fp16CoverageDelta = TextureMeanChannel(fp16.result.matte, 0) - TextureMeanChannel(fp32.result.matte, 0),
                int8SelectiveCoverageDelta = TextureMeanChannel(int8.result.matte, 0) - TextureMeanChannel(fp32.result.matte, 0),
                status = "ok"
            };
            return result;
        }
        finally
        {
            DestroyMattingRun(fp32);
            DestroyMattingRun(fp16);
            DestroyMattingRun(int8);
            DestroyImmediateSafe(input);
        }
    }

    private static async UniTask<NcnnInt8RunnerComparisonCase> RunYoloComparisonAsync()
    {
        var input = LoadTexture(Input03Path);
        YoloRun fp32 = null;
        YoloRun fp16 = null;
        YoloRun int8 = null;
        try
        {
            fp32 = await RunYoloAsync(input, NcnnPrecisionMode.FP32);
            fp16 = await RunYoloAsync(input, NcnnPrecisionMode.FP16);
            int8 = await RunYoloAsync(input, NcnnPrecisionMode.INT8Selective);

            var result = new NcnnInt8RunnerComparisonCase
            {
                runner = "yolo-seg",
                comparedOutput = "mask.r",
                fp32Summary = DescribeYolo(fp32.result),
                fp16Summary = DescribeYolo(fp16.result),
                int8SelectiveSummary = DescribeYolo(int8.result),
                fp32ElapsedMs = fp32.result.elapsedMs,
                fp16ElapsedMs = fp16.result.elapsedMs,
                int8SelectiveElapsedMs = int8.result.elapsedMs,
                fp32GpuStats = fp32.stats,
                fp16GpuStats = fp16.stats,
                int8SelectiveGpuStats = int8.stats,
                peakRtBufferComparison = CreatePeakComparison(fp32.stats, fp16.stats, int8.stats),
                packageWeightComparison = CreateYoloPackageWeightComparison(),
                effectiveWeightComparison = CreateEffectiveWeightComparison(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Yolo", "yolov8n_seg.ncnn.param"),
                    "yolo-seg.int8.model.json"),
                int8SelectiveCoverage = CreateQuantizationCoverage(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Yolo", "yolov8n_seg.ncnn.param"),
                    "yolo-seg.int8.model.json"),
                fp16VsFp32 = CompareTextureChannel(fp32.result.mask, fp16.result.mask, 0),
                int8SelectiveVsFp32 = CompareTextureChannel(fp32.result.mask, int8.result.mask, 0),
                fp16CoverageDelta = fp16.result.maskCoverage01 - fp32.result.maskCoverage01,
                int8SelectiveCoverageDelta = int8.result.maskCoverage01 - fp32.result.maskCoverage01,
                status = fp32.result.personCount == int8.result.personCount ? "ok" : "person-count-delta"
            };
            return result;
        }
        finally
        {
            DestroyYoloRun(fp32);
            DestroyYoloRun(fp16);
            DestroyYoloRun(int8);
            DestroyImmediateSafe(input);
        }
    }

    private static async UniTask<NcnnInt8RunnerComparisonCase> RunClipComparisonAsync()
    {
        var input = LoadTexture(Input02Path);
        ClipRun fp32 = null;
        ClipRun fp16 = null;
        ClipRun int8 = null;
        try
        {
            fp32 = await RunClipAsync(input, NcnnPrecisionMode.FP32);
            fp16 = await RunClipAsync(input, NcnnPrecisionMode.FP16);
            int8 = await RunClipAsync(input, NcnnPrecisionMode.INT8Selective);

            var fp16CosineDistance = 1f - Cosine(fp32.result.imageEmbedding, fp16.result.imageEmbedding);
            var int8CosineDistance = 1f - Cosine(fp32.result.imageEmbedding, int8.result.imageEmbedding);
            var result = new NcnnInt8RunnerComparisonCase
            {
                runner = "mobileclip-s0",
                comparedOutput = "imageEmbedding",
                fp32Summary = DescribeClip(fp32.result),
                fp16Summary = DescribeClip(fp16.result),
                int8SelectiveSummary = DescribeClip(int8.result),
                fp32ElapsedMs = fp32.result.elapsedMs,
                fp16ElapsedMs = fp16.result.elapsedMs,
                int8SelectiveElapsedMs = int8.result.elapsedMs,
                fp32GpuStats = fp32.stats,
                fp16GpuStats = fp16.stats,
                int8SelectiveGpuStats = int8.stats,
                peakRtBufferComparison = CreatePeakComparison(fp32.stats, fp16.stats, int8.stats),
                packageWeightComparison = CreateClipPackageWeightComparison(),
                effectiveWeightComparison = CreateEffectiveWeightComparison(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Clip", "mobileclip_s0_export", "image_encoder.ncnn.param"),
                    "clip-mobileclip-s0.int8.model.json"),
                int8SelectiveCoverage = CreateQuantizationCoverage(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Clip", "mobileclip_s0_export", "image_encoder.ncnn.param"),
                    "clip-mobileclip-s0.int8.model.json"),
                fp16VsFp32 = CompareVector(fp32.result.imageEmbedding, fp16.result.imageEmbedding),
                int8SelectiveVsFp32 = CompareVector(fp32.result.imageEmbedding, int8.result.imageEmbedding),
                fp16CosineDistance = fp16CosineDistance,
                int8SelectiveCosineDistance = int8CosineDistance,
                status = string.Equals(fp32.result.bestLabel, int8.result.bestLabel, StringComparison.Ordinal) ? "ok" : "best-label-delta"
            };
            return result;
        }
        finally
        {
            DestroyClipRun(fp32);
            DestroyClipRun(fp16);
            DestroyClipRun(int8);
            DestroyImmediateSafe(input);
        }
    }

    private static async UniTask<MattingRun> RunMattingAsync(Texture2D input, NcnnPrecisionMode precision)
    {
        var previousTracker = NcnnGpuResourceTracker.Enabled;
        NcnnGpuResourceTracker.Enabled = true;
        NcnnGpuResourceTracker.Reset("int8-runner-matting-" + precision);
        var go = new GameObject("MattingRunner_" + precision);
        var runner = go.AddComponent<MatterNcnnReproRunner>();
        runner.precisionMode = precision;
        runner.enableDebugDump = false;
        runner.forceBufferConvolution = false;
        runner.useCommandBuffer = false;
        try
        {
            var result = await runner.ProcessAsync(input, CancellationToken.None);
            Assert.That(result.error, Is.Null.Or.Empty, "Matting " + precision + " error");
            Assert.That(result.matte, Is.Not.Null, "Matting " + precision + " matte");
            return new MattingRun { go = go, result = result, stats = CaptureStats() };
        }
        catch
        {
            DestroyImmediateSafe(go);
            throw;
        }
        finally
        {
            NcnnGpuResourceTracker.Enabled = previousTracker;
        }
    }

    private static async UniTask<YoloRun> RunYoloAsync(Texture2D input, NcnnPrecisionMode precision)
    {
        var previousTracker = NcnnGpuResourceTracker.Enabled;
        NcnnGpuResourceTracker.Enabled = true;
        NcnnGpuResourceTracker.Reset("int8-runner-yolo-" + precision);
        var go = new GameObject("YoloRunner_" + precision);
        var runner = go.AddComponent<YoloSegNcnnReproRunner>();
        runner.precisionMode = precision;
        runner.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
        runner.enableDebugDump = false;
        runner.targetPersonOnly = true;
        runner.enableMaskClose = true;
        runner.enableMaskDilate = true;
        try
        {
            var result = await runner.ProcessAsync(input, CancellationToken.None);
            Assert.That(result.error, Is.Null.Or.Empty, "YOLO " + precision + " error");
            Assert.That(result.mask, Is.Not.Null, "YOLO " + precision + " mask");
            return new YoloRun { go = go, result = result, stats = CaptureStats() };
        }
        catch
        {
            DestroyImmediateSafe(go);
            throw;
        }
        finally
        {
            NcnnGpuResourceTracker.Enabled = previousTracker;
        }
    }

    private static async UniTask<ClipRun> RunClipAsync(Texture2D input, NcnnPrecisionMode precision)
    {
        var previousTracker = NcnnGpuResourceTracker.Enabled;
        NcnnGpuResourceTracker.Enabled = true;
        NcnnGpuResourceTracker.Reset("int8-runner-clip-" + precision);
        var go = new GameObject("ClipRunner_" + precision);
        var runner = go.AddComponent<ClipNcnnReproRunner>();
        runner.precisionMode = precision;
        runner.enableDebugDump = false;
        runner.useCommandBuffer = false;
        try
        {
            var result = await runner.ProcessAsync(input, CancellationToken.None);
            Assert.That(result.error, Is.Null.Or.Empty, "CLIP " + precision + " error");
            Assert.That(result.imageEmbedding, Is.Not.Null.And.Length.EqualTo(512), "CLIP " + precision + " embedding");
            return new ClipRun { go = go, result = result, stats = CaptureStats() };
        }
        catch
        {
            DestroyImmediateSafe(go);
            throw;
        }
        finally
        {
            NcnnGpuResourceTracker.Enabled = previousTracker;
        }
    }

    private static NcnnInt8RunnerGpuStats CaptureStats()
    {
        var stats = NcnnGpuResourceTracker.GetStatsSnapshot();
        return new NcnnInt8RunnerGpuStats
        {
            peakBufferBytes = stats.peakBufferBytes,
            peakTextureBytes = stats.peakTextureBytes,
            peakTotalBytes = stats.peakTotalBytes,
            peakTemporaryTextureBytes = stats.peakTemporaryTextureBytes,
            peakBufferCount = stats.peakBufferCount,
            peakTextureCount = stats.peakTextureCount
        };
    }

    private static NcnnInt8RunnerPeakRtBufferComparison CreatePeakComparison(
        NcnnInt8RunnerGpuStats fp32,
        NcnnInt8RunnerGpuStats fp16,
        NcnnInt8RunnerGpuStats int8Selective)
    {
        return new NcnnInt8RunnerPeakRtBufferComparison
        {
            fp32 = fp32,
            fp16 = fp16,
            int8Selective = int8Selective
        };
    }

    private static NcnnInt8RunnerPackageWeightComparison CreateMattingPackageWeightComparison()
    {
        var root = Directory.GetCurrentDirectory();
        var modelParam = Path.Combine(root, "Assets", "StreamingAssets", "Matting", "matting.param");
        var modelBin = Path.Combine(root, "Assets", "StreamingAssets", "Matting", "matting.bin");
        return CreatePackageWeightComparison(
            new[] { modelParam, modelBin, ManifestPath("matting.fp32.model.json") },
            new[] { modelParam, modelBin, ManifestPath("matting.fp16.model.json") },
            new[] { modelParam, modelBin, ManifestPath("matting.int8.model.json") });
    }

    private static NcnnInt8RunnerPackageWeightComparison CreateYoloPackageWeightComparison()
    {
        var root = Directory.GetCurrentDirectory();
        var modelParam = Path.Combine(root, "Assets", "StreamingAssets", "Yolo", "yolov8n_seg.ncnn.param");
        var modelBin = Path.Combine(root, "Assets", "StreamingAssets", "Yolo", "yolov8n_seg.ncnn.bin");
        return CreatePackageWeightComparison(
            new[] { modelParam, modelBin },
            new[] { modelParam, modelBin, ManifestPath("yolo-seg.fp16.model.json") },
            new[] { modelParam, modelBin, ManifestPath("yolo-seg.int8.model.json") });
    }

    private static NcnnInt8RunnerPackageWeightComparison CreateClipPackageWeightComparison()
    {
        var root = Directory.GetCurrentDirectory();
        var modelParam = Path.Combine(root, "Assets", "StreamingAssets", "Clip", "mobileclip_s0_export", "image_encoder.ncnn.param");
        var modelBin = Path.Combine(root, "Assets", "StreamingAssets", "Clip", "mobileclip_s0_export", "image_encoder.ncnn.bin");
        return CreatePackageWeightComparison(
            new[] { modelParam, modelBin, ManifestPath("clip-mobileclip-s0.fp32.model.json") },
            new[] { modelParam, modelBin, ManifestPath("clip-mobileclip-s0.fp16.model.json") },
            new[] { modelParam, modelBin, ManifestPath("clip-mobileclip-s0.int8.model.json") });
    }

    private static NcnnInt8RunnerPackageWeightComparison CreatePackageWeightComparison(
        string[] fp32,
        string[] fp16,
        string[] int8Selective)
    {
        return new NcnnInt8RunnerPackageWeightComparison
        {
            fp32 = SumExistingFileBytes(fp32),
            fp16 = SumExistingFileBytes(fp16),
            int8Selective = SumExistingFileBytes(int8Selective)
        };
    }

    private static string ManifestPath(string fileName)
    {
        return Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "InferenceManifests", fileName);
    }

    private static long SumExistingFileBytes(string[] paths)
    {
        long total = 0;
        foreach (var path in paths ?? Array.Empty<string>())
        {
            if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
                total += new FileInfo(path).Length;
        }
        return total;
    }

    private static NcnnInt8RunnerEffectiveWeightComparison CreateEffectiveWeightComparison(
        string paramPath,
        string manifestFileName)
    {
        var manifestPath = ManifestPath(manifestFileName);
        var manifest = NcnnModelManifestLoader.LoadFromFile(manifestPath);
        long totalWeightElements = 0;
        long int8SelectiveBytes = 0;

        foreach (var line in File.ReadLines(paramPath))
        {
            if (!TryParseWeightedLayer(line, out var operatorName, out var layerName, out var weightElements, out var outputChannels))
                continue;

            totalWeightElements += weightElements;
            if (manifest.TryGetQuantizedNodePlan(layerName, operatorName, out _))
            {
                int8SelectiveBytes += ((weightElements + 3) / 4) * sizeof(uint);
                int8SelectiveBytes += Math.Max(1, outputChannels) * sizeof(float);
            }
            else
            {
                int8SelectiveBytes += weightElements * sizeof(float);
            }
        }

        return new NcnnInt8RunnerEffectiveWeightComparison
        {
            fp32 = totalWeightElements * sizeof(float),
            fp16 = totalWeightElements * sizeof(ushort),
            int8Selective = int8SelectiveBytes
        };
    }

    private static NcnnInt8RunnerQuantizationCoverage CreateQuantizationCoverage(
        string paramPath,
        string manifestFileName)
    {
        var manifestPath = ManifestPath(manifestFileName);
        var manifest = NcnnModelManifestLoader.LoadFromFile(manifestPath);
        var coverage = new NcnnInt8RunnerQuantizationCoverage
        {
            manifest = manifestFileName
        };

        foreach (var line in File.ReadLines(paramPath))
        {
            if (!TryParseWeightedLayer(line, out var operatorName, out var layerName, out var weightElements, out _))
                continue;

            coverage.weightedLayerCount++;
            coverage.totalWeightElements += weightElements;
            if (manifest.TryGetQuantizedNodePlan(layerName, operatorName, out var plan))
            {
                coverage.int8LayerCount++;
                coverage.int8WeightElements += weightElements;
            }
            else if (plan != null && plan.mode == QuantizedNodeMode.Float)
            {
                coverage.floatOverrideLayerCount++;
            }
            else
            {
                coverage.unselectedLayerCount++;
            }
        }

        coverage.int8WeightCoverage = coverage.totalWeightElements > 0
            ? (float)((double)coverage.int8WeightElements / coverage.totalWeightElements)
            : 0f;
        return coverage;
    }

    private static bool TryParseWeightedLayer(
        string line,
        out string operatorName,
        out string layerName,
        out long weightElements,
        out long outputChannels)
    {
        operatorName = null;
        layerName = null;
        weightElements = 0;
        outputChannels = 0;
        if (string.IsNullOrWhiteSpace(line))
            return false;

        var tokens = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length < 2)
            return false;

        operatorName = tokens[0];
        layerName = tokens[1];
        var isWeighted =
            string.Equals(operatorName, "Convolution", StringComparison.Ordinal)
            || string.Equals(operatorName, "ConvolutionDepthWise", StringComparison.Ordinal)
            || string.Equals(operatorName, "InnerProduct", StringComparison.Ordinal)
            || string.Equals(operatorName, "Gemm", StringComparison.Ordinal);
        if (!isWeighted)
            return false;

        var p0 = GetParamInt64(tokens, "0");
        var p2 = GetParamInt64(tokens, "2");
        var p6 = GetParamInt64(tokens, "6");
        var p8 = GetParamInt64(tokens, "8");
        var p9 = GetParamInt64(tokens, "9");
        if (string.Equals(operatorName, "InnerProduct", StringComparison.Ordinal))
        {
            weightElements = p2;
            outputChannels = p0;
        }
        else if (string.Equals(operatorName, "Gemm", StringComparison.Ordinal))
        {
            weightElements = p8 > 0 && p9 > 0 ? p8 * p9 : 0;
            outputChannels = p8;
        }
        else
        {
            weightElements = p6;
            outputChannels = p0;
        }

        return weightElements > 0;
    }

    private static long GetParamInt64(string[] tokens, string key)
    {
        var prefix = key + "=";
        for (var i = 2; i < tokens.Length; i++)
        {
            if (!tokens[i].StartsWith(prefix, StringComparison.Ordinal))
                continue;
            if (long.TryParse(tokens[i].Substring(prefix.Length), out var value))
                return value;
            return 0;
        }
        return 0;
    }

    private static IEnumerator WaitForTask(Task task)
    {
        while (!task.IsCompleted)
            yield return null;

        if (task.IsFaulted)
            Assert.Fail(task.Exception?.GetBaseException().ToString() ?? "Unknown task failure");
        if (task.IsCanceled)
            Assert.Fail("Task was canceled.");
    }

    private static Texture2D LoadTexture(string path)
    {
        Assert.That(File.Exists(path), Is.True, "Input image missing: " + path);
        var bytes = File.ReadAllBytes(path);
        var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false, true)
        {
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
            name = Path.GetFileName(path)
        };
        if (!tex.LoadImage(bytes, false))
        {
            DestroyImmediateSafe(tex);
            Assert.Fail("Failed to decode input image: " + path);
        }
        return tex;
    }

    private static NcnnInt8RunnerError CompareTextureChannel(Texture2D expected, Texture2D actual, int channel)
    {
        Assert.That(expected, Is.Not.Null);
        Assert.That(actual, Is.Not.Null);
        Assert.That(actual.width, Is.EqualTo(expected.width));
        Assert.That(actual.height, Is.EqualTo(expected.height));

        var a = expected.GetPixels32();
        var b = actual.GetPixels32();
        double sumAbs = 0d;
        double sumSq = 0d;
        var max = 0f;
        for (var i = 0; i < a.Length; i++)
        {
            var diff = Mathf.Abs(GetChannel(a[i], channel) - GetChannel(b[i], channel)) / 255f;
            sumAbs += diff;
            sumSq += diff * diff;
            if (diff > max)
                max = diff;
        }
        return new NcnnInt8RunnerError
        {
            maxAbsoluteError = max,
            meanAbsoluteError = a.Length > 0 ? (float)(sumAbs / a.Length) : 0f,
            rootMeanSquareError = a.Length > 0 ? (float)Math.Sqrt(sumSq / a.Length) : 0f
        };
    }

    private static NcnnInt8RunnerError CompareVector(float[] expected, float[] actual)
    {
        Assert.That(expected, Is.Not.Null);
        Assert.That(actual, Is.Not.Null);
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        double sumAbs = 0d;
        double sumSq = 0d;
        var max = 0f;
        for (var i = 0; i < expected.Length; i++)
        {
            var diff = Mathf.Abs(expected[i] - actual[i]);
            sumAbs += diff;
            sumSq += diff * diff;
            if (diff > max)
                max = diff;
        }
        return new NcnnInt8RunnerError
        {
            maxAbsoluteError = max,
            meanAbsoluteError = expected.Length > 0 ? (float)(sumAbs / expected.Length) : 0f,
            rootMeanSquareError = expected.Length > 0 ? (float)Math.Sqrt(sumSq / expected.Length) : 0f
        };
    }

    private static float TextureMeanChannel(Texture2D texture, int channel)
    {
        if (texture == null)
            return 0f;
        var pixels = texture.GetPixels32();
        if (pixels.Length == 0)
            return 0f;
        double sum = 0d;
        for (var i = 0; i < pixels.Length; i++)
            sum += GetChannel(pixels[i], channel) / 255d;
        return (float)(sum / pixels.Length);
    }

    private static int GetChannel(Color32 value, int channel)
    {
        return channel == 0 ? value.r : channel == 1 ? value.g : channel == 2 ? value.b : value.a;
    }

    private static float Cosine(float[] a, float[] b)
    {
        Assert.That(a, Is.Not.Null);
        Assert.That(b, Is.Not.Null);
        Assert.That(b.Length, Is.EqualTo(a.Length));
        double dot = 0d;
        double aa = 0d;
        double bb = 0d;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            aa += a[i] * a[i];
            bb += b[i] * b[i];
        }
        if (aa <= 0d || bb <= 0d)
            return 0f;
        return (float)(dot / (Math.Sqrt(aa) * Math.Sqrt(bb)));
    }

    private static string DescribeTexture(Texture2D texture)
    {
        return texture == null ? "null" : texture.width + "x" + texture.height + " " + texture.format;
    }

    private static string DescribeYolo(YoloSegResult result)
    {
        return "personCount=" + result.personCount
            + " coverage=" + result.maskCoverage01.ToString("0.000000")
            + " detections=" + (result.detections == null ? 0 : result.detections.Length);
    }

    private static string DescribeClip(ClipClassificationResult result)
    {
        return "best=" + result.bestLabel
            + " prob=" + result.bestProbability.ToString("0.000000")
            + " scores=" + (result.scores == null ? 0 : result.scores.Length);
    }

    private static void DestroyMattingRun(MattingRun run)
    {
        if (run == null)
            return;
        DestroyImmediateSafe(run.result.texture);
        DestroyImmediateSafe(run.result.matte);
        DestroyImmediateSafe(run.go);
    }

    private static void DestroyYoloRun(YoloRun run)
    {
        if (run == null)
            return;
        DestroyImmediateSafe(run.result.texture);
        DestroyImmediateSafe(run.result.mask);
        DestroyImmediateSafe(run.result.overlay);
        DestroyImmediateSafe(run.go);
    }

    private static void DestroyClipRun(ClipRun run)
    {
        if (run == null)
            return;
        DestroyImmediateSafe(run.go);
    }

    private static void DestroyImmediateSafe(UnityEngine.Object obj)
    {
        if (obj != null)
            UnityEngine.Object.DestroyImmediate(obj);
    }

    private sealed class MattingRun
    {
        public GameObject go;
        public MattingResult result;
        public NcnnInt8RunnerGpuStats stats;
    }

    private sealed class YoloRun
    {
        public GameObject go;
        public YoloSegResult result;
        public NcnnInt8RunnerGpuStats stats;
    }

    private sealed class ClipRun
    {
        public GameObject go;
        public ClipClassificationResult result;
        public NcnnInt8RunnerGpuStats stats;
    }
}
#endif
