#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Aexis;
using Cysharp.Threading.Tasks;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

[Serializable]
public sealed class NcnnInt4RunnerComparisonReport
{
    public string reportVersion = "aiimage.int4-selective-runner-comparison/v2";
    public string quantizationVersion = "aiimage.int4-selective/v1";
    public string executionBackend = "runner-texture-path";
    public string mattingInputPath;
    public string yoloInputPath;
    public string clipInputPath;
    public NcnnInt4RunnerComparisonCase[] cases;
}

[Serializable]
public sealed class NcnnInt4RunnerComparisonCase
{
    public string runner;
    public string comparedOutput;
    public string fp32Summary;
    public string fp16Summary;
    public string int4SelectiveSummary;
    public long fp32ElapsedMs;
    public long fp16ElapsedMs;
    public long int4SelectiveElapsedMs;
    public NcnnInt4RunnerGpuStats fp32GpuStats;
    public NcnnInt4RunnerGpuStats fp16GpuStats;
    public NcnnInt4RunnerGpuStats int4SelectiveGpuStats;
    public NcnnInt4RunnerPeakRtBufferComparison peakRtBufferComparison;
    public NcnnInt4RunnerPackageWeightComparison packageWeightComparison;
    public NcnnInt4RunnerEffectiveWeightComparison effectiveWeightComparison;
    public NcnnInt4RunnerQuantizationCoverage int4SelectiveCoverage;
    public NcnnInt4RunnerError fp16VsFp32;
    public NcnnInt4RunnerError int4SelectiveVsFp32;
    public NcnnInt4RunnerClipDiagnostics clipDiagnostics;
    public float fp16CoverageDelta;
    public float int4SelectiveCoverageDelta;
    public float fp16CosineDistance;
    public float int4SelectiveCosineDistance;
    public string status;
}

[Serializable]
public sealed class NcnnInt4RunnerGpuStats
{
    public long peakBufferBytes;
    public long peakTextureBytes;
    public long peakTotalBytes;
    public long peakTemporaryTextureBytes;
    public int peakBufferCount;
    public int peakTextureCount;
}

[Serializable]
public sealed class NcnnInt4RunnerPeakRtBufferComparison
{
    public NcnnInt4RunnerGpuStats fp32;
    public NcnnInt4RunnerGpuStats fp16;
    public NcnnInt4RunnerGpuStats int4Selective;
}

[Serializable]
public sealed class NcnnInt4RunnerPackageWeightComparison
{
    public string measurement = "param+bin+manifest asset bytes currently packaged";
    public long fp32;
    public long fp16;
    public long int4Selective;
}

[Serializable]
public sealed class NcnnInt4RunnerEffectiveWeightComparison
{
    public string measurement = "estimated layer weight bytes: FP32/FP16 dense weights vs selected INT4 packed weights plus per-output scales; unselected layers remain FP32";
    public long fp32;
    public long fp16;
    public long int4Selective;
}

[Serializable]
public sealed class NcnnInt4RunnerQuantizationCoverage
{
    public string manifest;
    public int weightedLayerCount;
    public int int4LayerCount;
    public int floatOverrideLayerCount;
    public int unselectedLayerCount;
    public long totalWeightElements;
    public long int4WeightElements;
    public float int4WeightCoverage;
}

[Serializable]
public sealed class NcnnInt4RunnerError
{
    public float maxAbsoluteError;
    public float meanAbsoluteError;
    public float rootMeanSquareError;
}

[Serializable]
public sealed class NcnnInt4RunnerClipDiagnostics
{
    public string scoreScale = "logit = normalized image/text dot product * 100, probability = softmax(logits)";
    public NcnnInt4RunnerClipTopScore fp32Top1;
    public NcnnInt4RunnerClipTopScore fp32Top2;
    public NcnnInt4RunnerClipTopScore fp16Top1;
    public NcnnInt4RunnerClipTopScore fp16Top2;
    public NcnnInt4RunnerClipTopScore int4SelectiveTop1;
    public NcnnInt4RunnerClipTopScore int4SelectiveTop2;
    public float fp32Top1Top2LogitMargin;
    public float fp16Top1Top2LogitMargin;
    public float int4SelectiveTop1Top2LogitMargin;
    public float fp32Top1Top2ProbabilityMargin;
    public float fp16Top1Top2ProbabilityMargin;
    public float int4SelectiveTop1Top2ProbabilityMargin;
    public float fp16Top1LogitDeltaVsFp32;
    public float int4SelectiveTop1LogitDeltaVsFp32;
    public float fp16Top1ProbabilityDeltaVsFp32;
    public float int4SelectiveTop1ProbabilityDeltaVsFp32;
    public float fp16MeanAbsLogitDeltaVsFp32;
    public float int4SelectiveMeanAbsLogitDeltaVsFp32;
    public float fp16MaxAbsLogitDeltaVsFp32;
    public float int4SelectiveMaxAbsLogitDeltaVsFp32;
    public float fp16MeanAbsProbabilityDeltaVsFp32;
    public float int4SelectiveMeanAbsProbabilityDeltaVsFp32;
    public float fp16MaxAbsProbabilityDeltaVsFp32;
    public float int4SelectiveMaxAbsProbabilityDeltaVsFp32;
    public NcnnInt4RunnerClipScoreDelta[] labels;
}

[Serializable]
public sealed class NcnnInt4RunnerClipTopScore
{
    public string label;
    public float logit;
    public float probability;
}

[Serializable]
public sealed class NcnnInt4RunnerClipScoreDelta
{
    public string label;
    public string prompt;
    public float fp32Logit;
    public float fp16Logit;
    public float int4SelectiveLogit;
    public float fp16LogitDeltaVsFp32;
    public float int4SelectiveLogitDeltaVsFp32;
    public float fp32Probability;
    public float fp16Probability;
    public float int4SelectiveProbability;
    public float fp16ProbabilityDeltaVsFp32;
    public float int4SelectiveProbabilityDeltaVsFp32;
}

public sealed class NcnnInt4RunnerComparisonTests
{
    private static readonly string Input03Path = Path.Combine(Directory.GetCurrentDirectory(), "documents", "ClipCompareInput", "03.jpg");
    private static readonly string OutputPath = Path.Combine(Directory.GetCurrentDirectory(), "output", "int4-selective-runner-comparison.json");

    [UnityTest]
    public IEnumerator MattingYoloClip_Runners_Int4Selective_StayWithinFp16Fp32Envelope()
    {
        var requestedRunners = ParseRequestedRunners();
        var report = new NcnnInt4RunnerComparisonReport
        {
            mattingInputPath = Input03Path,
            yoloInputPath = Input03Path,
            clipInputPath = Input03Path,
            cases = Array.Empty<NcnnInt4RunnerComparisonCase>()
        };

        var cases = new List<NcnnInt4RunnerComparisonCase>(3);
        if (ShouldRunRunner(requestedRunners, "matting"))
        {
            var mattingTask = RunMattingComparisonAsync().AsTask();
            yield return WaitForTask(mattingTask);
            cases.Add(mattingTask.Result);
        }

        if (ShouldRunRunner(requestedRunners, "yolo-seg"))
        {
            var yoloTask = RunYoloComparisonAsync().AsTask();
            yield return WaitForTask(yoloTask);
            cases.Add(yoloTask.Result);
        }

        if (ShouldRunRunner(requestedRunners, "mobileclip-s0"))
        {
            var clipTask = RunClipComparisonAsync().AsTask();
            yield return WaitForTask(clipTask);
            cases.Add(clipTask.Result);
        }

        report.cases = cases.ToArray();
        Directory.CreateDirectory(Path.GetDirectoryName(OutputPath));
        File.WriteAllText(OutputPath, JsonUtility.ToJson(report, true) + "\n");
        Debug.Log("[INT4SelectiveRunnerComparison] wrote " + OutputPath);

        foreach (var result in report.cases)
        {
            if (string.Equals(result.runner, "matting", StringComparison.Ordinal))
            {
                Assert.That(result.int4SelectiveVsFp32.meanAbsoluteError, Is.LessThanOrEqualTo(0.06f), "Matting INT4 matte MAE");
                Assert.That(result.int4SelectiveVsFp32.rootMeanSquareError, Is.LessThanOrEqualTo(0.11f), "Matting INT4 matte RMSE");
            }
            else if (string.Equals(result.runner, "yolo-seg", StringComparison.Ordinal))
            {
                Assert.That(result.status, Is.EqualTo("ok"), "YOLO INT4 person count");
                Assert.That(result.int4SelectiveVsFp32.meanAbsoluteError, Is.LessThanOrEqualTo(0.10f), "YOLO INT4 mask MAE");
                Assert.That(Mathf.Abs(result.int4SelectiveCoverageDelta), Is.LessThanOrEqualTo(0.10f), "YOLO INT4 mask coverage delta");
            }
            else if (string.Equals(result.runner, "mobileclip-s0", StringComparison.Ordinal))
            {
                Assert.That(result.status, Is.EqualTo("ok"), "CLIP INT4 best label");
                Assert.That(result.int4SelectiveVsFp32.meanAbsoluteError, Is.LessThanOrEqualTo(0.05f), "CLIP INT4 embedding MAE");
                Assert.That(result.int4SelectiveCosineDistance, Is.LessThanOrEqualTo(0.05f), "CLIP INT4 embedding cosine distance");
            }
        }
    }

    private static HashSet<string> ParseRequestedRunners()
    {
        var value = Environment.GetEnvironmentVariable("AIIMAGE_INT4_COMPARE_RUNNERS");
        if (string.IsNullOrWhiteSpace(value))
            return null;

        var runners = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var token in value.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries))
        {
            if (string.Equals(token, "yolo", StringComparison.OrdinalIgnoreCase))
                runners.Add("yolo-seg");
            else if (string.Equals(token, "clip", StringComparison.OrdinalIgnoreCase))
                runners.Add("mobileclip-s0");
            else
                runners.Add(token);
        }
        return runners;
    }

    private static bool ShouldRunRunner(HashSet<string> requestedRunners, string runner)
    {
        return requestedRunners == null || requestedRunners.Contains(runner);
    }

    private static async UniTask<NcnnInt4RunnerComparisonCase> RunMattingComparisonAsync()
    {
        var input = LoadTexture(Input03Path);
        MattingRun fp32 = null;
        MattingRun fp16 = null;
        MattingRun int4 = null;
        try
        {
            fp32 = await RunMattingAsync(input, NcnnPrecisionMode.FP32);
            fp16 = await RunMattingAsync(input, NcnnPrecisionMode.FP16);
            int4 = await RunMattingAsync(input, NcnnPrecisionMode.INT4Selective);

            var result = new NcnnInt4RunnerComparisonCase
            {
                runner = "matting",
                comparedOutput = "matte.r",
                fp32Summary = DescribeTexture(fp32.result.matte),
                fp16Summary = DescribeTexture(fp16.result.matte),
                int4SelectiveSummary = DescribeTexture(int4.result.matte),
                fp32ElapsedMs = fp32.result.elapsedMs,
                fp16ElapsedMs = fp16.result.elapsedMs,
                int4SelectiveElapsedMs = int4.result.elapsedMs,
                fp32GpuStats = fp32.stats,
                fp16GpuStats = fp16.stats,
                int4SelectiveGpuStats = int4.stats,
                peakRtBufferComparison = CreatePeakComparison(fp32.stats, fp16.stats, int4.stats),
                packageWeightComparison = CreateMattingPackageWeightComparison(),
                effectiveWeightComparison = CreateEffectiveWeightComparison(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Matting", "matting.param"),
                    "matting.int4.model.json"),
                int4SelectiveCoverage = CreateQuantizationCoverage(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Matting", "matting.param"),
                    "matting.int4.model.json"),
                fp16VsFp32 = CompareTextureChannel(fp32.result.matte, fp16.result.matte, 0),
                int4SelectiveVsFp32 = CompareTextureChannel(fp32.result.matte, int4.result.matte, 0),
                fp16CoverageDelta = TextureMeanChannel(fp16.result.matte, 0) - TextureMeanChannel(fp32.result.matte, 0),
                int4SelectiveCoverageDelta = TextureMeanChannel(int4.result.matte, 0) - TextureMeanChannel(fp32.result.matte, 0),
                status = "ok"
            };
            return result;
        }
        finally
        {
            DestroyMattingRun(fp32);
            DestroyMattingRun(fp16);
            DestroyMattingRun(int4);
            DestroyImmediateSafe(input);
        }
    }

    private static async UniTask<NcnnInt4RunnerComparisonCase> RunYoloComparisonAsync()
    {
        var input = LoadTexture(Input03Path);
        YoloRun fp32 = null;
        YoloRun fp16 = null;
        YoloRun int4 = null;
        try
        {
            fp32 = await RunYoloAsync(input, NcnnPrecisionMode.FP32);
            fp16 = await RunYoloAsync(input, NcnnPrecisionMode.FP16);
            int4 = await RunYoloAsync(input, NcnnPrecisionMode.INT4Selective);

            var result = new NcnnInt4RunnerComparisonCase
            {
                runner = "yolo-seg",
                comparedOutput = "mask.r",
                fp32Summary = DescribeYolo(fp32.result),
                fp16Summary = DescribeYolo(fp16.result),
                int4SelectiveSummary = DescribeYolo(int4.result),
                fp32ElapsedMs = fp32.result.elapsedMs,
                fp16ElapsedMs = fp16.result.elapsedMs,
                int4SelectiveElapsedMs = int4.result.elapsedMs,
                fp32GpuStats = fp32.stats,
                fp16GpuStats = fp16.stats,
                int4SelectiveGpuStats = int4.stats,
                peakRtBufferComparison = CreatePeakComparison(fp32.stats, fp16.stats, int4.stats),
                packageWeightComparison = CreateYoloPackageWeightComparison(),
                effectiveWeightComparison = CreateEffectiveWeightComparison(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Yolo", "yolov8n_seg.ncnn.param"),
                    "yolo-seg.int4.model.json"),
                int4SelectiveCoverage = CreateQuantizationCoverage(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Yolo", "yolov8n_seg.ncnn.param"),
                    "yolo-seg.int4.model.json"),
                fp16VsFp32 = CompareTextureChannel(fp32.result.mask, fp16.result.mask, 0),
                int4SelectiveVsFp32 = CompareTextureChannel(fp32.result.mask, int4.result.mask, 0),
                fp16CoverageDelta = fp16.result.maskCoverage01 - fp32.result.maskCoverage01,
                int4SelectiveCoverageDelta = int4.result.maskCoverage01 - fp32.result.maskCoverage01,
                status = fp32.result.personCount == int4.result.personCount ? "ok" : "person-count-delta"
            };
            return result;
        }
        finally
        {
            DestroyYoloRun(fp32);
            DestroyYoloRun(fp16);
            DestroyYoloRun(int4);
            DestroyImmediateSafe(input);
        }
    }

    private static async UniTask<NcnnInt4RunnerComparisonCase> RunClipComparisonAsync()
    {
        var input = LoadTexture(Input03Path);
        ClipRun fp32 = null;
        ClipRun fp16 = null;
        ClipRun int4 = null;
        try
        {
            fp32 = await RunClipAsync(input, NcnnPrecisionMode.FP32);
            fp16 = await RunClipAsync(input, NcnnPrecisionMode.FP16);
            int4 = await RunClipAsync(input, NcnnPrecisionMode.INT4Selective);

            var fp16CosineDistance = 1f - Cosine(fp32.result.imageEmbedding, fp16.result.imageEmbedding);
            var int4CosineDistance = 1f - Cosine(fp32.result.imageEmbedding, int4.result.imageEmbedding);
            var result = new NcnnInt4RunnerComparisonCase
            {
                runner = "mobileclip-s0",
                comparedOutput = "imageEmbedding",
                fp32Summary = DescribeClip(fp32.result),
                fp16Summary = DescribeClip(fp16.result),
                int4SelectiveSummary = DescribeClip(int4.result),
                fp32ElapsedMs = fp32.result.elapsedMs,
                fp16ElapsedMs = fp16.result.elapsedMs,
                int4SelectiveElapsedMs = int4.result.elapsedMs,
                fp32GpuStats = fp32.stats,
                fp16GpuStats = fp16.stats,
                int4SelectiveGpuStats = int4.stats,
                peakRtBufferComparison = CreatePeakComparison(fp32.stats, fp16.stats, int4.stats),
                packageWeightComparison = CreateClipPackageWeightComparison(),
                effectiveWeightComparison = CreateEffectiveWeightComparison(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Clip", "mobileclip_s0_export", "image_encoder.ncnn.param"),
                    "clip-mobileclip-s0.int4.model.json"),
                int4SelectiveCoverage = CreateQuantizationCoverage(
                    Path.Combine(Directory.GetCurrentDirectory(), "Assets", "StreamingAssets", "Clip", "mobileclip_s0_export", "image_encoder.ncnn.param"),
                    "clip-mobileclip-s0.int4.model.json"),
                fp16VsFp32 = CompareVector(fp32.result.imageEmbedding, fp16.result.imageEmbedding),
                int4SelectiveVsFp32 = CompareVector(fp32.result.imageEmbedding, int4.result.imageEmbedding),
                clipDiagnostics = CreateClipDiagnostics(fp32.result, fp16.result, int4.result),
                fp16CosineDistance = fp16CosineDistance,
                int4SelectiveCosineDistance = int4CosineDistance,
                status = string.Equals(fp32.result.bestLabel, int4.result.bestLabel, StringComparison.Ordinal) ? "ok" : "best-label-delta"
            };
            return result;
        }
        finally
        {
            DestroyClipRun(fp32);
            DestroyClipRun(fp16);
            DestroyClipRun(int4);
            DestroyImmediateSafe(input);
        }
    }

    private static async UniTask<MattingRun> RunMattingAsync(Texture2D input, NcnnPrecisionMode precision)
    {
        var previousTracker = NcnnGpuResourceTracker.Enabled;
        NcnnGpuResourceTracker.Enabled = true;
        NcnnGpuResourceTracker.Reset("int4-runner-matting-" + precision);
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
        NcnnGpuResourceTracker.Reset("int4-runner-yolo-" + precision);
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
        NcnnGpuResourceTracker.Reset("int4-runner-clip-" + precision);
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

    private static NcnnInt4RunnerGpuStats CaptureStats()
    {
        var stats = NcnnGpuResourceTracker.GetStatsSnapshot();
        return new NcnnInt4RunnerGpuStats
        {
            peakBufferBytes = stats.peakBufferBytes,
            peakTextureBytes = stats.peakTextureBytes,
            peakTotalBytes = stats.peakTotalBytes,
            peakTemporaryTextureBytes = stats.peakTemporaryTextureBytes,
            peakBufferCount = stats.peakBufferCount,
            peakTextureCount = stats.peakTextureCount
        };
    }

    private static NcnnInt4RunnerPeakRtBufferComparison CreatePeakComparison(
        NcnnInt4RunnerGpuStats fp32,
        NcnnInt4RunnerGpuStats fp16,
        NcnnInt4RunnerGpuStats int4Selective)
    {
        return new NcnnInt4RunnerPeakRtBufferComparison
        {
            fp32 = fp32,
            fp16 = fp16,
            int4Selective = int4Selective
        };
    }

    private static NcnnInt4RunnerPackageWeightComparison CreateMattingPackageWeightComparison()
    {
        var root = Directory.GetCurrentDirectory();
        var modelParam = Path.Combine(root, "Assets", "StreamingAssets", "Matting", "matting.param");
        var modelBin = Path.Combine(root, "Assets", "StreamingAssets", "Matting", "matting.bin");
        return CreatePackageWeightComparison(
            new[] { modelParam, modelBin, ManifestPath("matting.fp32.model.json") },
            new[] { modelParam, modelBin, ManifestPath("matting.fp16.model.json") },
            new[] { modelParam, modelBin, ManifestPath("matting.int4.model.json") });
    }

    private static NcnnInt4RunnerPackageWeightComparison CreateYoloPackageWeightComparison()
    {
        var root = Directory.GetCurrentDirectory();
        var modelParam = Path.Combine(root, "Assets", "StreamingAssets", "Yolo", "yolov8n_seg.ncnn.param");
        var modelBin = Path.Combine(root, "Assets", "StreamingAssets", "Yolo", "yolov8n_seg.ncnn.bin");
        return CreatePackageWeightComparison(
            new[] { modelParam, modelBin },
            new[] { modelParam, modelBin, ManifestPath("yolo-seg.fp16.model.json") },
            new[] { modelParam, modelBin, ManifestPath("yolo-seg.int4.model.json") });
    }

    private static NcnnInt4RunnerPackageWeightComparison CreateClipPackageWeightComparison()
    {
        var root = Directory.GetCurrentDirectory();
        var modelParam = Path.Combine(root, "Assets", "StreamingAssets", "Clip", "mobileclip_s0_export", "image_encoder.ncnn.param");
        var modelBin = Path.Combine(root, "Assets", "StreamingAssets", "Clip", "mobileclip_s0_export", "image_encoder.ncnn.bin");
        return CreatePackageWeightComparison(
            new[] { modelParam, modelBin, ManifestPath("clip-mobileclip-s0.fp32.model.json") },
            new[] { modelParam, modelBin, ManifestPath("clip-mobileclip-s0.fp16.model.json") },
            new[] { modelParam, modelBin, ManifestPath("clip-mobileclip-s0.int4.model.json") });
    }

    private static NcnnInt4RunnerPackageWeightComparison CreatePackageWeightComparison(
        string[] fp32,
        string[] fp16,
        string[] int4Selective)
    {
        return new NcnnInt4RunnerPackageWeightComparison
        {
            fp32 = SumExistingFileBytes(fp32),
            fp16 = SumExistingFileBytes(fp16),
            int4Selective = SumExistingFileBytes(int4Selective)
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

    private static NcnnInt4RunnerEffectiveWeightComparison CreateEffectiveWeightComparison(
        string paramPath,
        string manifestFileName)
    {
        var manifestPath = ManifestPath(manifestFileName);
        var manifest = NcnnModelManifestLoader.LoadFromFile(manifestPath);
        long totalWeightElements = 0;
        long int4SelectiveBytes = 0;

        foreach (var line in File.ReadLines(paramPath))
        {
            if (!TryParseWeightedLayer(line, out var operatorName, out var layerName, out var weightElements, out var outputChannels))
                continue;

            totalWeightElements += weightElements;
            if (manifest.TryGetQuantizedNodePlan(layerName, operatorName, out _))
            {
                int4SelectiveBytes += ((weightElements + 7) / 8) * sizeof(uint);
                int4SelectiveBytes += Math.Max(1, outputChannels) * sizeof(float);
            }
            else
            {
                int4SelectiveBytes += weightElements * sizeof(float);
            }
        }

        return new NcnnInt4RunnerEffectiveWeightComparison
        {
            fp32 = totalWeightElements * sizeof(float),
            fp16 = totalWeightElements * sizeof(ushort),
            int4Selective = int4SelectiveBytes
        };
    }

    private static NcnnInt4RunnerQuantizationCoverage CreateQuantizationCoverage(
        string paramPath,
        string manifestFileName)
    {
        var manifestPath = ManifestPath(manifestFileName);
        var manifest = NcnnModelManifestLoader.LoadFromFile(manifestPath);
        var coverage = new NcnnInt4RunnerQuantizationCoverage
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
                coverage.int4LayerCount++;
                coverage.int4WeightElements += weightElements;
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

        coverage.int4WeightCoverage = coverage.totalWeightElements > 0
            ? (float)((double)coverage.int4WeightElements / coverage.totalWeightElements)
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

    private static NcnnInt4RunnerError CompareTextureChannel(Texture2D expected, Texture2D actual, int channel)
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
        return new NcnnInt4RunnerError
        {
            maxAbsoluteError = max,
            meanAbsoluteError = a.Length > 0 ? (float)(sumAbs / a.Length) : 0f,
            rootMeanSquareError = a.Length > 0 ? (float)Math.Sqrt(sumSq / a.Length) : 0f
        };
    }

    private static NcnnInt4RunnerError CompareVector(float[] expected, float[] actual)
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
        return new NcnnInt4RunnerError
        {
            maxAbsoluteError = max,
            meanAbsoluteError = expected.Length > 0 ? (float)(sumAbs / expected.Length) : 0f,
            rootMeanSquareError = expected.Length > 0 ? (float)Math.Sqrt(sumSq / expected.Length) : 0f
        };
    }

    private static NcnnInt4RunnerClipDiagnostics CreateClipDiagnostics(
        ClipClassificationResult fp32,
        ClipClassificationResult fp16,
        ClipClassificationResult int4)
    {
        var labels = fp32.scores ?? Array.Empty<ClipLabelScore>();
        var rows = new NcnnInt4RunnerClipScoreDelta[labels.Length];
        double fp16AbsLogit = 0d;
        double int4AbsLogit = 0d;
        double fp16AbsProbability = 0d;
        double int4AbsProbability = 0d;
        var fp16MaxLogit = 0f;
        var int4MaxLogit = 0f;
        var fp16MaxProbability = 0f;
        var int4MaxProbability = 0f;

        for (var i = 0; i < labels.Length; i++)
        {
            var fp32Score = labels[i];
            var fp16Score = FindClipScore(fp16.scores, fp32Score.label);
            var int4Score = FindClipScore(int4.scores, fp32Score.label);
            var fp16LogitDelta = fp16Score.similarity - fp32Score.similarity;
            var int4LogitDelta = int4Score.similarity - fp32Score.similarity;
            var fp16ProbabilityDelta = fp16Score.probability - fp32Score.probability;
            var int4ProbabilityDelta = int4Score.probability - fp32Score.probability;

            var fp16AbsLogitDelta = Mathf.Abs(fp16LogitDelta);
            var int4AbsLogitDelta = Mathf.Abs(int4LogitDelta);
            var fp16AbsProbabilityDelta = Mathf.Abs(fp16ProbabilityDelta);
            var int4AbsProbabilityDelta = Mathf.Abs(int4ProbabilityDelta);
            fp16AbsLogit += fp16AbsLogitDelta;
            int4AbsLogit += int4AbsLogitDelta;
            fp16AbsProbability += fp16AbsProbabilityDelta;
            int4AbsProbability += int4AbsProbabilityDelta;
            if (fp16AbsLogitDelta > fp16MaxLogit)
                fp16MaxLogit = fp16AbsLogitDelta;
            if (int4AbsLogitDelta > int4MaxLogit)
                int4MaxLogit = int4AbsLogitDelta;
            if (fp16AbsProbabilityDelta > fp16MaxProbability)
                fp16MaxProbability = fp16AbsProbabilityDelta;
            if (int4AbsProbabilityDelta > int4MaxProbability)
                int4MaxProbability = int4AbsProbabilityDelta;

            rows[i] = new NcnnInt4RunnerClipScoreDelta
            {
                label = fp32Score.label,
                prompt = fp32Score.prompt,
                fp32Logit = fp32Score.similarity,
                fp16Logit = fp16Score.similarity,
                int4SelectiveLogit = int4Score.similarity,
                fp16LogitDeltaVsFp32 = fp16LogitDelta,
                int4SelectiveLogitDeltaVsFp32 = int4LogitDelta,
                fp32Probability = fp32Score.probability,
                fp16Probability = fp16Score.probability,
                int4SelectiveProbability = int4Score.probability,
                fp16ProbabilityDeltaVsFp32 = fp16ProbabilityDelta,
                int4SelectiveProbabilityDeltaVsFp32 = int4ProbabilityDelta
            };
        }

        return new NcnnInt4RunnerClipDiagnostics
        {
            fp32Top1 = CreateClipTopScore(fp32.scores, 0),
            fp32Top2 = CreateClipTopScore(fp32.scores, 1),
            fp16Top1 = CreateClipTopScore(fp16.scores, 0),
            fp16Top2 = CreateClipTopScore(fp16.scores, 1),
            int4SelectiveTop1 = CreateClipTopScore(int4.scores, 0),
            int4SelectiveTop2 = CreateClipTopScore(int4.scores, 1),
            fp32Top1Top2LogitMargin = ClipTopMargin(fp32.scores, true),
            fp16Top1Top2LogitMargin = ClipTopMargin(fp16.scores, true),
            int4SelectiveTop1Top2LogitMargin = ClipTopMargin(int4.scores, true),
            fp32Top1Top2ProbabilityMargin = ClipTopMargin(fp32.scores, false),
            fp16Top1Top2ProbabilityMargin = ClipTopMargin(fp16.scores, false),
            int4SelectiveTop1Top2ProbabilityMargin = ClipTopMargin(int4.scores, false),
            fp16Top1LogitDeltaVsFp32 = FindClipScore(fp16.scores, fp32.bestLabel).similarity - FindClipScore(fp32.scores, fp32.bestLabel).similarity,
            int4SelectiveTop1LogitDeltaVsFp32 = FindClipScore(int4.scores, fp32.bestLabel).similarity - FindClipScore(fp32.scores, fp32.bestLabel).similarity,
            fp16Top1ProbabilityDeltaVsFp32 = FindClipScore(fp16.scores, fp32.bestLabel).probability - FindClipScore(fp32.scores, fp32.bestLabel).probability,
            int4SelectiveTop1ProbabilityDeltaVsFp32 = FindClipScore(int4.scores, fp32.bestLabel).probability - FindClipScore(fp32.scores, fp32.bestLabel).probability,
            fp16MeanAbsLogitDeltaVsFp32 = rows.Length > 0 ? (float)(fp16AbsLogit / rows.Length) : 0f,
            int4SelectiveMeanAbsLogitDeltaVsFp32 = rows.Length > 0 ? (float)(int4AbsLogit / rows.Length) : 0f,
            fp16MaxAbsLogitDeltaVsFp32 = fp16MaxLogit,
            int4SelectiveMaxAbsLogitDeltaVsFp32 = int4MaxLogit,
            fp16MeanAbsProbabilityDeltaVsFp32 = rows.Length > 0 ? (float)(fp16AbsProbability / rows.Length) : 0f,
            int4SelectiveMeanAbsProbabilityDeltaVsFp32 = rows.Length > 0 ? (float)(int4AbsProbability / rows.Length) : 0f,
            fp16MaxAbsProbabilityDeltaVsFp32 = fp16MaxProbability,
            int4SelectiveMaxAbsProbabilityDeltaVsFp32 = int4MaxProbability,
            labels = rows
        };
    }

    private static ClipLabelScore FindClipScore(ClipLabelScore[] scores, string label)
    {
        if (scores != null)
        {
            for (var i = 0; i < scores.Length; i++)
            {
                if (string.Equals(scores[i].label, label, StringComparison.Ordinal))
                    return scores[i];
            }
        }
        return default;
    }

    private static NcnnInt4RunnerClipTopScore CreateClipTopScore(ClipLabelScore[] scores, int index)
    {
        if (scores == null || index < 0 || index >= scores.Length)
            return null;
        return new NcnnInt4RunnerClipTopScore
        {
            label = scores[index].label,
            logit = scores[index].similarity,
            probability = scores[index].probability
        };
    }

    private static float ClipTopMargin(ClipLabelScore[] scores, bool logit)
    {
        if (scores == null || scores.Length < 2)
            return 0f;
        return logit
            ? scores[0].similarity - scores[1].similarity
            : scores[0].probability - scores[1].probability;
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
        public NcnnInt4RunnerGpuStats stats;
    }

    private sealed class YoloRun
    {
        public GameObject go;
        public YoloSegResult result;
        public NcnnInt4RunnerGpuStats stats;
    }

    private sealed class ClipRun
    {
        public GameObject go;
        public ClipClassificationResult result;
        public NcnnInt4RunnerGpuStats stats;
    }
}
#endif

