using System;
using System.Collections;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Aexis.Samples.Async;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

public sealed class NcnnRunnerBatchValidationTests
{
    private static readonly string Input02Path = Path.Combine(Directory.GetCurrentDirectory(), "documents", "ClipCompareInput", "02.png");
    private static readonly string Input03Path = Path.Combine(Directory.GetCurrentDirectory(), "documents", "ClipCompareInput", "03.jpg");

    [UnityTest]
    public IEnumerator CodeFormer_Runs_On_02()
    {
        var input = LoadTexture(Input02Path);
        var go = new GameObject("CodeFormerRunnerTest");
        var runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
        runner.enableDebugDump = false;
        runner.enableFaceRegionDebugDump = false;

        var task = runner.ProcessAsync(input, CancellationToken.None).AsTask();
        yield return WaitForTask(task);

        try
        {
            Assert.That(task.Result.error, Is.Null.Or.Empty, "CodeFormer error");
            Assert.That(task.Result.texture, Is.Not.Null, "CodeFormer output texture");
            Debug.Log("[NcnnRunnerBatchValidationTests] CodeFormer elapsedMs=" + task.Result.elapsedMs);
        }
        finally
        {
            DestroyImmediateSafe(task.Result.texture);
            DestroyImmediateSafe(go);
            DestroyImmediateSafe(input);
        }
    }

    [UnityTest]
    public IEnumerator Gfpgan_Runs_On_02()
    {
        var input = LoadTexture(Input02Path);
        var go = new GameObject("GfpganRunnerTest");
        var runner = go.AddComponent<GfpganNcnnReproRunner>();
        runner.enableFaceRegionDebugDump = false;

        var task = runner.ProcessAsync(input, CancellationToken.None).AsTask();
        yield return WaitForTask(task);

        try
        {
            Assert.That(task.Result.error, Is.Null.Or.Empty, "GFPGAN error");
            Assert.That(task.Result.texture, Is.Not.Null, "GFPGAN output texture");
            Debug.Log("[NcnnRunnerBatchValidationTests] GFPGAN elapsedMs=" + task.Result.elapsedMs);
        }
        finally
        {
            DestroyImmediateSafe(task.Result.texture);
            DestroyImmediateSafe(go);
            DestroyImmediateSafe(input);
        }
    }

    [UnityTest]
    public IEnumerator Matting_Runs_On_03()
    {
        var input = LoadTexture(Input03Path);
        var go = new GameObject("MattingRunnerTest");
        var runner = go.AddComponent<MatterNcnnReproRunner>();
        runner.enableDebugDump = false;
        runner.forceBufferConvolution = false;

        var task = runner.ProcessAsync(input, CancellationToken.None).AsTask();
        yield return WaitForTask(task);

        try
        {
            Assert.That(task.Result.error, Is.Null.Or.Empty, "Matting error");
            Assert.That(task.Result.texture, Is.Not.Null, "Matting composite");
            Assert.That(task.Result.matte, Is.Not.Null, "Matting matte");
            Debug.Log("[NcnnRunnerBatchValidationTests] Matting elapsedMs=" + task.Result.elapsedMs);
        }
        finally
        {
            DestroyImmediateSafe(task.Result.texture);
            DestroyImmediateSafe(task.Result.matte);
            DestroyImmediateSafe(go);
            DestroyImmediateSafe(input);
        }
    }

    [UnityTest]
    public IEnumerator YoloSeg_Runs_On_03()
    {
        var input = LoadTexture(Input03Path);
        var go = new GameObject("YoloSegRunnerTest");
        var runner = go.AddComponent<YoloSegNcnnReproRunner>();
        runner.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
        runner.enableDebugDump = false;
        runner.targetPersonOnly = true;
        runner.enableMaskClose = true;
        runner.enableMaskDilate = true;

        var task = runner.ProcessAsync(input, CancellationToken.None).AsTask();
        yield return WaitForTask(task);

        try
        {
            Assert.That(task.Result.error, Is.Null.Or.Empty, "YoloSeg error");
            Assert.That(task.Result.texture, Is.Not.Null, "YoloSeg transparent cutout");
            Assert.That(task.Result.mask, Is.Not.Null, "YoloSeg mask");
            Assert.That(task.Result.overlay, Is.Not.Null, "YoloSeg overlay");
            Debug.Log("[NcnnRunnerBatchValidationTests] YoloSeg elapsedMs=" + task.Result.elapsedMs + " personCount=" + task.Result.personCount + " coverage=" + task.Result.maskCoverage01);
        }
        finally
        {
            DestroyImmediateSafe(task.Result.texture);
            DestroyImmediateSafe(task.Result.mask);
            DestroyImmediateSafe(task.Result.overlay);
            DestroyImmediateSafe(go);
            DestroyImmediateSafe(input);
        }
    }

    [UnityTest]
    public IEnumerator YoloSegAndInpainting_Runs_On_02()
    {
        var input = LoadTexture(Input02Path);
        var go = new GameObject("YoloInpaintingRunnerTest");
        var yoloRunner = go.AddComponent<YoloSegNcnnReproRunner>();
        yoloRunner.modelVariant = YoloSegNcnnReproRunner.YoloSegModelVariant.YoloV8nSeg;
        yoloRunner.enableDebugDump = false;
        yoloRunner.targetPersonOnly = true;
        yoloRunner.enableMaskClose = true;
        yoloRunner.enableMaskDilate = true;

        var inpaintRunner = go.AddComponent<SDInpaintingNcnnReproRunner>();
        inpaintRunner.enableDebugDump = false;
        inpaintRunner.useOfficialUnetCache = false;
        inpaintRunner.defaultStepCount = 8;
        inpaintRunner.defaultStrength = 1f;
        inpaintRunner.defaultGuidanceScale = 7.5f;

        var task = RunYoloAndInpaintingAsync(input, yoloRunner, inpaintRunner, CancellationToken.None).AsTask();
        yield return WaitForTask(task);

        try
        {
            var result = task.Result;
            Assert.That(result.yolo.error, Is.Null.Or.Empty, "YOLO error");
            Assert.That(result.yolo.mask, Is.Not.Null, "YOLO mask");
            Assert.That(result.yolo.personCount, Is.EqualTo(4), "YOLO person count");
            Assert.That(result.inpaint.error, Is.Null.Or.Empty, "Inpainting error");
            Assert.That(result.inpaint.texture, Is.Not.Null, "Inpainting output texture");
            Assert.That(result.maskedPixels, Is.GreaterThan(0), "Masked pixels");
            Assert.That(result.maskedMeanAbsDiffRgb, Is.GreaterThan(1f), "Masked mean abs diff");
            Debug.Log("[NcnnRunnerBatchValidationTests] YOLO+Inpainting elapsedMs=" + result.inpaint.elapsedMs + " personCount=" + result.yolo.personCount + " maskedDiff=" + result.maskedMeanAbsDiffRgb.ToString("0.0000"));
        }
        finally
        {
            DestroyImmediateSafe(task.Result.yolo.texture);
            DestroyImmediateSafe(task.Result.yolo.mask);
            DestroyImmediateSafe(task.Result.yolo.overlay);
            DestroyImmediateSafe(task.Result.inpaint.texture);
            DestroyImmediateSafe(go);
            DestroyImmediateSafe(input);
        }
    }

    [UnityTest]
    public IEnumerator Clip_Runs_On_02()
    {
        var input = LoadTexture(Input02Path);
        var go = new GameObject("ClipRunnerTest");
        var runner = go.AddComponent<ClipNcnnReproRunner>();
        runner.enableDebugDump = false;
        var task = runner.ProcessAsync(input, CancellationToken.None).AsTask();
        yield return WaitForTask(task);

        try
        {
            Assert.That(task.Result.error, Is.Null.Or.Empty, "CLIP error");
            Assert.That(task.Result.scores, Is.Not.Null.And.Length.GreaterThan(0), "CLIP scores");
            Assert.That(task.Result.bestLabel, Is.Not.Null.And.Not.Empty, "CLIP best label");
            Debug.Log("[NcnnRunnerBatchValidationTests] CLIP elapsedMs=" + task.Result.elapsedMs + " best=" + task.Result.bestLabel + " prob=" + task.Result.bestProbability);
        }
        finally
        {
            DestroyImmediateSafe(go);
            DestroyImmediateSafe(input);
        }
    }

    [UnityTest]
    public IEnumerator RealEsrgan_Runs_On_02()
    {
        var input = LoadTexture(Input02Path);
        var go = new GameObject("RealEsrganRunnerTest");
        var runner = go.AddComponent<RealEsrganNcnnReproRunner>();
        runner.enableGpuLayerProfiling = false;
        runner.useCommandBuffer = false;

        var task = runner.ProcessAsync(input, CancellationToken.None).AsTask();
        yield return WaitForTask(task);

        try
        {
            Assert.That(task.Result.error, Is.Null.Or.Empty, "RealESRGAN error");
            Assert.That(task.Result.texture, Is.Not.Null, "RealESRGAN output texture");
            Debug.Log("[NcnnRunnerBatchValidationTests] RealESRGAN elapsedMs=" + task.Result.elapsedMs);
        }
        finally
        {
            DestroyImmediateSafe(task.Result.texture);
            DestroyImmediateSafe(go);
            DestroyImmediateSafe(input);
        }
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

    private static async UniTask<(YoloSegResult yolo, SDInpaintingNcnnReproResult inpaint, float maskedMeanAbsDiffRgb, int maskedPixels)> RunYoloAndInpaintingAsync(
        Texture2D input,
        YoloSegNcnnReproRunner yoloRunner,
        SDInpaintingNcnnReproRunner inpaintRunner,
        CancellationToken ct)
    {
        var yoloResult = await yoloRunner.ProcessAsync(input, ct);
        if (!string.IsNullOrWhiteSpace(yoloResult.error) || yoloResult.mask == null)
            return (yoloResult, default, 0f, 0);

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

        UnityEngine.Object.DestroyImmediate(yoloRunner);
        await UniTask.Yield();

        var inpaintResult = await inpaintRunner.ProcessAsync(
            input,
            yoloResult.mask,
            inpaintRunner.defaultPositivePrompt,
            inpaintRunner.defaultNegativePrompt,
            inpaintRunner.defaultStepCount,
            123456,
            inpaintRunner.defaultStrength,
            inpaintRunner.defaultGuidanceScale,
            ct);

        var maskedDiff = ComputeMaskedMeanAbsDiff(
            input,
            inpaintResult.texture,
            yoloResult.mask,
            inpaintRunner.blackMaskMeansInpaint,
            out var maskedPixels);
        return (yoloResult, inpaintResult, maskedDiff, maskedPixels);
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
            UnityEngine.Object.DestroyImmediate(tex);
            Assert.Fail("Failed to decode input image: " + path);
        }

        return tex;
    }

    private static void DestroyImmediateSafe(UnityEngine.Object obj)
    {
        if (obj != null)
            UnityEngine.Object.DestroyImmediate(obj);
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
}
