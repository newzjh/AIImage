using System;
using System.Collections;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Cysharp.Threading.Tasks;
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
    public IEnumerator Clip_Runs_On_02()
    {
        var input = LoadTexture(Input02Path);
        var go = new GameObject("ClipRunnerTest");
        var runner = go.AddComponent<ClipNcnnReproRunner>();
        runner.enableDebugDump = false;
        runner.enableTempPool = false;
        runner.maxPooledPerShape = 0;

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
}
