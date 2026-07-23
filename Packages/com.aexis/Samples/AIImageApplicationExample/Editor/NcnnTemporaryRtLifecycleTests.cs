#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.IO;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;
using Aexis.Execution;

public sealed class NcnnTemporaryRtLifecycleTests
{
    [Test]
    public void CommandBufferTemporaryRt_BranchLifetimesPairWithoutBuffers()
    {
        var previousEnabled = AexisGpuResourceTracker.Enabled;
        AexisGpuResourceTracker.Enabled = true;
        AexisGpuResourceTracker.Reset("cmd-temporary-rt-branch");

        var repro = new AexisGraphSession(new AexisOps());
        using var cmd = new CommandBuffer { name = "NcnnTemporaryRtLifecycleTests" };
        try
        {
            var input = repro.RentTempArray(cmd, 8, 8, 1, RenderTextureFormat.ARGBHalf);
            var branchA = repro.RentTempArray(cmd, 8, 8, 1, RenderTextureFormat.ARGBHalf);
            var branchB = repro.RentTempArray(cmd, 8, 8, 1, RenderTextureFormat.ARGBHalf);

            cmd.CopyTexture(input.nameID, branchA.nameID);
            repro.ReturnTempArray(cmd, branchA);
            cmd.CopyTexture(input.nameID, branchB.nameID);
            repro.ReturnTempArray(cmd, input);
            repro.ReturnTempArray(cmd, branchB);

            Graphics.ExecuteCommandBuffer(cmd);

            var snapshot = AexisGpuResourceTracker.GetStatsSnapshot();
            var timeline = string.Join("\n", AexisGpuResourceTracker.GetTimelineSnapshot());
            Assert.That(snapshot.liveTemporaryTextureCount, Is.EqualTo(0));
            Assert.That(snapshot.currentTemporaryTextureBytes, Is.EqualTo(0));
            Assert.That(snapshot.liveBufferCount, Is.EqualTo(0));
            Assert.That(CountOccurrences(timeline, "alloc_temp_rt"), Is.EqualTo(3));
            Assert.That(CountOccurrences(timeline, "free_temp_rt"), Is.EqualTo(3));
            Assert.That(timeline, Does.Not.Contain("alloc_buffer"));
            Assert.That(AexisGpuResourceTracker.GetUnreleasedTemporaryTextureDiagnostics(), Is.Empty);
        }
        finally
        {
            repro.Dispose();
            AexisGpuResourceTracker.Reset("cmd-temporary-rt-branch-cleanup");
            AexisGpuResourceTracker.Enabled = previousEnabled;
        }
    }

    [Test]
    public void TemporaryRtBudgetError_ReportsRequestPeakAndNode()
    {
        var previousEnabled = AexisGpuResourceTracker.Enabled;
        AexisGpuResourceTracker.Enabled = true;
        AexisGpuResourceTracker.Reset("temporary-rt-budget");
        try
        {
            var shape = new AexisGraphSession.BufferShape(3, 8, 8, 1, 4);
            var descriptor = new RenderTextureDescriptor(8, 8, RenderTextureFormat.ARGBHalf, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = 1,
                enableRandomWrite = true
            };
            var allocation = new AexisTemporaryRtDescriptor(shape, shape, descriptor, "test-owner", "budget-node", "budget-allocation");
            var requiredBytes = AexisGpuResourceTracker.EstimateTemporaryTextureBytes(allocation);
            AexisGpuResourceTracker.RegisterTemporaryTextureHandle(1234, allocation);

            var error = Assert.Throws<InvalidOperationException>(
                () => AexisGpuResourceTracker.EnsureTemporaryTextureBudget(requiredBytes, requiredBytes, "budget-node"));

            Assert.That(error.Message, Does.Contain("required_bytes=" + requiredBytes));
            Assert.That(error.Message, Does.Contain("current_peak_bytes=" + requiredBytes));
            Assert.That(error.Message, Does.Contain("node=budget-node"));
            AexisGpuResourceTracker.ReleaseTextureHandle(1234, "budget-test-release");
        }
        finally
        {
            AexisGpuResourceTracker.Reset("temporary-rt-budget-cleanup");
            AexisGpuResourceTracker.Enabled = previousEnabled;
        }
    }

    [Test]
    public void TemporaryRtPaths_UseUnityApisAndContainNoPool()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var repro = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Execution", "Graph", "AexisGraphSession.cs"));
        var tracker = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Execution", "Graph", "AexisGpuResourceTracker.cs"));

        Assert.That(repro, Does.Contain("RenderTexture.GetTemporary(desc)"));
        Assert.That(repro, Does.Contain("RenderTexture.ReleaseTemporary(rt)"));
        Assert.That(repro, Does.Contain("cmd.GetTemporaryRT(id, desc)"));
        Assert.That(repro, Does.Contain("cmd.ReleaseTemporaryRT(t.nameID)"));
        Assert.That(repro, Does.Not.Contain("TempPool"));
        Assert.That(repro, Does.Not.Contain("RtPool"));
        Assert.That(repro, Does.Not.Contain("new RenderTexture("));
        Assert.That(repro, Does.Contain("if (IsDebugOracleExecution && !DisallowInferenceTempComputeBuffers)"));
        Assert.That(tracker, Does.Contain("logical="));
        Assert.That(tracker, Does.Contain("storage="));
        Assert.That(tracker, Does.Contain("graphics_format="));
        Assert.That(tracker, Does.Contain("random_write="));
        Assert.That(tracker, Does.Not.Contain("ReuseTexture("));
        Assert.That(tracker, Does.Not.Contain("ReuseBuffer("));
        Assert.That(File.Exists(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Execution", "Graph", "NcnnTempComputeBufferPool.cs")), Is.False);
    }

    public static void RunBatchValidation()
    {
        var tests = new NcnnTemporaryRtLifecycleTests();
        tests.CommandBufferTemporaryRt_BranchLifetimesPairWithoutBuffers();
        tests.TemporaryRtBudgetError_ReportsRequestPeakAndNode();
        tests.TemporaryRtPaths_UseUnityApisAndContainNoPool();
        Debug.Log("[NcnnTemporaryRtLifecycleTests] passed");
    }

    private static int CountOccurrences(string source, string value)
    {
        var count = 0;
        var index = 0;
        while ((index = source.IndexOf(value, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += value.Length;
        }
        return count;
    }
}
#endif
