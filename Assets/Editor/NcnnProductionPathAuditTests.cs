using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

public sealed class NcnnProductionPathAuditTests
{
    private static readonly string[] ForbiddenPublicReadbackApis =
    {
        "GetBufferData(",
        "public ComputeBuffer GetBuffer("
    };

    private static readonly string[] LegacyBufferMaterializationApis =
    {
        "GetOrConvertToBuffer(",
        "Pack4ToBufferCHW(",
        "Pack4ToBufferCDHW(",
        "TextureToBuffer3(",
        "BufferToTexture3("
    };

    // These are the remaining explicitly audited legacy runner diagnostics. Any new use,
    // including an additional use in one of these files, must update the A2 audit deliberately.
    private static readonly Dictionary<string, int[]> AuditedLegacyRunnerApiCounts = new Dictionary<string, int[]>
    {
        { "Assets/Scripts/ClipNcnnReproRunner.cs", new[] { 0, 2, 0, 0, 0 } },
        { "Assets/Scripts/CodeFormerNcnnReproRunner2.cs", new[] { 0, 2, 0, 0, 0 } },
        { "Assets/Scripts/MatterNcnnReproRunner.cs", new[] { 0, 2, 0, 0, 0 } },
        { "Assets/Scripts/MONAINcnnReproRunner.cs", new[] { 0, 2, 3, 0, 0 } },
        { "Assets/Scripts/SDInpaintingNcnnReproRunner.cs", new[] { 0, 6, 2, 0, 0 } },
        { "Assets/Scripts/SDNcnnReproRunner.cs", new[] { 0, 4, 0, 0, 0 } },
        { "Assets/Scripts/YoloSegNcnnReproRunner.cs", new[] { 0, 1, 0, 0, 0 } }
    };

    [Test]
    public void InferResult_DoesNotExposeProductionBufferReadback()
    {
        var source = ReadAssetSource("Scripts/NcnnCompute/NcnnRepro.cs");
        foreach (var api in ForbiddenPublicReadbackApis)
            Assert.That(source, Does.Not.Contain(api), "Production InferResult must not expose " + api);

        Assert.That(source, Does.Contain("ReadTextureDataForOutput"));
        Assert.That(source, Does.Contain("#if UNITY_EDITOR || AIIMAGE_INFERENCE_DEBUG_ORACLE"));
        Assert.That(source, Does.Contain("public NcnnInferenceExecutionMode ExecutionMode { get; set; } = NcnnInferenceExecutionMode.ProductionTextureOnly;"));
    }

    [Test]
    public void ProductionBoundary_RejectsBufferInputsAndMaterializationWithTensorContract()
    {
        var repro = ReadAssetSource("Scripts/NcnnCompute/NcnnRepro.cs");
        var factory = ReadAssetSource("Scripts/NcnnLayers/NcnnLayerFactoryRepro.cs");

        Assert.That(repro, Does.Contain("if (!IsDebugOracleExecution)\n                return true;"));
        Assert.That(repro, Does.Contain("logical_shape="));
        Assert.That(repro, Does.Contain("storage_shape="));
        Assert.That(repro, Does.Contain("layout="));
        Assert.That(repro, Does.Contain("dtype="));
        Assert.That(repro, Does.Contain("rejected_fallback="));
        Assert.That(factory, Does.Not.Contain("rejects ComputeBuffer model input"));
        Assert.That(factory, Does.Contain("MaterializeScratchTextureFromBufferView"));
        Assert.That(factory, Does.Contain("SetCurrentBufferExecutionContext(context)"));
    }

    [Test]
    public void LegacyBufferCalls_AreConfinedToTheAuditedEngineDirectories()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var allowedRoots = new[]
        {
            Path.Combine(root, "Assets", "Scripts", "NcnnCompute"),
            Path.Combine(root, "Assets", "Scripts", "NcnnLayers")
        };
        var violations = new List<string>();

        foreach (var file in Directory.EnumerateFiles(Path.Combine(root, "Assets", "Scripts"), "*.cs", SearchOption.AllDirectories))
        {
            if (Array.Exists(allowedRoots, allowedRoot => file.StartsWith(allowedRoot, StringComparison.OrdinalIgnoreCase)))
                continue;

            var source = File.ReadAllText(file);
            var relativePath = MakeAssetRelative(root, file);
            var observed = new int[LegacyBufferMaterializationApis.Length];
            var total = 0;
            for (var i = 0; i < LegacyBufferMaterializationApis.Length; i++)
            {
                observed[i] = CountOccurrences(source, LegacyBufferMaterializationApis[i]);
                total += observed[i];
            }

            if (total == 0)
                continue;

            if (!AuditedLegacyRunnerApiCounts.TryGetValue(relativePath, out var expected))
            {
                violations.Add(relativePath + " => unapproved legacy Buffer API");
                continue;
            }

            for (var i = 0; i < observed.Length; i++)
            {
                if (observed[i] != expected[i])
                    violations.Add(relativePath + " => " + LegacyBufferMaterializationApis[i] + " expected=" + expected[i] + " actual=" + observed[i]);
            }
        }

        Assert.That(violations, Is.Empty, "Forbidden Buffer materialization escaped the audited engine/debug boundary:\n" + string.Join("\n", violations));
    }

    [Test]
    public void ProductionAuditLog_StatesThatIntermediateBufferMaterializationIsZero()
    {
        var source = ReadAssetSource("Scripts/NcnnCompute/NcnnRepro.cs");
        Assert.That(source, Does.Contain("[InferencePathAudit] mode=ProductionTextureOnly"));
        Assert.That(source, Does.Contain("intermediate_buffer_materializations=0"));
    }

    public static void RunBatchValidation()
    {
        var tests = new NcnnProductionPathAuditTests();
        tests.InferResult_DoesNotExposeProductionBufferReadback();
        tests.ProductionBoundary_RejectsBufferInputsAndMaterializationWithTensorContract();
        tests.LegacyBufferCalls_AreConfinedToTheAuditedEngineDirectories();
        tests.ProductionAuditLog_StatesThatIntermediateBufferMaterializationIsZero();
        Debug.Log("[NcnnProductionPathAudit] passed");
    }

    private static string ReadAssetSource(string relativePath)
    {
        return File.ReadAllText(Path.Combine(Application.dataPath, relativePath));
    }

    private static string MakeAssetRelative(string root, string path)
    {
        return path.Substring(root.Length + 1).Replace('\\', '/');
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
