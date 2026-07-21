#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEngine;

public sealed class Qwen35ArchitectureBoundaryTests
{
    private static readonly string[] IntegrationSources =
    {
        "Qwen35ByteLevelBpeTokenizer.cs",
        "Qwen35CompareManifest.cs",
        "Qwen35DecoderSession.cs",
        "Qwen35DeviceCompatibility.cs",
        "Qwen35MobileAssetSet.cs",
        "Qwen35MobileMemoryPolicy.cs",
        "Qwen35ModelContract.cs",
        "Qwen35NetworkAssetCatalog.cs",
        "Qwen35NetworkLoader.cs",
        "Qwen35Runner.cs",
        "Qwen35SharedTokenEmbeddingWeights.cs",
        "Qwen35VisionEncoderSession.cs",
        "Qwen35VisionPreprocessor.cs"
    };

    [Test]
    public void ReusableInferencePackages_DoNotContainQwen35IntegrationCode()
    {
        var projectRoot = ProjectRoot();
        var packageRoots = new[]
        {
            Path.Combine(projectRoot, "Packages", "com.aexis", "Runtime")
        };

        foreach (var packageRoot in packageRoots)
        {
            var modelFiles = Directory.GetFiles(packageRoot, "*", SearchOption.AllDirectories)
                .Where(path => string.Equals(Path.GetExtension(path), ".cs", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(Path.GetExtension(path), ".compute", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(Path.GetExtension(path), ".hlsl", StringComparison.OrdinalIgnoreCase))
                .Where(path => File.ReadAllText(path).IndexOf("Qwen35", StringComparison.OrdinalIgnoreCase) >= 0
                    || File.ReadAllText(path).IndexOf("Qwen3.5", StringComparison.OrdinalIgnoreCase) >= 0)
                .ToArray();

            Assert.That(modelFiles, Is.Empty, "Model-specific Qwen3.5 code must stay outside reusable inference packages.");
        }
    }

    [Test]
    public void Qwen35IntegrationSources_LiveBesideApplicationRunners()
    {
        var scriptsRoot = Path.Combine(AexisApplicationExamplePaths.SampleRootAbsolutePath, "Runtime", "Scripts");
        foreach (var source in IntegrationSources)
            Assert.That(File.Exists(Path.Combine(scriptsRoot, source)), Is.True, source + " must live under the package application sample.");
    }

    [Test]
    public void ReusablePackage_KeepsOnlyGenericRecurrentOperatorLayers()
    {
        var runtimeRoot = Path.Combine(
            ProjectRoot(),
            "Packages",
            "com.aexis",
            "Runtime",
            "Ncnn");
        Assert.That(File.Exists(Path.Combine(runtimeRoot, "Layers", "NcnnShortConvLayerRepro.cs")), Is.True);
        Assert.That(File.Exists(Path.Combine(runtimeRoot, "Layers", "NcnnGatedDeltaRuleLayerRepro.cs")), Is.True);
    }

    private static string ProjectRoot()
    {
        return Directory.GetParent(Application.dataPath)?.FullName
            ?? throw new InvalidOperationException("Unable to resolve the Unity project root.");
    }
}
#endif
