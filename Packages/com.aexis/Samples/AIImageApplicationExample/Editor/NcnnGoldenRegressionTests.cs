#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.Diagnostics;
using System.IO;
using Aexis.Samples.Json.Linq;
using NUnit.Framework;
using UnityEngine;

public sealed class NcnnGoldenRegressionTests
{
    private const string ManifestSchema = "aiimage.inference.golden/v1";
    private const string ReportSchema = "aiimage.inference.golden-report/v1";

    [Test]
    public void GoldenManifests_DeclareContractsThresholdsAndPlatformInformation()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var manifests = Directory.GetFiles(Path.Combine(root, "Tools", "GoldenRegression", "manifests"), "*.golden.json", SearchOption.TopDirectoryOnly);
        Assert.That(manifests, Has.Length.GreaterThanOrEqualTo(6));

        var requiredModels = new[] { "clip", "matting", "yolo-seg", "wholebrain" };
        var modelIds = new System.Collections.Generic.List<string>();
        foreach (var path in manifests)
        {
            var manifest = JObject.Parse(File.ReadAllText(path));
            Assert.That((string)manifest["schema_version"], Is.EqualTo(ManifestSchema), path);
            Assert.That((string)manifest["case_id"], Is.Not.Null.And.Not.Empty, path);
            Assert.That(new[] { "single_layer", "subgraph", "model" }, Has.Member((string)manifest["scope"]), path);
            Assert.That(new[] { "FP32", "FP16", "INT8" }, Has.Member((string)manifest["precision"]), path);
            Assert.That(manifest["input_fixtures"]?.Type, Is.EqualTo(JTokenType.Array), path);
            Assert.That(manifest["expected_tensors"]?.Type, Is.EqualTo(JTokenType.Array), path);
            Assert.That(manifest["platform"]?.Type, Is.EqualTo(JTokenType.Object), path);
            Assert.That(manifest["thresholds"]?.Type, Is.EqualTo(JTokenType.Object), path);
            foreach (var tensor in manifest["expected_tensors"])
            {
                Assert.That((string)tensor["node"], Is.Not.Null.And.Not.Empty, path);
                Assert.That((string)tensor["blob"], Is.Not.Null.And.Not.Empty, path);
                Assert.That((string)tensor["oracle_fixture"], Is.Not.Null.And.Not.Empty, path);
            }
            if ((string)manifest["scope"] == "model")
                modelIds.Add((string)manifest["case_id"]);
        }

        foreach (var model in requiredModels)
            Assert.That(modelIds, Has.Some.Contains(model));

        var wholeBrain = JObject.Parse(File.ReadAllText(Path.Combine(root, "Tools", "GoldenRegression", "manifests", "model-wholebrain-probe.golden.json")));
        Assert.That((string)wholeBrain["privacy"], Does.Contain("no patient data"));
        Assert.That((bool?)wholeBrain["allow_missing_observation"], Is.True);
    }

    [Test]
    public void GoldenRegressionTool_WritesReportsAndPinpointsInjectedTensor()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var tool = Path.Combine(root, "Tools", "GoldenRegression", "golden_regression.py");
        var manifests = Path.Combine(root, "Tools", "GoldenRegression", "manifests");
        var output = Path.Combine(Path.GetTempPath(), "AIImageGoldenRegressionTests", Guid.NewGuid().ToString("N"));

        var success = RunPython(tool, manifests, output, null);
        Assert.That(success.exitCode, Is.EqualTo(0), success.output);
        var reportPath = Path.Combine(output, "golden-report.json");
        Assert.That(File.Exists(reportPath), Is.True);
        Assert.That(File.Exists(Path.Combine(output, "golden-report.md")), Is.True);
        Assert.That((string)JObject.Parse(File.ReadAllText(reportPath))["schema_version"], Is.EqualTo(ReportSchema));

        var injectedOutput = Path.Combine(Path.GetTempPath(), "AIImageGoldenRegressionTests", Guid.NewGuid().ToString("N"));
        var failure = RunPython(tool, manifests, injectedOutput, "layer.sigmoid.pack4:sigmoid_0/sigmoid_out:2:0.1");
        Assert.That(failure.exitCode, Is.EqualTo(1), failure.output);
        var failureReport = File.ReadAllText(Path.Combine(injectedOutput, "golden-report.json"));
        Assert.That(failureReport, Does.Contain("sigmoid_0"));
        Assert.That(failureReport, Does.Contain("sigmoid_out"));
    }

    [Test]
    public void GoldenDebugReadback_IsEditorOnlyAndRequiresPinnedTexture()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var source = File.ReadAllText(Path.Combine(root, "Assets", "Editor", "NcnnGoldenDebugOracleReadback.cs"));
        Assert.That(source, Does.Contain("#if UNITY_EDITOR"));
        Assert.That(source, Does.Contain("Debug/Oracle test only"));
        Assert.That(source, Does.Contain("TryGetExistingTexture"));
        Assert.That(source, Does.Contain("GetExistingTextureData"));
        Assert.That(source, Does.Not.Contain("ComputeBuffer"));
    }

    private static (int exitCode, string output) RunPython(string tool, string manifests, string output, string injection)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "python",
            Arguments = Quote(tool) + " --manifest " + Quote(manifests) + " --output-dir " + Quote(output)
                + (string.IsNullOrWhiteSpace(injection) ? string.Empty : " --inject-perturbation " + Quote(injection)),
            WorkingDirectory = Path.GetDirectoryName(tool),
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        using var process = Process.Start(startInfo);
        Assert.That(process, Is.Not.Null, "Unable to launch Python golden comparison tool.");
        var stdout = process.StandardOutput.ReadToEnd();
        var stderr = process.StandardError.ReadToEnd();
        process.WaitForExit(30000);
        return (process.ExitCode, stdout + Environment.NewLine + stderr);
    }

    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }
}
#endif
