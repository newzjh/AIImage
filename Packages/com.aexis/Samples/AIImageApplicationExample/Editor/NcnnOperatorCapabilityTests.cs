#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.Linq;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using Aexis.Execution;

public sealed class NcnnOperatorCapabilityTests
{
    [Test]
    public void CapabilityDocument_HasStableSchemaAndDoesNotPromotePlaceholderOperators()
    {
        var document = AexisOperatorCapabilities.CreateDocument();
        var json = AexisOperatorCapabilities.ToStableJson(document);
        var reparsed = JsonUtility.FromJson<AexisOperatorCapabilityDocument>(json);

        Assert.That(AexisOperatorCapabilities.ToStableJson(AexisOperatorCapabilities.CreateDocument()), Is.EqualTo(json));
        Assert.That(reparsed.schemaVersion, Is.EqualTo(AexisOperatorCapabilities.SchemaVersion));
        Assert.That(reparsed.contract, Is.EqualTo(AexisOperatorCapabilities.Contract));
        Assert.That(reparsed.operators, Is.Not.Empty);
        Assert.That(
            reparsed.operators.Select(entry => entry.operatorName),
            Is.EqualTo(reparsed.operators.Select(entry => entry.operatorName).OrderBy(name => name, System.StringComparer.Ordinal)));

        var gemm = reparsed.operators.Single(entry => entry.operatorName == "Gemm");
        var layerNorm = reparsed.operators.Single(entry => entry.operatorName == "LayerNorm");
        Assert.That(gemm.status, Is.EqualTo(AexisOperatorCapabilityStatus.Partial));
        Assert.That(layerNorm.status, Is.EqualTo(AexisOperatorCapabilityStatus.Partial));
        Assert.That(AexisOperatorCapabilities.IsStrictlySupported(gemm, AexisOperatorCapabilityBackend.CommandBuffer, "FP32"), Is.False);
        Assert.That(AexisOperatorCapabilities.IsStrictlySupported(layerNorm, AexisOperatorCapabilityBackend.CommandBuffer, "FP32"), Is.False);
    }

    [Test]
    public void StrictPreflight_RejectsPartialNodeAndReportsDeclaredTensorContractWithoutExecution()
    {
        var model = NcnnParamParser.Parse(
            "7767517\n"
            + "2 2\n"
            + "Input in0 0 1 in0\n"
            + "Gemm gemm_0 1 1 in0 out 4=1 5=1 7=1 8=1 9=1\n");

        var report = AexisModelPreflight.Analyze(model, new AexisModelPreflightRequest
        {
            modelName = "partial-gemm",
            targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
            targetDtype = "FP32",
            strict = true,
            inputs = new[]
            {
                new AexisPreflightTensorDescriptor
                {
                    blob = "in0",
                    logicalShape = new[] { 2, 4, 3, 1, 1 },
                    storageShape = new[] { 2, 4, 3, 1, 1 },
                    layout = "Linear",
                    dtype = "FP32"
                }
            }
        });

        Assert.That(report.strictEligible, Is.False);
        Assert.That(report.missingDependencies, Is.Empty);
        Assert.That(report.declaredInputs.Single().logicalShape, Is.EqualTo(new[] { 2, 4, 3, 1, 1 }));
        Assert.That(report.nodes.Single(node => node.operatorName == "Gemm").strictEligible, Is.False);
        Assert.That(report.nodes.Single(node => node.operatorName == "Gemm").recommendedAction, Does.Contain("strict plans"));
    }

    [Test]
    public void Preflight_ReportsMissingRequiredNcnnParameters()
    {
        var model = NcnnParamParser.Parse(
            "7767517\n"
            + "2 2\n"
            + "Input in0 0 1 in0\n"
            + "Convolution conv_missing 1 1 in0 out\n");

        var report = AexisModelPreflight.Analyze(model, new AexisModelPreflightRequest
        {
            inputs = new[]
            {
                new AexisPreflightTensorDescriptor
                {
                    blob = "in0",
                    logicalShape = new[] { 4, 8, 8, 1, 3 },
                    storageShape = new[] { 4, 8, 8, 1, 3 },
                    layout = "NCHW",
                    dtype = "FP32"
                }
            }
        });

        Assert.That(report.missingParameters.Length, Is.EqualTo(3));
        Assert.That(report.missingParameters.Select(issue => issue.message), Has.Some.Contains("num_output"));
        Assert.That(report.strictEligible, Is.False);
    }

    [Test]
    public void Preflight_RequiresDescriptorForEveryModelInput()
    {
        var model = NcnnParamParser.Parse(
            "7767517\n"
            + "1 1\n"
            + "Input model_input 0 1 input_blob\n");

        var report = AexisModelPreflight.Analyze(model, new AexisModelPreflightRequest());

        Assert.That(report.missingDependencies.Select(issue => issue.code), Has.Some.EqualTo("missing-input-descriptor"));
        Assert.That(report.missingDependencies.Single(issue => issue.code == "missing-input-descriptor").blob, Is.EqualTo("input_blob"));
    }
}
#endif
