using System;
using System.IO;
using System.Linq;
using NcnnCompute;
using NUnit.Framework;
using UnityEngine;

public sealed class NcnnStrictTextureExecutionPlanTests
{
    [Test]
    public void StrictPlan_RejectsPartialLayerNormWithCompleteDiagnostics()
    {
        var error = Assert.Throws<StrictTextureInferencePlanException>(() => NcnnTextureExecutionPlanner.Compile(
            Parse("Input in0 0 1 in\nLayerNorm norm_0 1 1 in out\n"),
            CreateRequest()));

        Assert.That(error.Plan.strictEligible, Is.False);
        Assert.That(error.Plan.dispatchAllowed, Is.False);
        var diagnostic = error.Diagnostics.Single(item => item.operatorName == "LayerNorm");
        Assert.That(diagnostic.layerIndex, Is.EqualTo(1));
        Assert.That(diagnostic.layer, Is.EqualTo("norm_0"));
        Assert.That(diagnostic.canonicalOperator, Is.EqualTo("LayerNorm"));
        Assert.That(diagnostic.capabilityStatus, Is.EqualTo(NcnnOperatorCapabilityStatus.Partial));
        Assert.That(diagnostic.targetBackend, Is.EqualTo(NcnnOperatorCapabilityBackend.CommandBuffer));
        Assert.That(diagnostic.targetDtype, Is.EqualTo("FP16"));
        Assert.That(diagnostic.targetLayout, Is.EqualTo(NcnnTexturePlanLayout.Packed4));
        Assert.That(diagnostic.inputs.Single().logicalShape, Is.EqualTo(new[] { 3, 4, 4, 1, 4 }));
        Assert.That(diagnostic.rejectedPaths, Is.EquivalentTo(new[] { "alias-only", "placeholder", "materialize-from-buffer", "legacy-path" }));
        Assert.That(diagnostic.recommendedAction, Does.Contain("Implement and verify a real CommandBuffer Pack4 path"));
        Assert.That(error.Message, Does.Contain("target_dtype=FP16"));
        Assert.That(error.Message, Does.Contain("target_layout=Packed4"));
    }

    [Test]
    public void StrictPlan_AcceptsDescriptorProvenReshapeAndSplit()
    {
        var plan = NcnnTextureExecutionPlanner.Compile(
            Parse("Input in0 0 1 in\nReshape reshape_0 1 1 in reshaped\nSplit split_0 1 2 reshaped left right\n"),
            CreateRequest());

        Assert.That(plan.strictEligible, Is.True);
        Assert.That(plan.dispatchAllowed, Is.True);
        Assert.That(plan.diagnostics, Is.Empty);
        Assert.That(plan.nodes.Single(node => node.operatorName == "Reshape").usesDescriptorAlias, Is.True);
        var split = plan.nodes.Single(node => node.operatorName == "Split");
        Assert.That(split.usesDescriptorAlias, Is.True);
        Assert.That(split.outputs.Select(output => output.aliasGroup).Distinct(), Is.EqualTo(new[] { "input:in" }));
    }

    [Test]
    public void StrictPlan_AcceptsVerifiedCommandBufferPack4Graph()
    {
        var plan = NcnnTextureExecutionPlanner.Compile(
            Parse("Input in0 0 1 in\nReLU relu_0 1 1 in out\n"),
            CreateRequest());

        var relu = plan.nodes.Single(node => node.operatorName == "ReLU");
        Assert.That(relu.accepted, Is.True);
        Assert.That(relu.executionPath, Is.EqualTo("command-buffer-pack4"));
        Assert.That(relu.usesDescriptorAlias, Is.False);
        Assert.That(NcnnOperatorCapabilities.IsStrictlySupported(
            NcnnOperatorCapabilities.CreateDocument().operators.Single(capability => capability.operatorName == "ReLU"),
            NcnnOperatorCapabilityBackend.CommandBuffer,
            "FP16",
            NcnnTexturePlanLayout.Packed4), Is.True);
    }

    [Test]
    public void StrictPlan_RequiresRuntimeProofForPartialCommandBufferNode()
    {
        var model = Parse("Input in0 0 1 in\nConvolution conv_0 1 1 in out 0=4 1=1 6=16\n");
        var error = Assert.Throws<StrictTextureInferencePlanException>(() => NcnnTextureExecutionPlanner.Compile(model, CreateRequest()));
        var rejected = error.Diagnostics.Single(diagnostic => diagnostic.operatorName == "Convolution");
        Assert.That(rejected.code, Is.EqualTo("command-buffer-pack4-profile-rejected"));
        Assert.That(rejected.reason, Does.Contain("No loaded-runtime CommandBuffer Pack4 verifier"));

        var plan = NcnnTextureExecutionPlanner.Compile(model, CreateRequest(nodeVerifier: CreateVerifiedPartialNode));
        var convolution = plan.nodes.Single(node => node.operatorName == "Convolution");
        Assert.That(convolution.capabilityStatus, Is.EqualTo(NcnnOperatorCapabilityStatus.Partial));
        Assert.That(convolution.executionPath, Is.EqualTo("command-buffer-pack4:test-profile"));
        Assert.That(convolution.accepted, Is.True);
    }

    [Test]
    public void StrictPlan_RejectsPartialProfileProofThatNamesPlaceholder()
    {
        var model = Parse("Input in0 0 1 in\nConvolution conv_0 1 1 in out 0=4 1=1 6=16\n");
        var error = Assert.Throws<StrictTextureInferencePlanException>(() => NcnnTextureExecutionPlanner.Compile(
            model,
            CreateRequest(nodeVerifier: (layer, inputs, request) => new NcnnTextureExecutionPlanNodeVerification
            {
                accepted = true,
                executionPath = "command-buffer-pack4:placeholder",
                outputs = new[] { CreateOutputDescriptor("out", request) }
            })));

        var rejected = error.Diagnostics.Single(diagnostic => diagnostic.operatorName == "Convolution");
        Assert.That(rejected.code, Is.EqualTo("command-buffer-pack4-profile-rejected"));
        Assert.That(rejected.reason, Does.Contain("did not provide a real CommandBuffer Pack4 execution path"));
    }

    [Test]
    public void StrictPlan_AcceptsProfileProvenNoopAliasOnlyWithDescriptorEvidence()
    {
        var model = Parse("Input in0 0 1 in\nInterp resize_noop 1 1 in out 0=1 1=1 2=1\n");
        var plan = NcnnTextureExecutionPlanner.Compile(
            model,
            CreateRequest(nodeVerifier: (layer, inputs, request) => new NcnnTextureExecutionPlanNodeVerification
            {
                accepted = true,
                usesDescriptorAlias = true,
                executionPath = "descriptor-alias",
                outputs = new[]
                {
                    new NcnnTexturePlanTensorDescriptor
                    {
                        blob = "out",
                        logicalShape = (int[])inputs[0].logicalShape.Clone(),
                        storageShape = (int[])inputs[0].storageShape.Clone(),
                        layout = request.targetLayout,
                        dtype = request.targetDtype,
                        aliasGroup = inputs[0].aliasGroup,
                        textureBacked = true
                    }
                }
            }));

        var resize = plan.nodes.Single(node => node.operatorName == "Interp");
        Assert.That(resize.accepted, Is.True);
        Assert.That(resize.usesDescriptorAlias, Is.True);
        Assert.That(resize.executionPath, Is.EqualTo("descriptor-alias"));
    }

    [Test]
    public void DebugOracle_ExplicitlyRelaxesButDoesNotPromoteCapability()
    {
        var plan = NcnnTextureExecutionPlanner.Compile(
            Parse("Input in0 0 1 in\nLayerNorm norm_0 1 1 in out\n"),
            CreateRequest(debugOracleRelaxed: true));

        Assert.That(plan.strictEligible, Is.False);
        Assert.That(plan.dispatchAllowed, Is.True);
        Assert.That(plan.nodes.Single(node => node.operatorName == "LayerNorm").acceptedByDebugOracle, Is.True);
        Assert.That(plan.diagnostics.Single(diagnostic => diagnostic.operatorName == "LayerNorm").blocking, Is.False);
        Assert.That(NcnnOperatorCapabilities.IsStrictlySupported(
            NcnnOperatorCapabilities.CreateDocument().operators.Single(capability => capability.operatorName == "LayerNorm"),
            NcnnOperatorCapabilityBackend.CommandBuffer,
            "FP16",
            NcnnTexturePlanLayout.Packed4), Is.False);
    }

    [Test]
    public void ProductionRunners_KeepCommandBufferStrictPlanningAndDoNotEnableDebugOracle()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var repro = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.unitygpu", "Runtime", "NcnnCompute", "NcnnRepro.cs"));
        var factory = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.unitygpu", "Runtime", "NcnnLayers", "NcnnLayerFactoryRepro.cs"));
        var overrides = Directory.EnumerateFiles(Path.Combine(root, "Assets", "Scripts"), "*.cs", SearchOption.AllDirectories)
            .Where(path => File.ReadAllText(path).Contains("ExecutionMode = NcnnInferenceExecutionMode.DebugOracle"))
            .ToArray();

        Assert.That(repro, Does.Contain("public NcnnInferenceExecutionMode ExecutionMode { get; set; } = NcnnInferenceExecutionMode.ProductionTextureOnly;"));
        Assert.That(repro, Does.Contain("public bool StrictTextureInference => !IsExplicitDebugOracleExecution;"));
        Assert.That(repro, Does.Contain("&& ExecutionMode == NcnnInferenceExecutionMode.DebugOracle;"));
        Assert.That(repro, Does.Contain("CompleteTextureExecutionPlan(inputs, IsExplicitDebugOracleExecution);"));
        Assert.That(factory, Does.Contain("EnsureCommandBufferTextureExecutionPlan(textureInputs, textureInputShapes);"));
        Assert.That(overrides, Is.Empty);
    }

    // Unity batch entry point for environments where -runTests is unavailable.
    public static void RunBatchValidation()
    {
        var tests = new NcnnStrictTextureExecutionPlanTests();
        tests.StrictPlan_RejectsPartialLayerNormWithCompleteDiagnostics();
        tests.StrictPlan_AcceptsDescriptorProvenReshapeAndSplit();
        tests.StrictPlan_AcceptsVerifiedCommandBufferPack4Graph();
        tests.StrictPlan_RequiresRuntimeProofForPartialCommandBufferNode();
        tests.StrictPlan_RejectsPartialProfileProofThatNamesPlaceholder();
        tests.StrictPlan_AcceptsProfileProvenNoopAliasOnlyWithDescriptorEvidence();
        tests.DebugOracle_ExplicitlyRelaxesButDoesNotPromoteCapability();
        tests.ProductionRunners_KeepCommandBufferStrictPlanningAndDoNotEnableDebugOracle();
        UnityEngine.Debug.Log("[NcnnStrictTextureExecutionPlanTests] passed");
    }

    private static NcnnTextureExecutionPlanRequest CreateRequest(
        bool debugOracleRelaxed = false,
        NcnnTextureExecutionPlanNodeVerifier nodeVerifier = null)
    {
        return new NcnnTextureExecutionPlanRequest
        {
            modelName = "strict-texture-test",
            targetBackend = NcnnOperatorCapabilityBackend.CommandBuffer,
            targetDtype = "FP16",
            targetLayout = NcnnTexturePlanLayout.Packed4,
            strict = !debugOracleRelaxed,
            debugOracleRelaxed = debugOracleRelaxed,
            nodeVerifier = nodeVerifier,
            inputs = new[]
            {
                new NcnnTexturePlanTensorDescriptor
                {
                    blob = "in",
                    logicalShape = new[] { 3, 4, 4, 1, 4 },
                    storageShape = new[] { 3, 4, 4, 1, 4 },
                    layout = NcnnTexturePlanLayout.Packed4,
                    dtype = "FP16",
                    aliasGroup = "input:in",
                    textureBacked = true
                }
            }
        };
    }

    private static NcnnTextureExecutionPlanNodeVerification CreateVerifiedPartialNode(
        NcnnParamModel.Layer layer,
        System.Collections.Generic.IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
        NcnnTextureExecutionPlanRequest request)
    {
        return new NcnnTextureExecutionPlanNodeVerification
        {
            accepted = true,
            executionPath = "command-buffer-pack4:test-profile",
            outputs = new[] { CreateOutputDescriptor(layer.topNames.Single(), request) }
        };
    }

    private static NcnnTexturePlanTensorDescriptor CreateOutputDescriptor(string blob, NcnnTextureExecutionPlanRequest request)
    {
        return new NcnnTexturePlanTensorDescriptor
        {
            blob = blob,
            logicalShape = new[] { 3, 4, 4, 1, 4 },
            storageShape = new[] { 3, 4, 4, 1, 4 },
            layout = request.targetLayout,
            dtype = request.targetDtype,
            aliasGroup = "computed:test",
            textureBacked = true
        };
    }

    private static NcnnParamModel Parse(string layers)
    {
        var layerCount = layers.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries).Length;
        var blobCount = layers.IndexOf("Split", StringComparison.Ordinal) >= 0 ? 4 : 2;
        return NcnnParamParser.Parse("7767517\n" + layerCount + " " + blobCount + "\n" + layers);
    }
}
