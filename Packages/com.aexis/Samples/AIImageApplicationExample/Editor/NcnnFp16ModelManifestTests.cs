#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.IO;
using Aexis;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using Aexis.Execution;

public sealed class NcnnFp16ModelManifestTests
{
    [Test]
    public void ManifestParser_AcceptsFp32AndFp16VariantsForTheSameModel()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var fp32 = AexisModelManifestLoader.LoadFromFile(AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "clip-mobileclip-s0.fp32.model.json"));
        var fp16 = AexisModelManifestLoader.LoadFromFile(AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "clip-mobileclip-s0.fp16.model.json"));

        Assert.That(fp32.modelId, Is.EqualTo(fp16.modelId));
        Assert.That(fp32.precision.activationDataType, Is.EqualTo(TensorDataType.Float32));
        Assert.That(fp16.precision.activationDataType, Is.EqualTo(TensorDataType.Float16));
        Assert.That(fp16.precision.weightDataType, Is.EqualTo(TensorDataType.Float16));
        Assert.That(fp16.precision.sensitiveOutputDataType, Is.EqualTo(TensorDataType.Float32));

        var mattingMixed = AexisModelManifestLoader.LoadFromFile(AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "matting.fp16-weights.model.json"));
        Assert.That(mattingMixed.modelId, Is.EqualTo("matting.ncnn"));
        Assert.That(mattingMixed.precision.activationDataType, Is.EqualTo(TensorDataType.Float32));
        Assert.That(mattingMixed.precision.weightDataType, Is.EqualTo(TensorDataType.Float16));
    }

    [Test]
    public void ShippingDefaults_KeepMattingOptInAndSelectFp16ForValidatedRunners()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var clip = AexisModelManifestLoader.LoadFromFile(AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "clip-mobileclip-s0.fp16.model.json"));
        var esrgan = AexisModelManifestLoader.LoadFromFile(AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "esrgan-realesrgan-x4plus.fp16.model.json"));
        var matting = AexisModelManifestLoader.LoadFromFile(AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "matting.fp32.model.json"));

        Assert.That(clip.precision.activationDataType, Is.EqualTo(TensorDataType.Float16));
        Assert.That(esrgan.precision.activationDataType, Is.EqualTo(TensorDataType.Float16));
        Assert.That(matting.precision.activationDataType, Is.EqualTo(TensorDataType.Float32));
    }

    [Test]
    public void PrecisionMode_AutoOnlySelectsFp16ForValidatedModels()
    {
        Assert.That(AexisModelManifestLoader.ResolveAutoPrecision("mobileclip_s0_export"), Is.EqualTo(AexisPrecisionMode.FP16));
        Assert.That(AexisModelManifestLoader.ResolveAutoPrecision("realesrgan-x4plus"), Is.EqualTo(AexisPrecisionMode.FP16));
        Assert.That(AexisModelManifestLoader.ResolveAutoPrecision("matting.ncnn"), Is.EqualTo(AexisPrecisionMode.FP32));
        Assert.That(AexisModelManifestLoader.ResolveAutoPrecision("stable-diffusion"), Is.EqualTo(AexisPrecisionMode.FP32));
    }

    [Test]
    public void ExplicitPrecision_OverridesTheProcessWideManifest()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var configuredFp16 = AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "clip-mobileclip-s0.fp16.model.json");
        var previous = Environment.GetEnvironmentVariable(AexisModelManifestLoader.ManifestEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(AexisModelManifestLoader.ManifestEnvironmentVariable, configuredFp16);
            var manifest = AexisModelManifestLoader.ResolveRunnerManifest("mobileclip_s0_export", AexisPrecisionMode.FP32);

            Assert.That(manifest.precision.activationDataType, Is.EqualTo(TensorDataType.Float32));
            Assert.That(manifest.precision.weightDataType, Is.EqualTo(TensorDataType.Float32));
        }
        finally
        {
            Environment.SetEnvironmentVariable(AexisModelManifestLoader.ManifestEnvironmentVariable, previous);
        }
    }

    [Test]
    public void ExplicitFp32_UsesFloat4StorageWithoutAPackagedManifest()
    {
        using var ops = new AexisOps();
        using var session = NcnnInferenceSessionFactory.Create(ops, "gfpgan", AexisPrecisionMode.FP32);

        Assert.That(session.AppliedPrecisionMode, Is.EqualTo(AexisPrecisionMode.FP32));
        Assert.That(session.TensorTextureFormat, Is.EqualTo(RenderTextureFormat.ARGBFloat));
        Assert.That(session.ResolveActivationTextureFormat(4), Is.EqualTo(RenderTextureFormat.ARGBFloat));
    }

    [Test]
    public void ExplicitFp16_UsesVerifiedFaceRestorationManifestsWithoutFallingBack()
    {
        using var ops = new AexisOps();
        using var gfpgan = NcnnInferenceSessionFactory.Create(ops, "gfpgan", AexisPrecisionMode.FP16);
        using var codeformer = NcnnInferenceSessionFactory.Create(ops, "codeformer", AexisPrecisionMode.FP16);

        Assert.That(gfpgan.AppliedPrecisionMode, Is.EqualTo(AexisPrecisionMode.FP16));
        Assert.That(codeformer.AppliedPrecisionMode, Is.EqualTo(AexisPrecisionMode.FP16));
        Assert.That(gfpgan.TensorTextureFormat, Is.EqualTo(RenderTextureFormat.ARGBHalf));
        Assert.That(codeformer.TensorTextureFormat, Is.EqualTo(RenderTextureFormat.ARGBHalf));
    }

    [Test]
    public void ManifestAppliedToSession_SelectsHalf4AndFp32TargetWithoutChangingTheModelId()
    {
        var manifest = AexisModelManifestLoader.LoadFromJson(
            "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"fixture\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"FP16\",\"sensitiveOutputDtype\":\"FP32\",\"requireStrictTexturePlan\":true}}");
        using var session = NcnnInferenceSessionFactory.Create(new AexisOps(), manifest);

        Assert.That(session.TensorTextureFormat, Is.EqualTo(RenderTextureFormat.ARGBHalf));
        session.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
        Assert.That(session.TensorTextureFormat, Is.EqualTo(RenderTextureFormat.ARGBHalf));
        Assert.That(session.StrictTextureTargetDtype, Is.EqualTo("FP16"));
        Assert.That(session.ResolveActivationTextureFormat(4), Is.EqualTo(RenderTextureFormat.ARGBHalf));
        Assert.That(session.ResolveSensitiveOutputTextureFormat(), Is.EqualTo(RenderTextureFormat.ARGBFloat));
    }

    [Test]
    public void SensitiveOperators_DeclareFp32AccumulationWhileManifestControlsStorageCast()
    {
        Assert.That(AexisGraphSession.RequiresFp32AccumulatorOutput("LayerNorm"), Is.True);
        Assert.That(AexisGraphSession.RequiresFp32AccumulatorOutput("Softmax"), Is.True);
        Assert.That(AexisGraphSession.RequiresFp32AccumulatorOutput("Reduction"), Is.True);
        Assert.That(AexisGraphSession.RequiresFp32AccumulatorOutput("Convolution"), Is.False);

        var root = Path.GetDirectoryName(Application.dataPath);
        var source = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Resources", "Aexis", "Includes", "KernelGroups", "AexisKernels.Pack4PointwiseNorm.hlsl"));
        Assert.That(source, Does.Contain("float sumSquare = 0.0"));

        var reproSource = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Execution", "Graph", "AexisGraphSession.cs"));
        Assert.That(reproSource, Does.Contain("format = ResolveLinearTextureFormat(format);"));
        Assert.That(reproSource, Does.Not.Contain("format = format == RenderTextureFormat.ARGBHalf ? ResolveLinearMatTextureFormat() : format;"));
        Assert.That(reproSource, Does.Contain("Fp32ActivationStartLayerName"));
        Assert.That(reproSource, Does.Contain("UsesFp32ActivationIsland"));

        var codeFormerRunner = File.ReadAllText(Path.Combine(root, "Assets", "Scripts", "CodeFormerNcnnReproRunner2.cs"));
        Assert.That(codeFormerRunner, Does.Contain("Fp32ActivationStartLayerName = _generatorRepro.UsesFp16ActivationStorage"));
        Assert.That(codeFormerRunner, Does.Contain("\"Resize_512\""));

        var reshapeKernel = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Resources", "Aexis", "AexisCommon.compute"));
        Assert.That(reshapeKernel, Does.Contain("linearIndex >> 2"));
        Assert.That(reshapeKernel, Does.Contain("linearIndex & 3"));
    }

    [Test]
    public void Fp16WeightKernels_UsePackedHalfUploadsAndNoActivationBufferFallback()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var conv = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Resources", "Aexis", "Includes", "KernelGroups", "AexisKernels.Pack4Conv.hlsl"));
        var gemm = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Resources", "Aexis", "Includes", "KernelGroups", "AexisKernels.Pack4Matmul.hlsl"));
        var convolutionLayer = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Execution", "Layers", "AexisConvolutionLayer.cs"));

        Assert.That(conv, Does.Contain("_ConvW4Fp16"));
        Assert.That(conv, Does.Contain("_DwConvW4Fp16"));
        Assert.That(conv, Does.Contain("AexisMaskConvOutputTail"));
        Assert.That(conv, Does.Contain("f16tof32"));
        Assert.That(gemm, Does.Contain("_MatBFp16"));
        Assert.That(gemm, Does.Contain("f16tof32"));
        Assert.That(convolutionLayer, Does.Contain("SetFp16ConvWeights"));

        var innerProductLayer = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Execution", "Layers", "AexisInnerProductLayer.cs"));
        Assert.That(innerProductLayer, Does.Contain("wFp16"));
        Assert.That(innerProductLayer, Does.Contain("SetFp16GemmWeights"));

        var multiHeadAttentionLayer = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Execution", "Layers", "AexisMultiHeadAttentionLayer.cs"));
        Assert.That(multiHeadAttentionLayer, Does.Contain("owner.Ops.SetFp16GemmWeights(null);"));
    }

    [Test]
    public void ManifestParser_AcceptsSelectiveW8A8AndRejectsMissingCalibrationScale()
    {
        var valid = AexisModelManifestLoader.LoadFromJson(
            "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"fixture\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"INT8\",\"sensitiveOutputDtype\":\"FP32\"},\"quantization\":{\"quantizationVersion\":\"v1\",\"calibrationVersion\":\"fixture-v1\",\"calibrationMethod\":\"absmax\",\"weightScheme\":\"INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC\",\"accumulationDtype\":\"FP32\",\"activationQuantized\":true,\"nodePlans\":[{\"layerName\":\"conv_0\",\"operatorName\":\"Convolution\",\"mode\":\"W8A8\",\"activationScale\":0.03125}]}}");
        Assert.That(valid.TryGetQuantizedNodePlan("conv_0", "Convolution", out var plan), Is.True);
        Assert.That(plan.mode, Is.EqualTo(QuantizedNodeMode.Int8W8A8));

        Assert.Throws<InferenceContractException>(() => AexisModelManifestLoader.LoadFromJson(
            "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"fixture\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"INT8\",\"sensitiveOutputDtype\":\"FP32\"},\"quantization\":{\"quantizationVersion\":\"v1\",\"calibrationVersion\":\"fixture-v1\",\"calibrationMethod\":\"absmax\",\"weightScheme\":\"INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC\",\"accumulationDtype\":\"FP32\",\"activationQuantized\":true,\"nodePlans\":[{\"layerName\":\"conv_0\",\"operatorName\":\"Convolution\",\"mode\":\"W8A8\",\"activationScale\":0}]}}"));
    }
}
#endif
