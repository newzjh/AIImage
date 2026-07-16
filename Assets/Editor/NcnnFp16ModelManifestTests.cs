#if UNITY_EDITOR
using System;
using System.IO;
using AIImage.Inference.Core;
using NcnnCompute;
using NUnit.Framework;
using UnityEngine;

public sealed class NcnnFp16ModelManifestTests
{
    [Test]
    public void ManifestParser_AcceptsFp32AndFp16VariantsForTheSameModel()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var fp32 = NcnnModelManifestLoader.LoadFromFile(Path.Combine(root, "Assets", "StreamingAssets", "InferenceManifests", "clip-mobileclip-s0.fp32.model.json"));
        var fp16 = NcnnModelManifestLoader.LoadFromFile(Path.Combine(root, "Assets", "StreamingAssets", "InferenceManifests", "clip-mobileclip-s0.fp16.model.json"));

        Assert.That(fp32.modelId, Is.EqualTo(fp16.modelId));
        Assert.That(fp32.precision.activationDataType, Is.EqualTo(TensorDataType.Float32));
        Assert.That(fp16.precision.activationDataType, Is.EqualTo(TensorDataType.Float16));
        Assert.That(fp16.precision.weightDataType, Is.EqualTo(TensorDataType.Float16));
        Assert.That(fp16.precision.sensitiveOutputDataType, Is.EqualTo(TensorDataType.Float32));

        var mattingMixed = NcnnModelManifestLoader.LoadFromFile(Path.Combine(root, "Assets", "StreamingAssets", "InferenceManifests", "matting.fp16-weights.model.json"));
        Assert.That(mattingMixed.modelId, Is.EqualTo("matting.ncnn"));
        Assert.That(mattingMixed.precision.activationDataType, Is.EqualTo(TensorDataType.Float32));
        Assert.That(mattingMixed.precision.weightDataType, Is.EqualTo(TensorDataType.Float16));
    }

    [Test]
    public void ShippingDefaults_KeepMattingOptInAndSelectFp16ForValidatedRunners()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var clip = NcnnModelManifestLoader.LoadFromFile(Path.Combine(root, "Assets", "StreamingAssets", "InferenceManifests", "clip-mobileclip-s0.fp16.model.json"));
        var esrgan = NcnnModelManifestLoader.LoadFromFile(Path.Combine(root, "Assets", "StreamingAssets", "InferenceManifests", "esrgan-realesrgan-x4plus.fp16.model.json"));
        var matting = NcnnModelManifestLoader.LoadFromFile(Path.Combine(root, "Assets", "StreamingAssets", "InferenceManifests", "matting.fp32.model.json"));

        Assert.That(clip.precision.activationDataType, Is.EqualTo(TensorDataType.Float16));
        Assert.That(esrgan.precision.activationDataType, Is.EqualTo(TensorDataType.Float16));
        Assert.That(matting.precision.activationDataType, Is.EqualTo(TensorDataType.Float32));
    }

    [Test]
    public void PrecisionMode_AutoOnlySelectsFp16ForValidatedModels()
    {
        Assert.That(NcnnModelManifestLoader.ResolveAutoPrecision("mobileclip_s0_export"), Is.EqualTo(NcnnPrecisionMode.FP16));
        Assert.That(NcnnModelManifestLoader.ResolveAutoPrecision("realesrgan-x4plus"), Is.EqualTo(NcnnPrecisionMode.FP16));
        Assert.That(NcnnModelManifestLoader.ResolveAutoPrecision("matting.ncnn"), Is.EqualTo(NcnnPrecisionMode.FP32));
        Assert.That(NcnnModelManifestLoader.ResolveAutoPrecision("stable-diffusion"), Is.EqualTo(NcnnPrecisionMode.FP32));
    }

    [Test]
    public void ExplicitPrecision_OverridesTheProcessWideManifest()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var configuredFp16 = Path.Combine(root, "Assets", "StreamingAssets", "InferenceManifests", "clip-mobileclip-s0.fp16.model.json");
        var previous = Environment.GetEnvironmentVariable(NcnnModelManifestLoader.ManifestEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(NcnnModelManifestLoader.ManifestEnvironmentVariable, configuredFp16);
            var manifest = NcnnModelManifestLoader.ResolveRunnerManifest("mobileclip_s0_export", NcnnPrecisionMode.FP32);

            Assert.That(manifest.precision.activationDataType, Is.EqualTo(TensorDataType.Float32));
            Assert.That(manifest.precision.weightDataType, Is.EqualTo(TensorDataType.Float32));
        }
        finally
        {
            Environment.SetEnvironmentVariable(NcnnModelManifestLoader.ManifestEnvironmentVariable, previous);
        }
    }

    [Test]
    public void ExplicitFp32_UsesFloat4StorageWithoutAPackagedManifest()
    {
        using var ops = new NcnnOps();
        using var session = NcnnInferenceSessionFactory.Create(ops, "gfpgan", NcnnPrecisionMode.FP32);

        Assert.That(session.AppliedPrecisionMode, Is.EqualTo(NcnnPrecisionMode.FP32));
        Assert.That(session.TensorTextureFormat, Is.EqualTo(RenderTextureFormat.ARGBFloat));
        Assert.That(session.ResolveActivationTextureFormat(4), Is.EqualTo(RenderTextureFormat.ARGBFloat));
    }

    [Test]
    public void ExplicitFp16_UsesVerifiedFaceRestorationManifestsWithoutFallingBack()
    {
        using var ops = new NcnnOps();
        using var gfpgan = NcnnInferenceSessionFactory.Create(ops, "gfpgan", NcnnPrecisionMode.FP16);
        using var codeformer = NcnnInferenceSessionFactory.Create(ops, "codeformer", NcnnPrecisionMode.FP16);

        Assert.That(gfpgan.AppliedPrecisionMode, Is.EqualTo(NcnnPrecisionMode.FP16));
        Assert.That(codeformer.AppliedPrecisionMode, Is.EqualTo(NcnnPrecisionMode.FP16));
        Assert.That(gfpgan.TensorTextureFormat, Is.EqualTo(RenderTextureFormat.ARGBHalf));
        Assert.That(codeformer.TensorTextureFormat, Is.EqualTo(RenderTextureFormat.ARGBHalf));
    }

    [Test]
    public void ManifestAppliedToSession_SelectsHalf4AndFp32TargetWithoutChangingTheModelId()
    {
        var manifest = NcnnModelManifestLoader.LoadFromJson(
            "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"fixture\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"FP16\",\"sensitiveOutputDtype\":\"FP32\",\"requireStrictTexturePlan\":true}}");
        using var session = NcnnInferenceSessionFactory.Create(new NcnnOps(), manifest);

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
        Assert.That(NcnnRepro.RequiresFp32AccumulatorOutput("LayerNorm"), Is.True);
        Assert.That(NcnnRepro.RequiresFp32AccumulatorOutput("Softmax"), Is.True);
        Assert.That(NcnnRepro.RequiresFp32AccumulatorOutput("Reduction"), Is.True);
        Assert.That(NcnnRepro.RequiresFp32AccumulatorOutput("Convolution"), Is.False);

        var root = Path.GetDirectoryName(Application.dataPath);
        var source = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.kernels", "Runtime", "Resources", "NcnnComputeIncludes", "KernelGroups", "NcnnKernels.Pack4PointwiseNorm.hlsl"));
        Assert.That(source, Does.Contain("float sumSquare = 0.0"));

        var reproSource = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.unitygpu", "Runtime", "NcnnCompute", "NcnnRepro.cs"));
        Assert.That(reproSource, Does.Contain("format = ResolveLinearTextureFormat(format);"));
        Assert.That(reproSource, Does.Not.Contain("format = format == RenderTextureFormat.ARGBHalf ? ResolveLinearMatTextureFormat() : format;"));
        Assert.That(reproSource, Does.Contain("Fp32ActivationStartLayerName"));
        Assert.That(reproSource, Does.Contain("UsesFp32ActivationIsland"));

        var codeFormerRunner = File.ReadAllText(Path.Combine(root, "Assets", "Scripts", "CodeFormerNcnnReproRunner2.cs"));
        Assert.That(codeFormerRunner, Does.Contain("Fp32ActivationStartLayerName = _generatorRepro.UsesFp16ActivationStorage"));
        Assert.That(codeFormerRunner, Does.Contain("\"Resize_512\""));

        var reshapeKernel = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.kernels", "Runtime", "Resources", "NcnnCompute.compute"));
        Assert.That(reshapeKernel, Does.Contain("linearIndex >> 2"));
        Assert.That(reshapeKernel, Does.Contain("linearIndex & 3"));
    }

    [Test]
    public void Fp16WeightKernels_UsePackedHalfUploadsAndNoActivationBufferFallback()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var conv = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.kernels", "Runtime", "Resources", "NcnnComputeIncludes", "KernelGroups", "NcnnKernels.Pack4Conv.hlsl"));
        var gemm = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.kernels", "Runtime", "Resources", "NcnnComputeIncludes", "KernelGroups", "NcnnKernels.Pack4Matmul.hlsl"));
        var convolutionLayer = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.unitygpu", "Runtime", "NcnnLayers", "NcnnConvolutionLayerRepro.cs"));

        Assert.That(conv, Does.Contain("_ConvW4Fp16"));
        Assert.That(conv, Does.Contain("_DwConvW4Fp16"));
        Assert.That(conv, Does.Contain("NcnnMaskConvOutputTail"));
        Assert.That(conv, Does.Contain("f16tof32"));
        Assert.That(gemm, Does.Contain("_MatBFp16"));
        Assert.That(gemm, Does.Contain("f16tof32"));
        Assert.That(convolutionLayer, Does.Contain("SetFp16ConvWeights"));

        var innerProductLayer = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.unitygpu", "Runtime", "NcnnLayers", "NcnnInnerProductLayerRepro.cs"));
        Assert.That(innerProductLayer, Does.Contain("wFp16"));
        Assert.That(innerProductLayer, Does.Contain("SetFp16GemmWeights"));

        var multiHeadAttentionLayer = File.ReadAllText(Path.Combine(root, "Packages", "com.aiimage.inference.unitygpu", "Runtime", "NcnnLayers", "NcnnMultiHeadAttentionLayerRepro.cs"));
        Assert.That(multiHeadAttentionLayer, Does.Contain("owner.Ops.SetFp16GemmWeights(null);"));
    }

    [Test]
    public void ManifestParser_AcceptsSelectiveW8A8AndRejectsMissingCalibrationScale()
    {
        var valid = NcnnModelManifestLoader.LoadFromJson(
            "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"fixture\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"INT8\",\"sensitiveOutputDtype\":\"FP32\"},\"quantization\":{\"quantizationVersion\":\"v1\",\"calibrationVersion\":\"fixture-v1\",\"calibrationMethod\":\"absmax\",\"weightScheme\":\"INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC\",\"accumulationDtype\":\"FP32\",\"activationQuantized\":true,\"nodePlans\":[{\"layerName\":\"conv_0\",\"operatorName\":\"Convolution\",\"mode\":\"W8A8\",\"activationScale\":0.03125}]}}");
        Assert.That(valid.TryGetQuantizedNodePlan("conv_0", "Convolution", out var plan), Is.True);
        Assert.That(plan.mode, Is.EqualTo(QuantizedNodeMode.Int8W8A8));

        Assert.Throws<InferenceContractException>(() => NcnnModelManifestLoader.LoadFromJson(
            "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"fixture\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"INT8\",\"sensitiveOutputDtype\":\"FP32\"},\"quantization\":{\"quantizationVersion\":\"v1\",\"calibrationVersion\":\"fixture-v1\",\"calibrationMethod\":\"absmax\",\"weightScheme\":\"INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC\",\"accumulationDtype\":\"FP32\",\"activationQuantized\":true,\"nodePlans\":[{\"layerName\":\"conv_0\",\"operatorName\":\"Convolution\",\"mode\":\"W8A8\",\"activationScale\":0}]}}"));
    }
}
#endif
