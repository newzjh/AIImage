#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.Collections.Generic;
using System.IO;
using Aexis;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using Aexis.Execution;

public sealed class NcnnInt8WeightOnlyTests
{
    [Test]
    public void ManifestParser_AcceptsD2SelectiveInt8AndRequiresPlansForW8A8()
    {
        var manifest = AexisModelManifestLoader.LoadFromFile(
            AexisApplicationExamplePaths.ResolveStreamingAssetFilePath(
                "InferenceManifests",
                "clip-mobileclip-s0.int8.model.json"));

        Assert.That(manifest.IsInt8WeightOnly, Is.True);
        Assert.That(manifest.precision.activationDataType, Is.EqualTo(TensorDataType.Float16));
        Assert.That(manifest.precision.weightDataType, Is.EqualTo(TensorDataType.Int8));
        Assert.That(manifest.quantization.weightScheme, Is.EqualTo(WeightQuantizationScheme.Int8WeightOnlyPerOutputChannelSymmetric));
        Assert.That(manifest.quantization.symmetric, Is.True);
        Assert.That(manifest.quantization.zeroPoint, Is.Zero);
        Assert.That(manifest.quantization.accumulationDataType, Is.EqualTo(TensorDataType.Float32));
        Assert.That(manifest.quantization.activationQuantized, Is.True);
        Assert.That(manifest.quantization.unquantizedWeightDataType, Is.EqualTo(TensorDataType.Float32));
        Assert.That(manifest.TryGetQuantizedNodePlan("gemm_0", "Gemm", out var manifestGemm), Is.True);
        Assert.That(manifestGemm.mode, Is.EqualTo(QuantizedNodeMode.Int8W8A8));
        Assert.That(manifest.TryGetQuantizedNodePlan("linear_87", "InnerProduct", out var manifestLinear), Is.True);
        Assert.That(manifestLinear.mode, Is.EqualTo(QuantizedNodeMode.Int8WeightOnly));
        Assert.That(manifest.UsesInt8WeightOnlyForOperator("InnerProduct"), Is.False);
        Assert.That(manifest.UsesInt8WeightOnlyForOperator("LayerNorm"), Is.False);

        const string fp32 = "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"fp32\",\"precision\":{\"activationDtype\":\"FP32\",\"weightDtype\":\"FP32\",\"sensitiveOutputDtype\":\"FP32\"}}";
        var fp32Manifest = AexisModelManifestLoader.LoadFromJson(fp32);
        Assert.That(fp32Manifest.quantization, Is.Null);

        const string w8a8 = "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"bad\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"INT8\",\"sensitiveOutputDtype\":\"FP16\"},\"quantization\":{\"quantizationVersion\":\"v1\",\"calibrationVersion\":\"fixture\",\"calibrationMethod\":\"absmax\",\"weightScheme\":\"INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC\",\"outputChannelAxis\":0,\"symmetric\":true,\"zeroPoint\":0,\"accumulationDtype\":\"FP32\",\"activationQuantized\":true}}";
        var error = Assert.Throws<InferenceContractException>(() => AexisModelManifestLoader.LoadFromJson(w8a8));
        Assert.That(error.Message, Does.Contain("nodePlans"));

        const string selective = "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"ok\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"INT8\",\"sensitiveOutputDtype\":\"FP16\"},\"quantization\":{\"quantizationVersion\":\"aiimage.int8-selective/v1\",\"calibrationVersion\":\"fixture\",\"calibrationMethod\":\"absmax\",\"weightScheme\":\"INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC\",\"outputChannelAxis\":0,\"symmetric\":true,\"zeroPoint\":0,\"accumulationDtype\":\"FP32\",\"activationQuantized\":true,\"quantizedOperators\":[\"Gemm\"],\"nodePlans\":[{\"layerName\":\"gemm_0\",\"operatorName\":\"Gemm\",\"mode\":\"W8A8\",\"activationScale\":0.03125,\"activationZeroPoint\":0}]}}";
        var selectiveManifest = AexisModelManifestLoader.LoadFromJson(selective);
        Assert.That(selectiveManifest.TryGetQuantizedNodePlan("gemm_0", "Gemm", out var plan), Is.True);
        Assert.That(plan.mode, Is.EqualTo(QuantizedNodeMode.Int8W8A8));
    }

    [Test]
    public void WeightPacker_UsesPerOutputChannelSymmetricInt8WithoutFloatExpansion()
    {
        var source = new[] { -2f, -1f, 0f, 1f, 0.25f, -0.5f, 0.75f, -1f };
        var upload = AexisGraphSession.NewInt8WeightOnlyUpload(
            source,
            outputChannels: 2,
            valuesPerOutputChannel: 4,
            outputChannelsAreContiguous: true,
            label: "NcnnInt8WeightOnlyTests");
        try
        {
            var packed = new uint[2];
            var scales = new float[2];
            upload.packedWeights.GetData(packed);
            upload.scales.GetData(scales);

            Assert.That(scales[0], Is.EqualTo(2f / 127f).Within(1e-7f));
            Assert.That(scales[1], Is.EqualTo(1f / 127f).Within(1e-7f));
            Assert.That((byte)(packed[0] >> 0), Is.EqualTo(unchecked((byte)-127)));
            Assert.That((byte)(packed[0] >> 8), Is.EqualTo(unchecked((byte)-64)));
            Assert.That((byte)(packed[0] >> 16), Is.EqualTo(0));
            Assert.That((byte)(packed[0] >> 24), Is.EqualTo(64));
            Assert.That((byte)(packed[1] >> 0), Is.EqualTo(32));
            Assert.That((byte)(packed[1] >> 8), Is.EqualTo(unchecked((byte)-64)));
            Assert.That((byte)(packed[1] >> 16), Is.EqualTo(95));
            Assert.That((byte)(packed[1] >> 24), Is.EqualTo(unchecked((byte)-127)));
        }
        finally
        {
            AexisGpuResourceTracker.ReleaseBuffer(upload.packedWeights, "NcnnInt8WeightOnlyTests");
            AexisGpuResourceTracker.ReleaseBuffer(upload.scales, "NcnnInt8WeightOnlyTests");
            upload.packedWeights.Dispose();
            upload.scales.Dispose();
        }
    }

    [Test]
    public void CapabilityMatrix_AdvertisesOnlyD2WeightKernelOperators()
    {
        Assert.That(AexisOperatorCapabilities.TryGet("Convolution", out var convolution), Is.True);
        Assert.That(AexisOperatorCapabilities.TryGet("Gemm", out var gemm), Is.True);
        Assert.That(AexisOperatorCapabilities.TryGet("InnerProduct", out var innerProduct), Is.True);
        Assert.That(AexisOperatorCapabilities.TryGet("LayerNorm", out var layerNorm), Is.True);

        Assert.That(convolution.int8, Is.True);
        Assert.That(gemm.int8, Is.True);
        Assert.That(innerProduct.int8, Is.True);
        Assert.That(layerNorm.int8, Is.False);
    }

    [Test]
    public void StrictInt8Plan_RejectsWeightedOperatorWithoutD2KernelWithFullTensorDiagnostic()
    {
        var inputLayer = new AexisGraphModel.Layer
        {
            type = AexisLayerTypes.Input,
            typeName = "Input",
            name = "in",
            topNames = new[] { "input" }
        };
        var unsupportedLayer = new AexisGraphModel.Layer
        {
            type = AexisLayerTypes.BatchNorm,
            typeName = "BatchNorm",
            name = "affine",
            bottomNames = new[] { "input" },
            topNames = new[] { "output" }
        };
        var model = new AexisGraphModel
        {
            layers = new List<AexisGraphModel.Layer> { inputLayer, unsupportedLayer }
        };
        var descriptor = new AexisTexturePlanTensorDescriptor
        {
            blob = "input",
            logicalShape = new[] { 3, 8, 4, 1, 4 },
            storageShape = new[] { 1, 8, 4, 1, 1 },
            layout = AexisTexturePlanLayout.Packed4,
            dtype = "FP16",
            textureBacked = true
        };

        var exception = Assert.Throws<StrictTextureInferencePlanException>(() => AexisTextureExecutionPlanner.Compile(
            model,
            new AexisTextureExecutionPlanRequest
            {
                int8WeightOnly = true,
                inputs = new[] { descriptor }
            }));

        Assert.That(exception.Diagnostics.Count, Is.EqualTo(1));
        Assert.That(exception.Diagnostics[0].code, Is.EqualTo("missing-int8-weight-only-kernel"));
        Assert.That(exception.Message, Does.Contain("layer=affine"));
        Assert.That(exception.Message, Does.Contain("blob=input"));
        Assert.That(exception.Message, Does.Contain("logical_shape=[3,8,4,1,4]"));
        Assert.That(exception.Message, Does.Contain("storage_shape=[1,8,4,1,1]"));
        Assert.That(exception.Message, Does.Contain("layout=Packed4"));
        Assert.That(exception.Message, Does.Contain("dtype=FP16"));
        Assert.That(exception.Message, Does.Contain("materialize-from-buffer"));
    }

    [Test]
    public void SelectiveInt8Plan_DoesNotQuantizeUnselectedWeightedOperators()
    {
        var inputLayer = new AexisGraphModel.Layer
        {
            type = AexisLayerTypes.Input,
            typeName = "Input",
            name = "in",
            topNames = new[] { "input" }
        };
        var layerNorm = new AexisGraphModel.Layer
        {
            type = AexisLayerTypes.LayerNorm,
            typeName = "LayerNorm",
            name = "float_norm",
            bottomNames = new[] { "input" },
            topNames = new[] { "output" }
        };
        var plan = AexisTextureExecutionPlanner.Analyze(
            new AexisGraphModel { layers = new List<AexisGraphModel.Layer> { inputLayer, layerNorm } },
            new AexisTextureExecutionPlanRequest
            {
                int8WeightOnly = true,
                int8WeightOnlyOperators = new[] { "InnerProduct" },
                inputs = new[]
                {
                    new AexisTexturePlanTensorDescriptor
                    {
                        blob = "input",
                        logicalShape = new[] { 3, 8, 4, 1, 4 },
                        storageShape = new[] { 1, 8, 4, 1, 1 },
                        layout = AexisTexturePlanLayout.Packed4,
                        dtype = "FP16",
                        textureBacked = true
                    }
                }
            });

        Assert.That(plan.diagnostics, Has.None.Matches<AexisTextureExecutionPlanDiagnostic>(diagnostic =>
            diagnostic.code == "missing-int8-weight-only-kernel"));
    }

    [Test]
    public void SelectiveInt8Plan_WithExplicitLayerNamesDoesNotQuantizeOtherWeightedLayers()
    {
        var inputLayer = new AexisGraphModel.Layer
        {
            type = AexisLayerTypes.Input,
            typeName = "Input",
            name = "in",
            topNames = new[] { "input" }
        };
        var batchNorm = new AexisGraphModel.Layer
        {
            type = AexisLayerTypes.BatchNorm,
            typeName = "BatchNorm",
            name = "float_affine",
            bottomNames = new[] { "input" },
            topNames = new[] { "output" }
        };
        var plan = AexisTextureExecutionPlanner.Analyze(
            new AexisGraphModel { layers = new List<AexisGraphModel.Layer> { inputLayer, batchNorm } },
            new AexisTextureExecutionPlanRequest
            {
                int8WeightOnly = true,
                int8WeightOnlyLayerSelectionExplicit = true,
                int8WeightOnlyLayerNames = new[] { "selected_conv" },
                inputs = new[]
                {
                    new AexisTexturePlanTensorDescriptor
                    {
                        blob = "input",
                        logicalShape = new[] { 3, 8, 4, 1, 4 },
                        storageShape = new[] { 1, 8, 4, 1, 1 },
                        layout = AexisTexturePlanLayout.Packed4,
                        dtype = "FP16",
                        textureBacked = true
                    }
                }
            });

        Assert.That(plan.diagnostics, Has.None.Matches<AexisTextureExecutionPlanDiagnostic>(diagnostic =>
            diagnostic.code == "missing-int8-weight-only-kernel"));
    }

    [Test]
    public void SelectiveInt8Manifest_DefaultsQuantizedOperatorsToW8ExceptFloatOverrides()
    {
        const string json = "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"selective\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"INT8\",\"sensitiveOutputDtype\":\"FP16\"},\"quantization\":{\"quantizationVersion\":\"aiimage.int8-selective/v1\",\"calibrationVersion\":\"fixture\",\"calibrationMethod\":\"absmax\",\"weightScheme\":\"INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC\",\"outputChannelAxis\":0,\"symmetric\":true,\"zeroPoint\":0,\"accumulationDtype\":\"FP32\",\"activationQuantized\":true,\"quantizedOperators\":[\"Convolution\"],\"nodePlans\":[{\"layerName\":\"sensitive\",\"operatorName\":\"Convolution\",\"mode\":\"Float\",\"activationScale\":1.0,\"activationZeroPoint\":0},{\"layerName\":\"calibrated\",\"operatorName\":\"Convolution\",\"mode\":\"W8A8\",\"activationScale\":0.125,\"activationZeroPoint\":0}]}}";
        var manifest = AexisModelManifestLoader.LoadFromJson(json);

        Assert.That(manifest.TryGetQuantizedNodePlan("unplanned", "Convolution", out var defaultPlan), Is.True);
        Assert.That(defaultPlan.mode, Is.EqualTo(QuantizedNodeMode.Int8WeightOnly));
        Assert.That(manifest.TryGetQuantizedNodePlan("sensitive", "Convolution", out var floatPlan), Is.False);
        Assert.That(floatPlan.mode, Is.EqualTo(QuantizedNodeMode.Float));
        Assert.That(manifest.TryGetQuantizedNodePlan("calibrated", "Convolution", out var w8a8Plan), Is.True);
        Assert.That(w8a8Plan.mode, Is.EqualTo(QuantizedNodeMode.Int8W8A8));
        Assert.That(manifest.TryGetQuantizedNodePlan("norm", "LayerNorm", out var noPlan), Is.False);
        Assert.That(noPlan, Is.Null);
    }

    [Test]
    public void SelectiveInt8Manifest_WithEmptyOperatorsUsesExplicitNodePlansOnly()
    {
        const string json = "{\"schemaVersion\":\"aiimage.model-manifest/v1\",\"modelId\":\"explicit\",\"precision\":{\"activationDtype\":\"FP16\",\"weightDtype\":\"INT8\",\"sensitiveOutputDtype\":\"FP16\"},\"quantization\":{\"quantizationVersion\":\"aiimage.int8-selective/v1\",\"calibrationVersion\":\"fixture\",\"calibrationMethod\":\"absmax\",\"weightScheme\":\"INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC\",\"outputChannelAxis\":0,\"symmetric\":true,\"zeroPoint\":0,\"accumulationDtype\":\"FP32\",\"activationQuantized\":true,\"quantizedOperators\":[],\"nodePlans\":[{\"layerName\":\"selected\",\"operatorName\":\"Convolution\",\"mode\":\"W8\",\"activationScale\":1.0,\"activationZeroPoint\":0}]}}";
        var manifest = AexisModelManifestLoader.LoadFromJson(json);

        Assert.That(manifest.TryGetQuantizedNodePlan("selected", "Convolution", out var selected), Is.True);
        Assert.That(selected.mode, Is.EqualTo(QuantizedNodeMode.Int8WeightOnly));
        Assert.That(manifest.TryGetQuantizedNodePlan("unplanned", "Convolution", out var unplanned), Is.False);
        Assert.That(unplanned, Is.Null);
        Assert.That(manifest.UsesInt8WeightOnlyForOperator("Convolution"), Is.False);
    }

    [Test]
    public void ShippingSelectiveInt8Manifests_RecordRunnerNodePlans()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var clip = AexisModelManifestLoader.LoadFromFile(AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "clip-mobileclip-s0.int8.model.json"));
        var matting = AexisModelManifestLoader.LoadFromFile(AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "matting.int8.model.json"));
        var yolo = AexisModelManifestLoader.LoadFromFile(AexisApplicationExamplePaths.ResolveStreamingAssetFilePath("InferenceManifests", "yolo-seg.int8.model.json"));

        Assert.That(clip.TryGetQuantizedNodePlan("gemm_0", "Gemm", out var clipGemm), Is.True);
        Assert.That(clipGemm.mode, Is.EqualTo(QuantizedNodeMode.Int8W8A8));
        Assert.That(clip.TryGetQuantizedNodePlan("linear_87", "InnerProduct", out var clipLinear), Is.True);
        Assert.That(clipLinear.mode, Is.EqualTo(QuantizedNodeMode.Int8WeightOnly));
        Assert.That(matting.TryGetQuantizedNodePlan("Conv_0", "Convolution", out var mattingConv), Is.False);
        Assert.That(mattingConv, Is.Not.Null);
        Assert.That(mattingConv.mode, Is.EqualTo(QuantizedNodeMode.Float));
        Assert.That(matting.TryGetQuantizedNodePlan("Conv_4", "Convolution", out var mattingConv4), Is.False);
        Assert.That(mattingConv4, Is.Not.Null);
        Assert.That(mattingConv4.mode, Is.EqualTo(QuantizedNodeMode.Float));
        Assert.That(matting.TryGetQuantizedNodePlan("Conv_6", "Convolution", out var mattingConv6), Is.False);
        Assert.That(mattingConv6, Is.Null);
        Assert.That(matting.TryGetQuantizedNodePlan("Conv_236", "Convolution", out var mattingConv236), Is.True);
        Assert.That(mattingConv236.mode, Is.EqualTo(QuantizedNodeMode.Int8WeightOnly));
        Assert.That(yolo.TryGetQuantizedNodePlan("conv_0", "Convolution", out var yoloConv), Is.True);
        Assert.That(yoloConv.mode, Is.EqualTo(QuantizedNodeMode.Int8W8A8));
    }

    [Test]
    public void Int8Kernels_ReadPackedWeightsAndAccumulateIntoFloatTextures()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var conv = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Resources", "Aexis", "Includes", "KernelGroups", "AexisKernels.Pack4Conv.hlsl"));
        var gemm = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Resources", "Aexis", "Includes", "KernelGroups", "AexisKernels.Pack4Matmul.hlsl"));
        var bindings = File.ReadAllText(Path.Combine(root, "Packages", "com.aexis", "Runtime", "Resources", "Aexis", "AexisCommon.compute"));

        Assert.That(conv, Does.Contain("AexisReadRawConvWeight"));
        Assert.That(conv, Does.Contain("_ConvWInt8Packed"));
        Assert.That(gemm, Does.Contain("_MatBInt8Packed"));
        Assert.That(gemm, Does.Contain("_MatBInt8Scales[col]"));
        Assert.That(gemm, Does.Contain("AexisQuantizeGemmActivationForInt8"));
        Assert.That(gemm, Does.Contain("_UseInt8Activations"));
        Assert.That(gemm, Does.Contain("float acc"));
        Assert.That(bindings, Does.Contain("StructuredBuffer<uint> _ConvWInt8Packed"));
        Assert.That(bindings, Does.Contain("StructuredBuffer<uint> _MatBInt8Packed"));
        Assert.That(bindings, Does.Not.Contain("RWStructuredBuffer<float> _Int8ExpandedWeights"));
    }
}
#endif
