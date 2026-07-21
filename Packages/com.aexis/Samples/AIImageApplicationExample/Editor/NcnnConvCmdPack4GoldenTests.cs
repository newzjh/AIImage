#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.Linq;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;
using Aexis.Execution;

public sealed class NcnnConvCmdPack4GoldenTests
{
    private sealed class Profile
    {
        public string name;
        public bool deconvolution;
        public int inputChannels;
        public int outputChannels;
        public int group;
        public int inputWidth;
        public int inputHeight;
        public int kernelWidth;
        public int kernelHeight;
        public int strideWidth;
        public int strideHeight;
        public int padLeft;
        public int padTop;
    }

    [Test]
    public void CommandBufferPack4_ConvolutionFamilyMatchesNcnnOihwFp32Reference()
    {
        var profiles = new[]
        {
            new Profile { name = "conv-normal-tail", inputChannels = 3, outputChannels = 5, group = 1, inputWidth = 5, inputHeight = 4, kernelWidth = 2, kernelHeight = 3, strideWidth = 1, strideHeight = 1, padLeft = 0, padTop = 1 },
            new Profile { name = "conv-depthwise-stride-tail", inputChannels = 3, outputChannels = 6, group = 3, inputWidth = 5, inputHeight = 5, kernelWidth = 3, kernelHeight = 3, strideWidth = 2, strideHeight = 2, padLeft = 1, padTop = 1 },
            new Profile { name = "conv-group-tail", inputChannels = 6, outputChannels = 10, group = 2, inputWidth = 4, inputHeight = 3, kernelWidth = 1, kernelHeight = 1, strideWidth = 1, strideHeight = 1, padLeft = 0, padTop = 0 },
            new Profile { name = "deconv-group-tail", deconvolution = true, inputChannels = 6, outputChannels = 10, group = 2, inputWidth = 3, inputHeight = 2, kernelWidth = 2, kernelHeight = 3, strideWidth = 2, strideHeight = 1, padLeft = 0, padTop = 1 },
            new Profile { name = "deconv-depthwise-tail", deconvolution = true, inputChannels = 3, outputChannels = 6, group = 3, inputWidth = 2, inputHeight = 2, kernelWidth = 3, kernelHeight = 2, strideWidth = 1, strideHeight = 2, padLeft = 1, padTop = 0 }
        };

        foreach (var profile in profiles)
            RunProfile(profile);
    }

    [Test]
    public void CommandBufferPack4_ConvLayersRejectRatherThanPublishPlaceholders()
    {
        var root = System.IO.Path.GetDirectoryName(Application.dataPath);
        var layers = new[]
        {
            "AexisConvolutionLayer.cs",
            "AexisConvolutionDepthWiseLayer.cs",
            "AexisDeconvolutionLayer.cs",
            "AexisDeconvolutionDepthWiseLayer.cs"
        };

        foreach (var layer in layers)
        {
            var path = System.IO.Path.Combine(root, "Packages", "com.aexis", "Runtime", "Ncnn", "Layers", layer);
            var source = System.IO.File.ReadAllText(path);
            Assert.That(source, Does.Not.Contain("CmdPlaceholder"), layer);
            Assert.That(source, Does.Not.Contain("PublishCmdTensorLikeInput"), layer);
            Assert.That(source, Does.Contain("rejected_fallback=Buffer/materialize-from-buffer/placeholder"), layer);
        }
    }

    [Test]
    public void CommandBufferPack4_MattingConvSubgraphHasNoBufferFallback()
    {
        var root = System.IO.Path.GetDirectoryName(Application.dataPath);
        var paramPath = System.IO.Path.Combine(root, "Assets", "StreamingAssets", "Matting", "matting.param");
        var binPath = System.IO.Path.Combine(root, "Assets", "StreamingAssets", "Matting", "matting.bin");
        var sourceLines = System.IO.File.ReadAllLines(paramPath);
        Assert.That(sourceLines.Length, Is.GreaterThanOrEqualTo(4), "Matting fixture must contain its input and first convolution.");

        var subgraphParam = "7767517\n2 2\n" + sourceLines[2] + "\n" + sourceLines[3] + "\n";
        using var stream = System.IO.File.OpenRead(binPath);
        using var reader = new NcnnBinReader(stream);
        using var repro = new AexisGraphSession(new AexisOps())
        {
            TensorTextureFormat = RenderTextureFormat.ARGBFloat,
            StrictTextureTargetDtype = "FP32",
            DisallowBufferAccess = true,
            DisallowBufferOutputs = true,
            DisallowBufferToTextureMaterialization = true,
            DisallowInferenceTempComputeBuffers = true
        };
        repro.LoadModel(subgraphParam, reader);

        using var commandBuffer = new CommandBuffer { name = "NcnnMattingConvCmdSubgraph" };
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, 16, 16, 1, RenderTextureFormat.ARGBFloat);
            output = repro.ForwardPack4(
                commandBuffer,
                source,
                new AexisGraphSession.BufferShape(3, 16, 16, 1, 3),
                out var outputShape,
                "input");

            Assert.That(output, Is.Not.Null);
            Assert.That(outputShape, Is.EqualTo(new AexisGraphSession.BufferShape(3, 16, 16, 1, 64)));
            var convNode = repro.LastTextureExecutionPlan.nodes.Single(node => node.layer == "Conv_0");
            Assert.That(repro.LastTextureExecutionPlan.strictEligible, Is.True);
            Assert.That(repro.LastTextureExecutionPlan.dispatchAllowed, Is.True);
            Assert.That(convNode.executionPath, Is.EqualTo("command-buffer-pack4:convolution"));
            Assert.That(convNode.accepted, Is.True);

            repro.ReturnTempArray(commandBuffer, output);
            output = null;
            repro.ReturnTempArray(commandBuffer, source);
            source = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);
            Debug.Log("[NcnnConvCmdPack4GoldenTests] Matting Cmd subgraph passed | layer=Conv_0 | buffer_fallback=false");
        }
        finally
        {
            if (output != null)
                repro.ReturnTempArray(commandBuffer, output);
            if (source != null)
                repro.ReturnTempArray(commandBuffer, source);
        }
    }

    public static void RunBatchValidation()
    {
        var tests = new NcnnConvCmdPack4GoldenTests();
        tests.CommandBufferPack4_ConvolutionFamilyMatchesNcnnOihwFp32Reference();
        tests.CommandBufferPack4_ConvLayersRejectRatherThanPublishPlaceholders();
        tests.CommandBufferPack4_MattingConvSubgraphHasNoBufferFallback();
        Debug.Log("[NcnnConvCmdPack4GoldenTests] passed");
    }

    private static void RunProfile(Profile profile)
    {
        var input = CreateInput(profile);
        var weights = CreateWeights(profile);
        var bias = CreateBias(profile);
        var outputWidth = profile.deconvolution
            ? (profile.inputWidth - 1) * profile.strideWidth + profile.kernelWidth - profile.padLeft * 2
            : (profile.inputWidth + profile.padLeft * 2 - profile.kernelWidth) / profile.strideWidth + 1;
        var outputHeight = profile.deconvolution
            ? (profile.inputHeight - 1) * profile.strideHeight + profile.kernelHeight - profile.padTop * 2
            : (profile.inputHeight + profile.padTop * 2 - profile.kernelHeight) / profile.strideHeight + 1;
        var expected = ReferenceOihw(profile, input, weights, bias, outputWidth, outputHeight);
        var outputPacks = Mathf.CeilToInt(profile.outputChannels / 4f);

        var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        var persistentOutputs = new RenderTexture[outputPacks];
        for (var pack = 0; pack < outputPacks; pack++)
        {
            persistentOutputs[pack] = new RenderTexture(new RenderTextureDescriptor(outputWidth, outputHeight, RenderTextureFormat.ARGBFloat, 0)
            {
                dimension = TextureDimension.Tex2D,
                volumeDepth = 1,
                enableRandomWrite = true,
                msaaSamples = 1
            });
            persistentOutputs[pack].Create();
        }

        using var commandBuffer = new CommandBuffer { name = "NcnnConvCmdPack4Golden:" + profile.name };
        using var inputBuffer = new ComputeBuffer(input.Length, sizeof(float), ComputeBufferType.Structured);
        using var weightBuffer = new ComputeBuffer(weights.Length, sizeof(float), ComputeBufferType.Structured);
        using var biasBuffer = new ComputeBuffer(bias.Length, sizeof(float), ComputeBufferType.Structured);
        inputBuffer.SetData(input);
        weightBuffer.SetData(weights);
        biasBuffer.SetData(bias);

        try
        {
            var source = repro.RentTempArray(commandBuffer, profile.inputWidth, profile.inputHeight, Mathf.CeilToInt(profile.inputChannels / 4f), RenderTextureFormat.ARGBFloat);
            var destination = repro.RentTempArray(commandBuffer, outputWidth, outputHeight, outputPacks, RenderTextureFormat.ARGBFloat);
            repro.Ops.FillPack4FromBufferCHW(commandBuffer, inputBuffer, profile.inputWidth, profile.inputHeight, profile.inputChannels, source);
            if (profile.deconvolution)
            {
                repro.Ops.Deconvolution2dGroupPack4(commandBuffer, source, weightBuffer, biasBuffer, profile.inputChannels, profile.outputChannels, profile.group, profile.kernelWidth, profile.kernelHeight, profile.strideWidth, profile.strideHeight, profile.padLeft, profile.padTop, 1, 1, 0, 0f, destination);
            }
            else
            {
                repro.Ops.Conv2dGroupPack4(commandBuffer, source, weightBuffer, biasBuffer, profile.inputChannels, profile.outputChannels, profile.group, profile.kernelWidth, profile.kernelHeight, profile.strideWidth, profile.strideHeight, profile.padLeft, profile.padTop, 1, 1, 0, 0f, destination);
            }
            for (var pack = 0; pack < outputPacks; pack++)
                commandBuffer.CopyTexture(destination.nameID, pack, 0, persistentOutputs[pack], 0, 0);
            repro.ReturnTempArray(commandBuffer, destination);
            repro.ReturnTempArray(commandBuffer, source);
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var sliceLength = outputWidth * outputHeight;
            var actual = new Vector4[outputPacks * sliceLength];
            for (var pack = 0; pack < outputPacks; pack++)
            {
                var readback = AsyncGPUReadback.Request(persistentOutputs[pack], 0);
                readback.WaitForCompletion();
                Assert.That(readback.hasError, Is.False, profile.name + " pack=" + pack);
                var slice = readback.GetData<Vector4>();
                Assert.That(slice.Length, Is.EqualTo(sliceLength), profile.name + " pack=" + pack);
                for (var index = 0; index < sliceLength; index++)
                    actual[pack * sliceLength + index] = slice[index];
            }
            AssertPack4Equals(profile, actual, expected, outputWidth, outputHeight);
        }
        finally
        {
            foreach (var persistentOutput in persistentOutputs)
            {
                if (persistentOutput == null)
                    continue;
                if (persistentOutput.IsCreated())
                    persistentOutput.Release();
                UnityEngine.Object.DestroyImmediate(persistentOutput);
            }
            repro.Dispose();
        }
    }

    private static float[] CreateInput(Profile profile)
    {
        var values = new float[profile.inputChannels * profile.inputWidth * profile.inputHeight];
        for (var index = 0; index < values.Length; index++)
            values[index] = ((index * 3) % 11 - 5) * 0.125f;
        return values;
    }

    private static float[] CreateWeights(Profile profile)
    {
        var count = profile.outputChannels * (profile.inputChannels / profile.group) * profile.kernelWidth * profile.kernelHeight;
        var values = new float[count];
        for (var index = 0; index < values.Length; index++)
            values[index] = ((index * 5) % 13 - 6) * 0.0625f;
        return values;
    }

    private static float[] CreateBias(Profile profile)
    {
        var values = new float[profile.outputChannels];
        for (var index = 0; index < values.Length; index++)
            values[index] = (index - profile.outputChannels / 2) * 0.05f;
        return values;
    }

    private static float[] ReferenceOihw(Profile profile, float[] input, float[] weights, float[] bias, int outputWidth, int outputHeight)
    {
        var result = new float[profile.outputChannels * outputWidth * outputHeight];
        var inputChannelsPerGroup = profile.inputChannels / profile.group;
        var outputChannelsPerGroup = profile.outputChannels / profile.group;
        for (var outputChannel = 0; outputChannel < profile.outputChannels; outputChannel++)
        {
            var group = outputChannel / outputChannelsPerGroup;
            for (var outputY = 0; outputY < outputHeight; outputY++)
            {
                for (var outputX = 0; outputX < outputWidth; outputX++)
                {
                    var sum = bias[outputChannel];
                    for (var inputChannelLocal = 0; inputChannelLocal < inputChannelsPerGroup; inputChannelLocal++)
                    {
                        var inputChannel = group * inputChannelsPerGroup + inputChannelLocal;
                        for (var kernelY = 0; kernelY < profile.kernelHeight; kernelY++)
                        {
                            var sourceY = profile.deconvolution
                                ? outputY + profile.padTop - kernelY
                                : outputY * profile.strideHeight - profile.padTop + kernelY;
                            if (profile.deconvolution && (sourceY < 0 || sourceY % profile.strideHeight != 0))
                                continue;
                            if (profile.deconvolution)
                                sourceY /= profile.strideHeight;
                            if (sourceY < 0 || sourceY >= profile.inputHeight)
                                continue;
                            for (var kernelX = 0; kernelX < profile.kernelWidth; kernelX++)
                            {
                                var sourceX = profile.deconvolution
                                    ? outputX + profile.padLeft - kernelX
                                    : outputX * profile.strideWidth - profile.padLeft + kernelX;
                                if (profile.deconvolution && (sourceX < 0 || sourceX % profile.strideWidth != 0))
                                    continue;
                                if (profile.deconvolution)
                                    sourceX /= profile.strideWidth;
                                if (sourceX < 0 || sourceX >= profile.inputWidth)
                                    continue;

                                var inputIndex = (inputChannel * profile.inputHeight + sourceY) * profile.inputWidth + sourceX;
                                var weightIndex = (((outputChannel * inputChannelsPerGroup + inputChannelLocal) * profile.kernelHeight + kernelY) * profile.kernelWidth + kernelX);
                                sum += input[inputIndex] * weights[weightIndex];
                            }
                        }
                    }
                    result[(outputChannel * outputHeight + outputY) * outputWidth + outputX] = sum;
                }
            }
        }
        return result;
    }

    private static void AssertPack4Equals(Profile profile, Vector4[] actual, float[] expected, int width, int height)
    {
        for (var channel = 0; channel < profile.outputChannels; channel++)
        {
            var pack = channel / 4;
            var lane = channel % 4;
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var vector = actual[(pack * height + y) * width + x];
                    var observed = lane == 0 ? vector.x : lane == 1 ? vector.y : lane == 2 ? vector.z : vector.w;
                    var reference = expected[(channel * height + y) * width + x];
                    Assert.That(observed, Is.EqualTo(reference).Within(1e-5f), profile.name + " c=" + channel + " x=" + x + " y=" + y);
                }
            }
        }

        var tailStart = profile.outputChannels;
        var totalLanes = Mathf.CeilToInt(profile.outputChannels / 4f) * 4;
        for (var channel = tailStart; channel < totalLanes; channel++)
        {
            var pack = channel / 4;
            var lane = channel % 4;
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var vector = actual[(pack * height + y) * width + x];
                    var observed = lane == 0 ? vector.x : lane == 1 ? vector.y : lane == 2 ? vector.z : vector.w;
                    Assert.That(observed, Is.EqualTo(0f).Within(1e-6f), profile.name + " tail c=" + channel + " x=" + x + " y=" + y);
                }
            }
        }
    }
}
#endif
