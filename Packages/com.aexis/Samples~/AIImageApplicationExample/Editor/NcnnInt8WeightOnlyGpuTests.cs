#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.Linq;
using Aexis;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class NcnnInt8WeightOnlyGpuTests
{
    [Test]
    public void CommandBufferKernels_DequantizePackedInt8WeightsPerOutputChannel()
    {
        AssertConvKernel();
        AssertGemmKernel();
        AssertGemmW8A8Kernel();
    }

    private static void AssertConvKernel()
    {
        var input = new[] { 0.25f, -0.5f, 1f };
        var weights = new[] { -1f, 0.5f };
        var bias = new[] { 0.1f, -0.2f };
        var upload = NcnnGraphSession.NewInt8WeightOnlyUpload(weights, 2, 1, true, "NcnnInt8WeightOnlyGpuTests.Conv");
        using var ops = new NcnnOps();
        using var repro = new NcnnGraphSession(ops) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnInt8WeightOnlyGpuTests.Conv" };
        using var inputBuffer = NewFloatBuffer(input);
        using var biasBuffer = NewFloatBuffer(bias);
        var persistent = CreatePersistent(3, 1);
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, 3, 1, 1, RenderTextureFormat.ARGBFloat);
            output = repro.RentTempArray(commandBuffer, 3, 1, 1, RenderTextureFormat.ARGBFloat);
            ops.FillPack4FromBufferCHW(commandBuffer, inputBuffer, 3, 1, 1, source);
            ops.SetInt8ConvWeights(upload.packedWeights, upload.scales);
            ops.Conv2dGroupPack4(commandBuffer, source, upload.packedWeights, biasBuffer, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0f, output);
            commandBuffer.CopyTexture(output.nameID, 0, 0, persistent, 0, 0);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, source);
            output = null;
            source = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var actual = Readback(persistent);
            for (var index = 0; index < input.Length; index++)
            {
                Assert.That(actual[index].x, Is.EqualTo(input[index] * -1f + bias[0]).Within(1e-5f), "conv channel=0 index=" + index);
                Assert.That(actual[index].y, Is.EqualTo(input[index] * 0.5f + bias[1]).Within(1e-5f), "conv channel=1 index=" + index);
            }
        }
        finally
        {
            if (output != null)
                repro.ReturnTempArray(commandBuffer, output);
            if (source != null)
                repro.ReturnTempArray(commandBuffer, source);
            ReleasePersistent(persistent);
            ReleaseUpload(upload, "NcnnInt8WeightOnlyGpuTests.Conv");
        }
    }

    private static void AssertGemmKernel()
    {
        var input = new[] { 1f, -1f, 0.5f, 2f };
        var weights = new[] { 1f, -0.5f, 0.25f, 2f, 0.5f, 1f, -1f, 0.25f };
        var bias = new[] { 0.1f, -0.2f };
        var upload = NcnnGraphSession.NewInt8WeightOnlyUpload(weights, 2, 4, true, "NcnnInt8WeightOnlyGpuTests.Gemm");
        using var ops = new NcnnOps();
        using var repro = new NcnnGraphSession(ops) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnInt8WeightOnlyGpuTests.Gemm" };
        using var inputBuffer = NewFloatBuffer(input);
        using var biasBuffer = NewFloatBuffer(bias);
        using var unusedFloatBinding = NewFloatBuffer(new[] { 0f });
        var persistent = CreatePersistent(1, 1);
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, 1, 1, 1, RenderTextureFormat.ARGBFloat);
            output = repro.RentTempArray(commandBuffer, 1, 1, 1, RenderTextureFormat.ARGBFloat);
            ops.FillPack4FromBufferCHW(commandBuffer, inputBuffer, 1, 1, 4, source);
            ops.SetInt8GemmWeights(upload.packedWeights, upload.scales);
            ops.Gemm2DPack4LinearTextureA(commandBuffer, source, true, unusedFloatBinding, biasBuffer, 1, 2, 4, true, 1f, 1f, true, 4, output);
            commandBuffer.CopyTexture(output.nameID, 0, 0, persistent, 0, 0);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, source);
            output = null;
            source = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var expected0 = bias[0] + DotDequantized(input, weights, 0, 4);
            var expected1 = bias[1] + DotDequantized(input, weights, 1, 4);
            var actual = Readback(persistent)[0];
            Assert.That(actual.x, Is.EqualTo(expected0).Within(1e-5f), "gemm output=0");
            Assert.That(actual.y, Is.EqualTo(expected1).Within(1e-5f), "gemm output=1");
        }
        finally
        {
            if (output != null)
                repro.ReturnTempArray(commandBuffer, output);
            if (source != null)
                repro.ReturnTempArray(commandBuffer, source);
            ReleasePersistent(persistent);
            ReleaseUpload(upload, "NcnnInt8WeightOnlyGpuTests.Gemm");
        }
    }

    private static void AssertGemmW8A8Kernel()
    {
        var input = new[] { 1.1f, -0.9f, 0.3f, 1.7f };
        var weights = new[] { 1f, -0.5f, 0.25f, 2f, 0.5f, 1f, -1f, 0.25f };
        var bias = new[] { 0.1f, -0.2f };
        const float activationScale = 0.5f;
        var upload = NcnnGraphSession.NewInt8WeightOnlyUpload(weights, 2, 4, true, "NcnnInt8WeightOnlyGpuTests.GemmW8A8");
        using var ops = new NcnnOps();
        using var repro = new NcnnGraphSession(ops) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnInt8WeightOnlyGpuTests.GemmW8A8" };
        using var inputBuffer = NewFloatBuffer(input);
        using var biasBuffer = NewFloatBuffer(bias);
        using var unusedFloatBinding = NewFloatBuffer(new[] { 0f });
        var persistent = CreatePersistent(1, 1);
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, 1, 1, 1, RenderTextureFormat.ARGBFloat);
            output = repro.RentTempArray(commandBuffer, 1, 1, 1, RenderTextureFormat.ARGBFloat);
            ops.FillPack4FromBufferCHW(commandBuffer, inputBuffer, 1, 1, 4, source);
            ops.SetInt8GemmWeights(upload.packedWeights, upload.scales);
            ops.SetInt8ActivationQuantization(new QuantizedNodePlan
            {
                mode = QuantizedNodeMode.Int8W8A8,
                activationScale = activationScale,
                activationZeroPoint = 0
            });
            ops.Gemm2DPack4LinearTextureA(commandBuffer, source, true, unusedFloatBinding, biasBuffer, 1, 2, 4, true, 1f, 1f, true, 4, output);
            commandBuffer.CopyTexture(output.nameID, 0, 0, persistent, 0, 0);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, source);
            output = null;
            source = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var quantizedInput = QuantizeActivations(input, activationScale);
            var expected0 = bias[0] + DotDequantized(quantizedInput, weights, 0, 4);
            var expected1 = bias[1] + DotDequantized(quantizedInput, weights, 1, 4);
            var actual = Readback(persistent)[0];
            Assert.That(actual.x, Is.EqualTo(expected0).Within(1e-5f), "gemm W8A8 output=0");
            Assert.That(actual.y, Is.EqualTo(expected1).Within(1e-5f), "gemm W8A8 output=1");
        }
        finally
        {
            if (output != null)
                repro.ReturnTempArray(commandBuffer, output);
            if (source != null)
                repro.ReturnTempArray(commandBuffer, source);
            ReleasePersistent(persistent);
            ReleaseUpload(upload, "NcnnInt8WeightOnlyGpuTests.GemmW8A8");
        }
    }

    private static float DotDequantized(float[] input, float[] weights, int outputChannel, int valuesPerOutputChannel)
    {
        var start = outputChannel * valuesPerOutputChannel;
        var maxAbs = 0f;
        for (var index = 0; index < valuesPerOutputChannel; index++)
            maxAbs = Mathf.Max(maxAbs, Mathf.Abs(weights[start + index]));
        var scale = maxAbs > 0f ? maxAbs / 127f : 1f;
        var sum = 0f;
        for (var index = 0; index < valuesPerOutputChannel; index++)
        {
            var quantized = Mathf.Clamp(Mathf.RoundToInt(weights[start + index] / scale), -127, 127);
            sum += input[index] * quantized * scale;
        }
        return sum;
    }

    private static float[] QuantizeActivations(float[] input, float scale)
    {
        var output = new float[input.Length];
        for (var index = 0; index < input.Length; index++)
        {
            var quantized = Mathf.Clamp(Mathf.RoundToInt(input[index] / scale), -128, 127);
            output[index] = quantized * scale;
        }
        return output;
    }

    private static ComputeBuffer NewFloatBuffer(float[] values)
    {
        var buffer = new ComputeBuffer(values.Length, sizeof(float), ComputeBufferType.Structured);
        buffer.SetData(values);
        return buffer;
    }

    private static RenderTexture CreatePersistent(int width, int height)
    {
        var texture = new RenderTexture(new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGBFloat, 0)
        {
            dimension = TextureDimension.Tex2D,
            volumeDepth = 1,
            enableRandomWrite = true,
            msaaSamples = 1
        });
        texture.Create();
        return texture;
    }

    private static Vector4[] Readback(RenderTexture texture)
    {
        var request = AsyncGPUReadback.Request(texture, 0);
        request.WaitForCompletion();
        Assert.That(request.hasError, Is.False, "INT8 output readback");
        return request.GetData<Vector4>().ToArray();
    }

    private static void ReleasePersistent(RenderTexture texture)
    {
        if (texture == null)
            return;
        if (texture.IsCreated())
            texture.Release();
        UnityEngine.Object.DestroyImmediate(texture);
    }

    private static void ReleaseUpload(NcnnGraphSession.Int8WeightOnlyUpload upload, string label)
    {
        if (upload?.packedWeights != null)
        {
            NcnnGpuResourceTracker.ReleaseBuffer(upload.packedWeights, label);
            upload.packedWeights.Dispose();
        }
        if (upload?.scales != null)
        {
            NcnnGpuResourceTracker.ReleaseBuffer(upload.scales, label);
            upload.scales.Dispose();
        }
    }
}
#endif
