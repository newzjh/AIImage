#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;
using Aexis.Execution;

public sealed class NcnnLinearCmdPack4GoldenTests
{
    [Test]
    public void CommandBufferPack4_MatMulMatchesFp32OracleForTailBatchAndTranspose()
    {
        RunMatMulProfile("matmul-tail-broadcast", aRows: 3, k: 5, n: 7, aD: 2, aC: 3, bD: 1, bC: 3, transB: false);
        RunMatMulProfile("matmul-tail-transpose", aRows: 2, k: 5, n: 7, aD: 2, aC: 3, bD: 2, bC: 1, transB: true);
    }

    [Test]
    public void CommandBufferPack4_LinearMatProjectionToPack4MatchesFp32Oracle()
    {
        const int rows = 3;
        const int k = 5;
        const int n = 8;
        var input = CreateValues(rows * k, 0.0625f, 11);
        var weights = CreateValues(n * k, 0.046875f, 13);
        var bias = CreateValues(n, 0.03125f, 7);
        var expected = ReferenceLinearProjection(input, weights, bias, rows, k, n);
        var outputPacks = Mathf.CeilToInt(n / 4f);

        AexisGpuResourceTracker.Reset("NcnnLinearCmdPack4GoldenTests.linear-projection");
        using var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnLinearCmdPack4Golden:linear-projection" };
        using var inputBuffer = CreateBuffer(input);
        using var weightBuffer = CreateBuffer(weights);
        using var biasBuffer = CreateBuffer(bias);
        var persistent = CreatePersistentSlices(outputPacks, rows, 1);
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempMat(commandBuffer, k, rows, RenderTextureFormat.RFloat);
            output = repro.RentTempArray(commandBuffer, outputPacks, rows, 1, RenderTextureFormat.ARGBFloat);
            repro.Ops.FillLinearMatFromBuffer(commandBuffer, inputBuffer, k, rows, source);
            repro.Ops.Gemm2DPack4LinearTextureA(
                commandBuffer,
                source,
                false,
                weightBuffer,
                biasBuffer,
                rows,
                n,
                k,
                transB: true,
                alpha: 1f,
                beta: 1f,
                useC: true,
                broadcastTypeC: 4,
                output);
            CopySlices(commandBuffer, output, persistent);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, source);
            output = null;
            source = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var actual = ReadPack4LinearProjection(persistent[0], outputPacks, rows, n);
            var maxError = AssertClose(actual, expected, "linear-projection", 3e-4f);
            LogReport("linear-projection", "[m=" + rows + ",k=" + k + ",n=" + n + "]", maxError);
        }
        finally
        {
            Return(repro, commandBuffer, ref output);
            Return(repro, commandBuffer, ref source);
            ReleasePersistent(persistent);
        }
    }

    [Test]
    public void CommandBufferPack4_SoftmaxMatchesFp32OracleForEveryLogicalAxis()
    {
        for (var axis = 0; axis < 4; axis++)
            RunSoftmaxProfile(axis);
    }

    [Test]
    public void CommandBufferPack4_LayerNormAndReductionMatchFp32Oracle()
    {
        RunLayerNormProfile(RenderTextureFormat.ARGBFloat);
        RunLayerNormProfile(RenderTextureFormat.ARGBHalf);
        RunReductionProfile(axis: 0);
        RunReductionProfile(axis: 1);
        RunReductionProfile(axis: 2);
    }

    [Test]
    public void CommandBufferPack4_SdpaMaskAndCausalMatchFp32Oracle()
    {
        const int sequence = 3;
        const int embed = 3;
        const int heads = 2;
        var query = CreateValues(heads * sequence * embed, 0.07f, 3);
        var key = CreateValues(heads * sequence * embed, 0.05f, 5);
        var value = CreateValues(heads * sequence * embed, 0.11f, 7);
        var mask = new[] { 0f, -0.25f, -1.5f, 0f, 0f, -0.5f, 0.1f, 0f, 0f };
        var expected = ReferenceSdpa(query, key, value, mask, sequence, embed, heads, 1f / Mathf.Sqrt(embed), causal: true);

        AexisGpuResourceTracker.Reset("NcnnLinearCmdPack4GoldenTests.Sdpa");
        using var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnLinearCmdPack4Golden:sdpa-mask-causal" };
        using var queryBuffer = CreateBuffer(query);
        using var keyBuffer = CreateBuffer(key);
        using var valueBuffer = CreateBuffer(value);
        using var maskBuffer = CreateBuffer(mask);
        var outputPacks = Mathf.CeilToInt(heads / 4f);
        var persistent = CreatePersistentSlices(embed, sequence, outputPacks);

        ComputeTexture qTex = null;
        ComputeTexture kTex = null;
        ComputeTexture vTex = null;
        ComputeTexture maskTex = null;
        ComputeTexture output = null;
        try
        {
            qTex = repro.RentTempArray(commandBuffer, embed, sequence, outputPacks, RenderTextureFormat.ARGBFloat);
            kTex = repro.RentTempArray(commandBuffer, embed, sequence, outputPacks, RenderTextureFormat.ARGBFloat);
            vTex = repro.RentTempArray(commandBuffer, embed, sequence, outputPacks, RenderTextureFormat.ARGBFloat);
            maskTex = repro.RentTempArray(commandBuffer, sequence, sequence, 1, RenderTextureFormat.ARGBFloat);
            output = repro.RentTempArray(commandBuffer, embed, sequence, outputPacks, RenderTextureFormat.ARGBFloat);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, queryBuffer, embed, sequence, 1, heads, qTex);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, keyBuffer, embed, sequence, 1, heads, kTex);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, valueBuffer, embed, sequence, 1, heads, vTex);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, maskBuffer, sequence, sequence, 1, 1, maskTex);
            repro.Ops.SdpaAttentionPack4Cdhw(
                commandBuffer,
                qTex,
                kTex,
                vTex,
                sequence,
                sequence,
                embed,
                embed,
                heads,
                heads,
                1f / Mathf.Sqrt(embed),
                output,
                maskTex,
                causal: true);
            CopySlices(commandBuffer, output, persistent);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, maskTex);
            repro.ReturnTempArray(commandBuffer, vTex);
            repro.ReturnTempArray(commandBuffer, kTex);
            repro.ReturnTempArray(commandBuffer, qTex);
            output = null;
            maskTex = null;
            vTex = null;
            kTex = null;
            qTex = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var actual = ReadPack4Slices(persistent, embed, sequence, heads);
            var maxError = AssertClose(actual, expected, "sdpa-mask-causal", 3e-4f);
            LogReport("sdpa-mask-causal", "[w=" + embed + ",h=" + sequence + ",d=1,c=" + heads + "]", maxError);
        }
        finally
        {
            Return(repro, commandBuffer, ref output);
            Return(repro, commandBuffer, ref maskTex);
            Return(repro, commandBuffer, ref vTex);
            Return(repro, commandBuffer, ref kTex);
            Return(repro, commandBuffer, ref qTex);
            ReleasePersistent(persistent);
        }
    }

    [Test]
    public void CommandBufferPack4_LinearAndAttentionLayersRejectPlaceholders()
    {
        var root = System.IO.Path.GetDirectoryName(Application.dataPath);
        var layers = new[]
        {
            "AexisGemmLayer.cs",
            "AexisMatMulLayer.cs",
            "AexisInnerProductLayer.cs",
            "AexisLayerNormLayer.cs",
            "AexisSoftmaxLayer.cs",
            "AexisReductionLayer.cs",
            "AexisSdpaLayer.cs",
            "AexisMultiHeadAttentionLayer.cs"
        };

        foreach (var layer in layers)
        {
            var path = System.IO.Path.Combine(root, "Packages", "com.aexis", "Runtime", "Ncnn", "Layers", layer);
            var source = System.IO.File.ReadAllText(path);
            Assert.That(source, Does.Not.Contain("PublishCmdPlaceholder"), layer);
            Assert.That(source, Does.Not.Contain("[CmdPlaceholder]"), layer);
            Assert.That(source, Does.Contain("rejectedFallback").Or.Contain("rejected_fallback"), layer);
        }
    }

    public static void RunBatchValidation()
    {
        var tests = new NcnnLinearCmdPack4GoldenTests();
        tests.CommandBufferPack4_MatMulMatchesFp32OracleForTailBatchAndTranspose();
        tests.CommandBufferPack4_LinearMatProjectionToPack4MatchesFp32Oracle();
        tests.CommandBufferPack4_SoftmaxMatchesFp32OracleForEveryLogicalAxis();
        tests.CommandBufferPack4_LayerNormAndReductionMatchFp32Oracle();
        tests.CommandBufferPack4_SdpaMaskAndCausalMatchFp32Oracle();
        tests.CommandBufferPack4_LinearAndAttentionLayersRejectPlaceholders();
        Debug.Log("[NcnnLinearCmdPack4GoldenTests] passed");
    }

    private static void RunMatMulProfile(string name, int aRows, int k, int n, int aD, int aC, int bD, int bC, bool transB)
    {
        var bRows = transB ? n : k;
        var bCols = transB ? k : n;
        var a = CreateValues(aC * aD * aRows * k, 0.0625f, 11);
        var b = CreateValues(bC * bD * bRows * bCols, 0.046875f, 13);
        var expected = ReferenceMatMul(a, b, aRows, k, n, aD, aC, bD, bC, transB);
        var outD = Mathf.Max(aD, bD);
        var outC = Mathf.Max(aC, bC);
        var aPacks = Mathf.CeilToInt(aC / 4f);
        var bPacks = Mathf.CeilToInt(bC / 4f);
        var outPacks = Mathf.CeilToInt(outC / 4f);

        AexisGpuResourceTracker.Reset("NcnnLinearCmdPack4GoldenTests." + name);
        using var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnLinearCmdPack4Golden:" + name };
        using var aBuffer = CreateBuffer(a);
        using var bBuffer = CreateBuffer(b);
        var persistent = CreatePersistentSlices(n, aRows, outD * outPacks);
        ComputeTexture aTex = null;
        ComputeTexture bTex = null;
        ComputeTexture output = null;
        try
        {
            aTex = repro.RentTempArray(commandBuffer, k, aRows, aD * aPacks, RenderTextureFormat.ARGBFloat);
            bTex = repro.RentTempArray(commandBuffer, bCols, bRows, bD * bPacks, RenderTextureFormat.ARGBFloat);
            output = repro.RentTempArray(commandBuffer, n, aRows, outD * outPacks, RenderTextureFormat.ARGBFloat);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, aBuffer, k, aRows, aD, aC, aTex);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, bBuffer, bCols, bRows, bD, bC, bTex);
            repro.Ops.MatMulPack4Cdhw(commandBuffer, aTex, aRows, k, aD, aC, bTex, bRows, bCols, bD, bC, transB, outD, outC, output);
            CopySlices(commandBuffer, output, persistent);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, bTex);
            repro.ReturnTempArray(commandBuffer, aTex);
            output = null;
            bTex = null;
            aTex = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var actual = ReadPack4Slices(persistent, n, aRows, outC, outD);
            var maxError = AssertClose(actual, expected, name, 2e-4f);
            LogReport(name, "[w=" + n + ",h=" + aRows + ",d=" + outD + ",c=" + outC + "]", maxError);
        }
        finally
        {
            Return(repro, commandBuffer, ref output);
            Return(repro, commandBuffer, ref bTex);
            Return(repro, commandBuffer, ref aTex);
            ReleasePersistent(persistent);
        }
    }

    private static void RunSoftmaxProfile(int axis)
    {
        const int w = 3;
        const int h = 4;
        const int d = 2;
        const int c = 3;
        var input = CreateValues(w * h * d * c, 0.09375f, 17);
        var expected = ReferenceSoftmax(input, w, h, d, c, axis);
        var packs = Mathf.CeilToInt(c / 4f);
        var name = "softmax-axis-" + axis;

        AexisGpuResourceTracker.Reset("NcnnLinearCmdPack4GoldenTests." + name);
        using var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnLinearCmdPack4Golden:" + name };
        using var inputBuffer = CreateBuffer(input);
        var persistent = CreatePersistentSlices(w, h, d * packs);
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, w, h, d * packs, RenderTextureFormat.ARGBFloat);
            output = repro.RentTempArray(commandBuffer, w, h, d * packs, RenderTextureFormat.ARGBFloat);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, inputBuffer, w, h, d, c, source);
            repro.Ops.SoftmaxPack4Cdhw(commandBuffer, source, w, h, d, c, output, axis);
            CopySlices(commandBuffer, output, persistent);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, source);
            output = null;
            source = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var actual = ReadPack4Slices(persistent, w, h, c, d);
            var maxError = AssertClose(actual, expected, name, 2e-4f);
            LogReport(name, "[w=" + w + ",h=" + h + ",d=" + d + ",c=" + c + "]", maxError);
        }
        finally
        {
            Return(repro, commandBuffer, ref output);
            Return(repro, commandBuffer, ref source);
            ReleasePersistent(persistent);
        }
    }

    private static void RunLayerNormProfile(RenderTextureFormat textureFormat)
    {
        const int w = 5;
        const int h = 3;
        const int c = 3;
        const float eps = 1e-5f;
        var input = CreateValues(w * h * c, 0.125f, 19);
        var gamma = new[] { 0.9f, 1.1f, 0.8f, 1.2f, 1.0f };
        var beta = new[] { -0.2f, 0.05f, 0.1f, -0.1f, 0.15f };
        var expected = ReferenceLayerNorm(input, gamma, beta, w, h, c, eps);
        var dtype = textureFormat == RenderTextureFormat.ARGBHalf ? "FP16" : "FP32";
        var name = "layernorm-width-" + dtype.ToLowerInvariant();

        AexisGpuResourceTracker.Reset("NcnnLinearCmdPack4GoldenTests." + name);
        using var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = textureFormat };
        using var commandBuffer = new CommandBuffer { name = "NcnnLinearCmdPack4Golden:" + name };
        using var inputBuffer = CreateBuffer(input);
        using var gammaBuffer = CreateBuffer(gamma);
        using var betaBuffer = CreateBuffer(beta);
        var persistent = CreatePersistentSlices(w, h, 1, textureFormat);
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, w, h, 1, textureFormat);
            output = repro.RentTempArray(commandBuffer, w, h, 1, textureFormat);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, inputBuffer, w, h, 1, c, source);
            repro.Ops.LayerNormPack4WidthTex(commandBuffer, source, w, h, 1, c, 1, eps, true, gammaBuffer, betaBuffer, output);
            CopySlices(commandBuffer, output, persistent);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, source);
            output = null;
            source = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var actual = ReadPack4Slices(persistent, w, h, c, format: textureFormat);
            var maxError = AssertClose(actual, expected, name, textureFormat == RenderTextureFormat.ARGBHalf ? 3e-3f : 3e-4f);
            LogReport(name, "[w=" + w + ",h=" + h + ",d=1,c=" + c + "]", maxError, dtype);
        }
        finally
        {
            Return(repro, commandBuffer, ref output);
            Return(repro, commandBuffer, ref source);
            ReleasePersistent(persistent);
        }
    }

    private static void RunReductionProfile(int axis)
    {
        const int inputW = 5;
        const int inputH = 3;
        const float coeff = 0.5f;
        var input = CreateValues(inputW * inputH, 0.125f, 17);
        var outputW = axis == 0 ? inputW : 1;
        var outputH = axis == 1 ? inputH : 1;
        var expected = ReferenceReductionSum(input, inputW, inputH, axis, coeff);
        var name = "reduction-sum-axis-" + axis;

        AexisGpuResourceTracker.Reset("NcnnLinearCmdPack4GoldenTests." + name);
        using var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnLinearCmdPack4Golden:" + name };
        using var inputBuffer = CreateBuffer(input);
        var persistent = CreatePersistentSlices(outputW, outputH, 1);
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, inputW, inputH, 1, RenderTextureFormat.ARGBFloat);
            output = repro.RentTempArray(commandBuffer, outputW, outputH, 1, RenderTextureFormat.ARGBFloat);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, inputBuffer, inputW, inputH, 1, 1, source);
            repro.Ops.ReductionScalar2D(commandBuffer, source, inputW, inputH, axis, 0, coeff, output);
            CopySlices(commandBuffer, output, persistent);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, source);
            output = null;
            source = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            var actual = ReadPack4Slices(persistent, outputW, outputH, 1);
            var maxError = AssertClose(actual, expected, name, 2e-4f);
            LogReport(name, "[w=" + outputW + ",h=" + outputH + ",d=1,c=1]", maxError);
        }
        finally
        {
            Return(repro, commandBuffer, ref output);
            Return(repro, commandBuffer, ref source);
            ReleasePersistent(persistent);
        }
    }

    private static float[] CreateValues(int count, float scale, int period)
    {
        var values = new float[count];
        for (var i = 0; i < count; i++)
            values[i] = ((i * 7) % period - period / 2) * scale;
        return values;
    }

    private static float[] ReferenceLinearProjection(float[] input, float[] weights, float[] bias, int rows, int k, int n)
    {
        var result = new float[rows * n];
        for (var row = 0; row < rows; row++)
        for (var col = 0; col < n; col++)
        {
            var sum = bias[col];
            for (var kk = 0; kk < k; kk++)
                sum += input[row * k + kk] * weights[col * k + kk];
            result[row * n + col] = sum;
        }
        return result;
    }

    private static ComputeBuffer CreateBuffer(float[] values)
    {
        var buffer = new ComputeBuffer(values.Length, sizeof(float), ComputeBufferType.Structured);
        buffer.SetData(values);
        return buffer;
    }

    private static RenderTexture[] CreatePersistentSlices(int width, int height, int slices, RenderTextureFormat format = RenderTextureFormat.ARGBFloat)
    {
        var targets = new RenderTexture[slices];
        for (var i = 0; i < slices; i++)
        {
            targets[i] = new RenderTexture(new RenderTextureDescriptor(width, height, format, 0)
            {
                dimension = TextureDimension.Tex2D,
                volumeDepth = 1,
                enableRandomWrite = true,
                msaaSamples = 1
            });
            targets[i].Create();
        }
        return targets;
    }

    private static void CopySlices(CommandBuffer commandBuffer, ComputeTexture source, RenderTexture[] targets)
    {
        for (var slice = 0; slice < targets.Length; slice++)
            commandBuffer.CopyTexture(source.nameID, slice, 0, targets[slice], 0, 0);
    }

    private static float[] ReadPack4Slices(
        RenderTexture[] slices,
        int width,
        int height,
        int channels,
        int depth = 1,
        RenderTextureFormat format = RenderTextureFormat.ARGBFloat)
    {
        var packs = Mathf.CeilToInt(channels / 4f);
        var result = new float[width * height * depth * channels];
        for (var slice = 0; slice < slices.Length; slice++)
        {
            var readback = AsyncGPUReadback.Request(slices[slice], 0);
            readback.WaitForCompletion();
            Assert.That(readback.hasError, Is.False, "readback slice=" + slice);
            var values = format == RenderTextureFormat.ARGBHalf
                ? default
                : readback.GetData<Vector4>();
            var halfValues = format == RenderTextureFormat.ARGBHalf
                ? readback.GetData<ushort>()
                : default;
            var z = slice / packs;
            var pack = slice - z * packs;
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    for (var lane = 0; lane < 4; lane++)
                    {
                        var channel = pack * 4 + lane;
                        if (z >= depth || channel >= channels)
                            continue;
                        var laneValue = format == RenderTextureFormat.ARGBHalf
                            ? HalfToFloat(halfValues[(y * width + x) * 4 + lane])
                            : values[y * width + x][lane];
                        result[Index(width, height, depth, channels, x, y, z, channel)] = laneValue;
                    }
                }
            }
        }
        return result;
    }

    private static float[] ReadPack4LinearProjection(RenderTexture texture, int packs, int rows, int columns)
    {
        var readback = AsyncGPUReadback.Request(texture, 0);
        readback.WaitForCompletion();
        Assert.That(readback.hasError, Is.False, "linear-projection readback");
        var values = readback.GetData<Vector4>();
        var result = new float[rows * columns];
        for (var row = 0; row < rows; row++)
        for (var pack = 0; pack < packs; pack++)
        {
            var value = values[row * packs + pack];
            var baseColumn = pack * 4;
            if (baseColumn < columns) result[row * columns + baseColumn] = value.x;
            if (baseColumn + 1 < columns) result[row * columns + baseColumn + 1] = value.y;
            if (baseColumn + 2 < columns) result[row * columns + baseColumn + 2] = value.z;
            if (baseColumn + 3 < columns) result[row * columns + baseColumn + 3] = value.w;
        }
        return result;
    }

    private static float HalfToFloat(ushort value)
    {
        var sign = (uint)(value & 0x8000) << 16;
        var exponent = (value >> 10) & 0x1f;
        var mantissa = value & 0x03ff;
        uint bits;
        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                exponent = -14;
                while ((mantissa & 0x0400) == 0)
                {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x03ff;
                bits = sign | (uint)((exponent + 127) << 23) | ((uint)mantissa << 13);
            }
        }
        else if (exponent == 0x1f)
        {
            bits = sign | 0x7f800000u | ((uint)mantissa << 13);
        }
        else
        {
            bits = sign | (uint)((exponent + 112) << 23) | ((uint)mantissa << 13);
        }

        return BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
    }

    private static float[] ReferenceMatMul(float[] a, float[] b, int rows, int k, int n, int aD, int aC, int bD, int bC, bool transB)
    {
        var outD = Mathf.Max(aD, bD);
        var outC = Mathf.Max(aC, bC);
        var result = new float[rows * n * outD * outC];
        var bRows = transB ? n : k;
        var bCols = transB ? k : n;
        for (var batchC = 0; batchC < outC; batchC++)
        for (var batchD = 0; batchD < outD; batchD++)
        for (var row = 0; row < rows; row++)
        for (var col = 0; col < n; col++)
        {
            var sum = 0f;
            for (var kk = 0; kk < k; kk++)
            {
                var aValue = a[Index(k, rows, aD, aC, kk, row, aD == 1 ? 0 : batchD, aC == 1 ? 0 : batchC)];
                var bx = transB ? kk : col;
                var by = transB ? col : kk;
                var bValue = b[Index(bCols, bRows, bD, bC, bx, by, bD == 1 ? 0 : batchD, bC == 1 ? 0 : batchC)];
                sum += aValue * bValue;
            }
            result[Index(n, rows, outD, outC, col, row, batchD, batchC)] = sum;
        }
        return result;
    }

    private static float[] ReferenceSoftmax(float[] input, int w, int h, int d, int c, int axis)
    {
        var output = new float[input.Length];
        for (var channel = 0; channel < c; channel++)
        for (var z = 0; z < d; z++)
        for (var y = 0; y < h; y++)
        for (var x = 0; x < w; x++)
        {
            var length = axis == 0 ? w : axis == 1 ? h : axis == 2 ? d : c;
            var max = float.NegativeInfinity;
            for (var i = 0; i < length; i++)
                max = Mathf.Max(max, input[Index(w, h, d, c, axis == 0 ? i : x, axis == 1 ? i : y, axis == 2 ? i : z, axis == 3 ? i : channel)]);
            var sum = 0f;
            for (var i = 0; i < length; i++)
                sum += Mathf.Exp(input[Index(w, h, d, c, axis == 0 ? i : x, axis == 1 ? i : y, axis == 2 ? i : z, axis == 3 ? i : channel)] - max);
            output[Index(w, h, d, c, x, y, z, channel)] = Mathf.Exp(input[Index(w, h, d, c, x, y, z, channel)] - max) / sum;
        }
        return output;
    }

    private static float[] ReferenceLayerNorm(float[] input, float[] gamma, float[] beta, int w, int h, int c, float eps)
    {
        var output = new float[input.Length];
        for (var channel = 0; channel < c; channel++)
        for (var y = 0; y < h; y++)
        {
            var sum = 0f;
            var sumSquares = 0f;
            for (var x = 0; x < w; x++)
            {
                var value = input[Index(w, h, 1, c, x, y, 0, channel)];
                sum += value;
                sumSquares += value * value;
            }
            var mean = sum / w;
            var variance = sumSquares / w - mean * mean;
            var invStd = 1f / Mathf.Sqrt(Mathf.Max(variance + eps, 1e-20f));
            for (var x = 0; x < w; x++)
                output[Index(w, h, 1, c, x, y, 0, channel)] = (input[Index(w, h, 1, c, x, y, 0, channel)] - mean) * invStd * gamma[x] + beta[x];
        }
        return output;
    }

    private static float[] ReferenceReductionSum(float[] input, int w, int h, int axis, float coeff)
    {
        if (axis == 2)
        {
            var sum = 0f;
            for (var i = 0; i < input.Length; i++)
                sum += input[i];
            return new[] { sum * coeff };
        }

        if (axis == 1)
        {
            var output = new float[h];
            for (var y = 0; y < h; y++)
            {
                var sum = 0f;
                for (var x = 0; x < w; x++)
                    sum += input[y * w + x];
                output[y] = sum * coeff;
            }
            return output;
        }

        var columns = new float[w];
        for (var x = 0; x < w; x++)
        {
            var sum = 0f;
            for (var y = 0; y < h; y++)
                sum += input[y * w + x];
            columns[x] = sum * coeff;
        }
        return columns;
    }

    private static float[] ReferenceSdpa(float[] q, float[] k, float[] v, float[] mask, int sequence, int embed, int heads, float scale, bool causal)
    {
        var output = new float[q.Length];
        for (var head = 0; head < heads; head++)
        for (var row = 0; row < sequence; row++)
        {
            var scores = new float[sequence];
            var max = float.NegativeInfinity;
            for (var col = 0; col < sequence; col++)
            {
                var score = 0f;
                for (var i = 0; i < embed; i++)
                    score += q[Index(embed, sequence, 1, heads, i, row, 0, head)] * k[Index(embed, sequence, 1, heads, i, col, 0, head)];
                score = causal && col > row ? -1e20f : score * scale + mask[col + row * sequence];
                scores[col] = score;
                max = Mathf.Max(max, score);
            }
            var sum = 0f;
            for (var col = 0; col < sequence; col++)
                sum += Mathf.Exp(scores[col] - max);
            for (var outX = 0; outX < embed; outX++)
            {
                var value = 0f;
                for (var col = 0; col < sequence; col++)
                    value += Mathf.Exp(scores[col] - max) / sum * v[Index(embed, sequence, 1, heads, outX, col, 0, head)];
                output[Index(embed, sequence, 1, heads, outX, row, 0, head)] = value;
            }
        }
        return output;
    }

    private static float AssertClose(float[] actual, float[] expected, string name, float tolerance)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), name);
        var maxError = 0f;
        for (var i = 0; i < actual.Length; i++)
        {
            maxError = Mathf.Max(maxError, Mathf.Abs(actual[i] - expected[i]));
            Assert.That(actual[i], Is.EqualTo(expected[i]).Within(tolerance), name + " index=" + i);
        }
        return maxError;
    }

    private static void LogReport(string profile, string shape, float maxError, string dtype = "FP32")
    {
        var stats = AexisGpuResourceTracker.GetStatsSnapshot();
        Debug.Log("[NcnnLinearCmdPack4GoldenTests]"
            + " | profile=" + profile
            + " | shape=" + shape
            + " | dtype=" + dtype
            + " | max_abs_error=" + maxError.ToString("G9")
            + " | peak_rt_bytes=" + stats.peakTemporaryTextureBytes
            + " | peak_rt_count=" + stats.peakTextureCount);
    }

    private static int Index(int w, int h, int d, int c, int x, int y, int z, int channel)
    {
        return (((channel * d + z) * h + y) * w) + x;
    }

    private static void Return(AexisGraphSession repro, CommandBuffer commandBuffer, ref ComputeTexture texture)
    {
        if (texture == null)
            return;
        repro.ReturnTempArray(commandBuffer, texture);
        texture = null;
    }

    private static void ReleasePersistent(RenderTexture[] targets)
    {
        foreach (var target in targets)
        {
            if (target == null)
                continue;
            if (target.IsCreated())
                target.Release();
            UnityEngine.Object.DestroyImmediate(target);
        }
    }
}
#endif
