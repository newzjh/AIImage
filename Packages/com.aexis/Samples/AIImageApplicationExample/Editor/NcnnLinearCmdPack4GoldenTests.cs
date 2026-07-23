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
    public void CommandBufferPack4_Pack4LinearMatProjectionToPack4MatchesFp32Oracle()
    {
        const int rows = 3;
        const int k = 8;
        const int n = 12;
        var input = CreateValues(rows * k, 0.0625f, 17);
        var weights = CreateValues(n * k, 0.046875f, 19);
        var bias = CreateValues(n, 0.03125f, 11);
        var expected = ReferenceLinearProjection(input, weights, bias, rows, k, n);
        var inputPacks = Mathf.CeilToInt(k / 4f);
        var outputPacks = Mathf.CeilToInt(n / 4f);
        var packedInput = new float[inputPacks * rows * 4];
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < k; column++)
            {
                var pack = column / 4;
                var lane = column % 4;
                packedInput[((lane * rows + row) * inputPacks) + pack] = input[row * k + column];
            }
        }

        AexisGpuResourceTracker.Reset("NcnnLinearCmdPack4GoldenTests.pack4-linear-projection");
        using var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnLinearCmdPack4Golden:pack4-linear-projection" };
        using var inputBuffer = CreateBuffer(packedInput);
        using var weightBuffer = CreateBuffer(weights);
        using var biasBuffer = CreateBuffer(bias);
        var persistent = CreatePersistentSlices(outputPacks, rows, 1);
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, inputPacks, rows, 1, RenderTextureFormat.ARGBFloat);
            output = repro.RentTempArray(commandBuffer, outputPacks, rows, 1, RenderTextureFormat.ARGBFloat);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, inputBuffer, inputPacks, rows, 1, 4, source);
            repro.Ops.Gemm2DPack4LinearTextureA(
                commandBuffer,
                source,
                true,
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
            var maxError = AssertClose(actual, expected, "pack4-linear-projection", 3e-4f);
            LogReport("pack4-linear-projection", "[m=" + rows + ",k=" + k + ",n=" + n + "]", maxError);
        }
        finally
        {
            Return(repro, commandBuffer, ref output);
            Return(repro, commandBuffer, ref source);
            ReleasePersistent(persistent);
        }
    }

    [Test]
    public void CommandBufferPack4_SdpaKvCacheConcatAndCausalOffsetMatchFp32Oracle()
    {
        const int pastSequence = 3;
        const int currentSequence = 2;
        const int totalSequence = pastSequence + currentSequence;
        const int embed = 3;
        const int heads = 5;
        var query = CreateValues(heads * currentSequence * embed, 0.07f, 3);
        var pastKey = CreateValues(heads * pastSequence * embed, 0.05f, 5);
        var currentKey = CreateValues(heads * currentSequence * embed, 0.04f, 11);
        var pastValue = CreateValues(heads * pastSequence * embed, 0.11f, 7);
        var currentValue = CreateValues(heads * currentSequence * embed, 0.09f, 13);
        var expectedKeyCache = ConcatSequenceReference(pastKey, currentKey, embed, pastSequence, currentSequence, heads);
        var expectedValueCache = ConcatSequenceReference(pastValue, currentValue, embed, pastSequence, currentSequence, heads);
        var expectedOutput = ReferenceSdpaWithCache(
            query,
            expectedKeyCache,
            expectedValueCache,
            currentSequence,
            totalSequence,
            embed,
            heads,
            1f / Mathf.Sqrt(embed),
            causal: true,
            causalQueryOffset: pastSequence);

        AexisGpuResourceTracker.Reset("NcnnLinearCmdPack4GoldenTests.SdpaKvCache");
        using var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnLinearCmdPack4Golden:sdpa-kv-cache" };
        using var queryBuffer = CreateBuffer(query);
        using var pastKeyBuffer = CreateBuffer(pastKey);
        using var currentKeyBuffer = CreateBuffer(currentKey);
        using var pastValueBuffer = CreateBuffer(pastValue);
        using var currentValueBuffer = CreateBuffer(currentValue);
        var headPacks = Mathf.CeilToInt(heads / 4f);
        var outputPersistent = CreatePersistentSlices(embed, currentSequence, headPacks);
        var keyCachePersistent = CreatePersistentSlices(embed, totalSequence, headPacks);
        var valueCachePersistent = CreatePersistentSlices(embed, totalSequence, headPacks);

        ComputeTexture qTex = null;
        ComputeTexture pastKTex = null;
        ComputeTexture currentKTex = null;
        ComputeTexture pastVTex = null;
        ComputeTexture currentVTex = null;
        ComputeTexture keyCacheTex = null;
        ComputeTexture valueCacheTex = null;
        ComputeTexture output = null;
        try
        {
            qTex = repro.RentTempArray(commandBuffer, embed, currentSequence, headPacks, RenderTextureFormat.ARGBFloat);
            pastKTex = repro.RentTempArray(commandBuffer, embed, pastSequence, headPacks, RenderTextureFormat.ARGBFloat);
            currentKTex = repro.RentTempArray(commandBuffer, embed, currentSequence, headPacks, RenderTextureFormat.ARGBFloat);
            pastVTex = repro.RentTempArray(commandBuffer, embed, pastSequence, headPacks, RenderTextureFormat.ARGBFloat);
            currentVTex = repro.RentTempArray(commandBuffer, embed, currentSequence, headPacks, RenderTextureFormat.ARGBFloat);
            keyCacheTex = repro.RentTempArray(commandBuffer, embed, totalSequence, headPacks, RenderTextureFormat.ARGBFloat);
            valueCacheTex = repro.RentTempArray(commandBuffer, embed, totalSequence, headPacks, RenderTextureFormat.ARGBFloat);
            output = repro.RentTempArray(commandBuffer, embed, currentSequence, headPacks, RenderTextureFormat.ARGBFloat);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, queryBuffer, embed, currentSequence, 1, heads, qTex);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, pastKeyBuffer, embed, pastSequence, 1, heads, pastKTex);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, currentKeyBuffer, embed, currentSequence, 1, heads, currentKTex);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, pastValueBuffer, embed, pastSequence, 1, heads, pastVTex);
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, currentValueBuffer, embed, currentSequence, 1, heads, currentVTex);
            repro.Ops.ConcatSequencePack4Cdhw(commandBuffer, pastKTex, currentKTex, pastSequence, currentSequence, keyCacheTex);
            repro.Ops.ConcatSequencePack4Cdhw(commandBuffer, pastVTex, currentVTex, pastSequence, currentSequence, valueCacheTex);
            repro.Ops.SdpaAttentionPack4Cdhw(
                commandBuffer,
                qTex,
                keyCacheTex,
                valueCacheTex,
                currentSequence,
                totalSequence,
                embed,
                embed,
                heads,
                heads,
                1f / Mathf.Sqrt(embed),
                output,
                causal: true,
                causalQueryOffset: pastSequence);
            CopySlices(commandBuffer, output, outputPersistent);
            CopySlices(commandBuffer, keyCacheTex, keyCachePersistent);
            CopySlices(commandBuffer, valueCacheTex, valueCachePersistent);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, valueCacheTex);
            repro.ReturnTempArray(commandBuffer, keyCacheTex);
            repro.ReturnTempArray(commandBuffer, currentVTex);
            repro.ReturnTempArray(commandBuffer, pastVTex);
            repro.ReturnTempArray(commandBuffer, currentKTex);
            repro.ReturnTempArray(commandBuffer, pastKTex);
            repro.ReturnTempArray(commandBuffer, qTex);
            output = null;
            valueCacheTex = null;
            keyCacheTex = null;
            currentVTex = null;
            pastVTex = null;
            currentKTex = null;
            pastKTex = null;
            qTex = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            AssertClose(ReadPack4Slices(outputPersistent, embed, currentSequence, heads), expectedOutput, "sdpa-kv-cache-output", 3e-4f);
            AssertClose(ReadPack4Slices(keyCachePersistent, embed, totalSequence, heads), expectedKeyCache, "sdpa-kv-cache-key", 1e-5f);
            AssertClose(ReadPack4Slices(valueCachePersistent, embed, totalSequence, heads), expectedValueCache, "sdpa-kv-cache-value", 1e-5f);
        }
        finally
        {
            Return(repro, commandBuffer, ref output);
            Return(repro, commandBuffer, ref valueCacheTex);
            Return(repro, commandBuffer, ref keyCacheTex);
            Return(repro, commandBuffer, ref currentVTex);
            Return(repro, commandBuffer, ref pastVTex);
            Return(repro, commandBuffer, ref currentKTex);
            Return(repro, commandBuffer, ref pastKTex);
            Return(repro, commandBuffer, ref qTex);
            ReleasePersistent(outputPersistent);
            ReleasePersistent(keyCachePersistent);
            ReleasePersistent(valueCachePersistent);
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
            var path = System.IO.Path.Combine(root, "Packages", "com.aexis", "Runtime", "Execution", "Layers", layer);
            var source = System.IO.File.ReadAllText(path);
            Assert.That(source, Does.Not.Contain("PublishCmdPlaceholder"), layer);
            Assert.That(source, Does.Not.Contain("[CmdPlaceholder]"), layer);
            Assert.That(source, Does.Contain("rejectedFallback").Or.Contain("rejected_fallback"), layer);
        }
    }

    [Test]
    public void CommandBufferPack4_GfpganDynamicWeightsStayTextureNativeAndMatchFp32Oracle()
    {
        const int styleInputDim = 5;
        const int hiddenDim = 8;
        const int output3x3Channels = 5;
        const int output1x1Channels = 3;
        const int width = 3;
        const int height = 2;
        const int kernel3x3 = 9;
        var inputPacks = hiddenDim / 4;
        var output3x3Packs = Mathf.CeilToInt(output3x3Channels / 4f);
        var output1x1Packs = Mathf.CeilToInt(output1x1Channels / 4f);
        var styleInput = CreateValues(styleInputDim, 0.0625f, 11);
        var modulationW = CreateValues(hiddenDim * styleInputDim, 0.046875f, 13);
        var modulationB = CreateValues(hiddenDim, 0.03125f, 7);
        var self3x3 = CreateValues(output3x3Channels * hiddenDim * kernel3x3, 0.0234375f, 17);
        var self1x1 = CreateValues(output1x1Channels * hiddenDim, 0.0390625f, 19);
        var input = CreateValues(width * height * hiddenDim, 0.09375f, 23);
        var expectedStyle = ReferenceGfpganStyle(styleInput, modulationW, modulationB, styleInputDim, hiddenDim);
        var expectedDemod = ReferenceGfpganDemod(expectedStyle, self3x3, hiddenDim, output3x3Channels, kernel3x3);
        var expectedDyn3x3 = ReferenceGfpganDynamicWeights(expectedStyle, expectedDemod, self3x3, hiddenDim, output3x3Channels, kernel3x3, true);
        var expectedDyn1x1 = ReferenceGfpganDynamicWeights(expectedStyle, null, self1x1, hiddenDim, output1x1Channels, 1, false);
        var expectedOutput3x3 = ReferenceGfpganConv(input, width, height, hiddenDim, expectedDyn3x3, output3x3Channels, kernel3x3, 1);
        var expectedOutput1x1 = ReferenceGfpganConv(input, width, height, hiddenDim, expectedDyn1x1, output1x1Channels, 1, 0);
        var dyn3x3Count = output3x3Packs * inputPacks * kernel3x3 * 4;
        var dyn1x1Count = output1x1Packs * inputPacks * 4;

        AexisGpuResourceTracker.Reset("NcnnLinearCmdPack4GoldenTests.GfpganDynamicWeights");
        using var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        using var commandBuffer = new CommandBuffer { name = "NcnnLinearCmdPack4Golden:gfpgan-dynamic" };
        using var styleInputBuffer = CreateBuffer(styleInput);
        using var modulationWBuffer = CreateBuffer(modulationW);
        using var modulationBBuffer = CreateBuffer(modulationB);
        using var self3x3Buffer = CreateBuffer(self3x3);
        using var self1x1Buffer = CreateBuffer(self1x1);
        using var inputBuffer = CreateBuffer(input);
        using var zeroBias3x3 = CreateVector4Buffer(output3x3Packs);
        using var zeroBias1x1 = CreateVector4Buffer(output1x1Packs);
        var stylePersistent = CreatePersistentSlices(hiddenDim, 1, 1, RenderTextureFormat.RFloat);
        var demodPersistent = CreatePersistentSlices(output3x3Channels, 1, 1, RenderTextureFormat.RFloat);
        var dyn3x3Persistent = CreatePersistentSlices(dyn3x3Count, 1, 1);
        var dyn1x1Persistent = CreatePersistentSlices(dyn1x1Count, 1, 1);
        var output3x3Persistent = CreatePersistentSlices(width, height, output3x3Packs);
        var output1x1Persistent = CreatePersistentSlices(width, height, output1x1Packs);
        var noiseAPersistent = CreatePersistentSlices(width, height, inputPacks);
        var noiseBPersistent = CreatePersistentSlices(width, height, inputPacks);
        ComputeTexture styleTexture = null;
        ComputeTexture styleVector = null;
        ComputeTexture demod = null;
        ComputeTexture dynamic3x3 = null;
        ComputeTexture dynamic1x1 = null;
        ComputeTexture source = null;
        ComputeTexture output3x3 = null;
        ComputeTexture output1x1 = null;
        ComputeTexture noiseA = null;
        ComputeTexture noiseB = null;
        try
        {
            styleTexture = repro.RentTempMat(commandBuffer, styleInputDim, 1, RenderTextureFormat.RFloat);
            styleVector = repro.RentTempMat(commandBuffer, hiddenDim, 1, RenderTextureFormat.RFloat);
            demod = repro.RentTempMat(commandBuffer, output3x3Channels, 1, RenderTextureFormat.RFloat);
            dynamic3x3 = repro.RentTempArray(commandBuffer, dyn3x3Count, 1, 1, RenderTextureFormat.ARGBFloat);
            dynamic1x1 = repro.RentTempArray(commandBuffer, dyn1x1Count, 1, 1, RenderTextureFormat.ARGBFloat);
            source = repro.RentTempArray(commandBuffer, width, height, inputPacks, RenderTextureFormat.ARGBFloat);
            output3x3 = repro.RentTempArray(commandBuffer, width, height, output3x3Packs, RenderTextureFormat.ARGBFloat);
            output1x1 = repro.RentTempArray(commandBuffer, width, height, output1x1Packs, RenderTextureFormat.ARGBFloat);
            noiseA = repro.RentTempArray(commandBuffer, width, height, inputPacks, RenderTextureFormat.ARGBFloat);
            noiseB = repro.RentTempArray(commandBuffer, width, height, inputPacks, RenderTextureFormat.ARGBFloat);

            repro.Ops.FillLinearMatFromBuffer(commandBuffer, styleInputBuffer, styleInputDim, 1, styleTexture);
            repro.Ops.GfpganStyleModulation(commandBuffer, styleTexture, 0, styleInputDim, modulationWBuffer, modulationBBuffer, hiddenDim, styleVector);
            repro.Ops.GfpganStyleDemod(commandBuffer, styleVector, self3x3Buffer, hiddenDim, output3x3Channels, kernel3x3, demod);
            repro.Ops.GfpganBuildDynamicWeight(commandBuffer, styleVector, demod, self3x3Buffer, hiddenDim, output3x3Channels, kernel3x3, true, dynamic3x3);
            repro.Ops.GfpganBuildDynamicWeight(commandBuffer, styleVector, null, self1x1Buffer, hiddenDim, output1x1Channels, 1, false, dynamic1x1);
            repro.Ops.FillPack4FromBufferCHW(commandBuffer, inputBuffer, width, height, hiddenDim, source);
            repro.Ops.Conv3x3Pack4DynamicWeight(commandBuffer, source, inputPacks, dynamic3x3, zeroBias3x3, output3x3Packs, 1, 0, 0f, output3x3);
            repro.Ops.Conv1x1Pack4DynamicWeight(commandBuffer, source, inputPacks, dynamic1x1, zeroBias1x1, output1x1Packs, 0, 0f, output1x1);
            repro.Ops.FillPack4FromBufferCHW(commandBuffer, inputBuffer, width, height, hiddenDim, noiseA);
            repro.Ops.FillPack4FromBufferCHW(commandBuffer, inputBuffer, width, height, hiddenDim, noiseB);
            repro.Ops.GfpganAddNoisePack4(commandBuffer, noiseA, 0.25f, 12345, inputPacks);
            repro.Ops.GfpganAddNoisePack4(commandBuffer, noiseB, 0.25f, 12345, inputPacks);

            commandBuffer.CopyTexture(styleVector.nameID, 0, 0, stylePersistent[0], 0, 0);
            commandBuffer.CopyTexture(demod.nameID, 0, 0, demodPersistent[0], 0, 0);
            CopySlices(commandBuffer, dynamic3x3, dyn3x3Persistent);
            CopySlices(commandBuffer, dynamic1x1, dyn1x1Persistent);
            CopySlices(commandBuffer, output3x3, output3x3Persistent);
            CopySlices(commandBuffer, output1x1, output1x1Persistent);
            CopySlices(commandBuffer, noiseA, noiseAPersistent);
            CopySlices(commandBuffer, noiseB, noiseBPersistent);

            repro.ReturnTempArray(commandBuffer, noiseB); noiseB = null;
            repro.ReturnTempArray(commandBuffer, noiseA); noiseA = null;
            repro.ReturnTempArray(commandBuffer, output1x1); output1x1 = null;
            repro.ReturnTempArray(commandBuffer, output3x3); output3x3 = null;
            repro.ReturnTempArray(commandBuffer, source); source = null;
            repro.ReturnTempArray(commandBuffer, dynamic1x1); dynamic1x1 = null;
            repro.ReturnTempArray(commandBuffer, dynamic3x3); dynamic3x3 = null;
            repro.ReturnTempArray(commandBuffer, demod); demod = null;
            repro.ReturnTempArray(commandBuffer, styleVector); styleVector = null;
            repro.ReturnTempArray(commandBuffer, styleTexture); styleTexture = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            AssertClose(ReadbackScalars(stylePersistent[0]), expectedStyle, "gfpgan-style", 3e-4f);
            AssertClose(ReadbackScalars(demodPersistent[0]), expectedDemod, "gfpgan-demod", 4e-4f);
            AssertClose(Flatten(ReadbackVectors(dyn3x3Persistent)), expectedDyn3x3, "gfpgan-dynamic-3x3", 5e-4f);
            AssertClose(Flatten(ReadbackVectors(dyn1x1Persistent)), expectedDyn1x1, "gfpgan-dynamic-1x1", 3e-4f);
            AssertClose(ReadPack4Slices(output3x3Persistent, width, height, output3x3Channels), expectedOutput3x3, "gfpgan-conv-3x3", 6e-4f);
            AssertClose(ReadPack4Slices(output1x1Persistent, width, height, output1x1Channels), expectedOutput1x1, "gfpgan-conv-1x1", 4e-4f);
            var noiseAValues = ReadPack4Slices(noiseAPersistent, width, height, hiddenDim);
            AssertClose(noiseAValues, ReadPack4Slices(noiseBPersistent, width, height, hiddenDim), "gfpgan-noise-deterministic", 0f);
            AssertGfpganNoiseIsBroadcast(input, noiseAValues, width, height, hiddenDim);
        }
        finally
        {
            Return(repro, commandBuffer, ref noiseB); Return(repro, commandBuffer, ref noiseA);
            Return(repro, commandBuffer, ref output1x1); Return(repro, commandBuffer, ref output3x3);
            Return(repro, commandBuffer, ref source); Return(repro, commandBuffer, ref dynamic1x1);
            Return(repro, commandBuffer, ref dynamic3x3); Return(repro, commandBuffer, ref demod);
            Return(repro, commandBuffer, ref styleVector); Return(repro, commandBuffer, ref styleTexture);
            ReleasePersistent(stylePersistent); ReleasePersistent(demodPersistent);
            ReleasePersistent(dyn3x3Persistent); ReleasePersistent(dyn1x1Persistent);
            ReleasePersistent(output3x3Persistent); ReleasePersistent(output1x1Persistent);
            ReleasePersistent(noiseAPersistent); ReleasePersistent(noiseBPersistent);
        }
    }

    public static void RunBatchValidation()
    {
        var tests = new NcnnLinearCmdPack4GoldenTests();
        tests.CommandBufferPack4_MatMulMatchesFp32OracleForTailBatchAndTranspose();
        tests.CommandBufferPack4_LinearMatProjectionToPack4MatchesFp32Oracle();
        tests.CommandBufferPack4_Pack4LinearMatProjectionToPack4MatchesFp32Oracle();
        tests.CommandBufferPack4_SoftmaxMatchesFp32OracleForEveryLogicalAxis();
        tests.CommandBufferPack4_LayerNormAndReductionMatchFp32Oracle();
        tests.CommandBufferPack4_SdpaMaskAndCausalMatchFp32Oracle();
        tests.CommandBufferPack4_SdpaKvCacheConcatAndCausalOffsetMatchFp32Oracle();
        tests.CommandBufferPack4_GfpganDynamicWeightsStayTextureNativeAndMatchFp32Oracle();
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

    private static ComputeBuffer CreateVector4Buffer(int count)
    {
        var buffer = new ComputeBuffer(count, sizeof(float) * 4, ComputeBufferType.Structured);
        buffer.SetData(new Vector4[count]);
        return buffer;
    }

    private static float[] ReadbackScalars(RenderTexture texture)
    {
        var request = AsyncGPUReadback.Request(texture, 0);
        request.WaitForCompletion();
        Assert.That(request.hasError, Is.False, "scalar readback");
        return request.GetData<float>().ToArray();
    }

    private static Vector4[] ReadbackVectors(RenderTexture[] textures)
    {
        var count = 0;
        for (var i = 0; i < textures.Length; i++)
            count += textures[i].width * textures[i].height;
        var result = new Vector4[count];
        var offset = 0;
        for (var i = 0; i < textures.Length; i++)
        {
            var request = AsyncGPUReadback.Request(textures[i], 0);
            request.WaitForCompletion();
            Assert.That(request.hasError, Is.False, "vector readback slice=" + i);
            var values = request.GetData<Vector4>().ToArray();
            Array.Copy(values, 0, result, offset, values.Length);
            offset += values.Length;
        }
        return result;
    }

    private static float[] Flatten(Vector4[] values)
    {
        var output = new float[values.Length * 4];
        for (var i = 0; i < values.Length; i++)
        {
            output[i * 4 + 0] = values[i].x;
            output[i * 4 + 1] = values[i].y;
            output[i * 4 + 2] = values[i].z;
            output[i * 4 + 3] = values[i].w;
        }
        return output;
    }

    private static float[] ReferenceGfpganStyle(float[] styles, float[] modulationW, float[] modulationB, int inputDim, int outputDim)
    {
        var output = new float[outputDim];
        for (var outputChannel = 0; outputChannel < outputDim; outputChannel++)
        {
            var sum = modulationB[outputChannel];
            var offset = outputChannel * inputDim;
            for (var inputChannel = 0; inputChannel < inputDim; inputChannel++)
                sum += modulationW[offset + inputChannel] * styles[inputChannel];
            output[outputChannel] = sum;
        }
        return output;
    }

    private static float[] ReferenceGfpganDemod(float[] style, float[] selfWeight, int hiddenDim, int outputChannels, int kernelArea)
    {
        var output = new float[outputChannels];
        for (var outputChannel = 0; outputChannel < outputChannels; outputChannel++)
        {
            var sum = 0f;
            var outputBase = outputChannel * hiddenDim * kernelArea;
            for (var inputChannel = 0; inputChannel < hiddenDim; inputChannel++)
            for (var kernel = 0; kernel < kernelArea; kernel++)
            {
                var value = selfWeight[outputBase + inputChannel * kernelArea + kernel] * style[inputChannel];
                sum += value * value;
            }
            output[outputChannel] = 1f / Mathf.Sqrt(sum + 1e-8f);
        }
        return output;
    }

    private static float[] ReferenceGfpganDynamicWeights(float[] style, float[] demod, float[] selfWeight, int hiddenDim, int outputChannels, int kernelArea, bool useDemod)
    {
        var inputPacks = hiddenDim / 4;
        var outputPacks = Mathf.CeilToInt(outputChannels / 4f);
        var output = new float[outputPacks * inputPacks * kernelArea * 4 * 4];
        for (var outputPack = 0; outputPack < outputPacks; outputPack++)
        for (var inputPack = 0; inputPack < inputPacks; inputPack++)
        for (var kernel = 0; kernel < kernelArea; kernel++)
        for (var outputLane = 0; outputLane < 4; outputLane++)
        {
            var outputChannel = outputPack * 4 + outputLane;
            var vectorOffset = ((((outputPack * inputPacks + inputPack) * kernelArea + kernel) * 4 + outputLane) * 4);
            if (outputChannel >= outputChannels)
                continue;
            var scale = useDemod ? demod[outputChannel] : 1f;
            var outputBase = outputChannel * hiddenDim * kernelArea;
            for (var inputLane = 0; inputLane < 4; inputLane++)
            {
                var inputChannel = inputPack * 4 + inputLane;
                output[vectorOffset + inputLane] = selfWeight[outputBase + inputChannel * kernelArea + kernel] * style[inputChannel] * scale;
            }
        }
        return output;
    }

    private static float[] ReferenceGfpganConv(float[] input, int width, int height, int inputChannels, float[] dynamicWeights, int outputChannels, int kernelArea, int pad)
    {
        var output = new float[width * height * outputChannels];
        var inputPacks = inputChannels / 4;
        var kernelWidth = kernelArea == 1 ? 1 : 3;
        for (var outputChannel = 0; outputChannel < outputChannels; outputChannel++)
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            var sum = 0f;
            var outputPack = outputChannel / 4;
            var outputLane = outputChannel & 3;
            for (var inputPack = 0; inputPack < inputPacks; inputPack++)
            for (var ky = 0; ky < kernelWidth; ky++)
            for (var kx = 0; kx < kernelWidth; kx++)
            {
                var sourceX = x + kx - pad;
                var sourceY = y + ky - pad;
                if (sourceX < 0 || sourceX >= width || sourceY < 0 || sourceY >= height)
                    continue;
                var kernel = ky * kernelWidth + kx;
                var vectorOffset = ((((outputPack * inputPacks + inputPack) * kernelArea + kernel) * 4 + outputLane) * 4);
                for (var inputLane = 0; inputLane < 4; inputLane++)
                {
                    var inputChannel = inputPack * 4 + inputLane;
                    sum += input[(inputChannel * height + sourceY) * width + sourceX] * dynamicWeights[vectorOffset + inputLane];
                }
            }
            output[(outputChannel * height + y) * width + x] = sum;
        }
        return output;
    }

    private static void AssertGfpganNoiseIsBroadcast(float[] input, float[] noisy, int width, int height, int channels)
    {
        for (var pack = 0; pack < channels / 4; pack++)
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            var firstChannel = pack * 4;
            var delta = noisy[(firstChannel * height + y) * width + x] - input[(firstChannel * height + y) * width + x];
            for (var lane = 1; lane < 4; lane++)
            {
                var channel = firstChannel + lane;
                var laneDelta = noisy[(channel * height + y) * width + x] - input[(channel * height + y) * width + x];
                Assert.That(laneDelta, Is.EqualTo(delta).Within(1e-6f), "GFPGAN noise must broadcast across Pack4 lanes.");
            }
        }
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

    private static float[] ConcatSequenceReference(
        float[] past,
        float[] current,
        int width,
        int pastHeight,
        int currentHeight,
        int channels)
    {
        var totalHeight = pastHeight + currentHeight;
        var output = new float[width * totalHeight * channels];
        for (var channel = 0; channel < channels; channel++)
        for (var y = 0; y < totalHeight; y++)
        for (var x = 0; x < width; x++)
        {
            output[Index(width, totalHeight, 1, channels, x, y, 0, channel)] = y < pastHeight
                ? past[Index(width, pastHeight, 1, channels, x, y, 0, channel)]
                : current[Index(width, currentHeight, 1, channels, x, y - pastHeight, 0, channel)];
        }
        return output;
    }

    private static float[] ReferenceSdpaWithCache(
        float[] query,
        float[] key,
        float[] value,
        int queryLength,
        int keyLength,
        int embed,
        int heads,
        float scale,
        bool causal,
        int causalQueryOffset)
    {
        var output = new float[query.Length];
        for (var head = 0; head < heads; head++)
        for (var row = 0; row < queryLength; row++)
        {
            var scores = new float[keyLength];
            var max = float.NegativeInfinity;
            for (var col = 0; col < keyLength; col++)
            {
                var score = 0f;
                for (var i = 0; i < embed; i++)
                {
                    score += query[Index(embed, queryLength, 1, heads, i, row, 0, head)]
                        * key[Index(embed, keyLength, 1, heads, i, col, 0, head)];
                }
                score = causal && col > row + causalQueryOffset ? -1e20f : score * scale;
                scores[col] = score;
                max = Mathf.Max(max, score);
            }

            var sum = 0f;
            for (var col = 0; col < keyLength; col++)
                sum += Mathf.Exp(scores[col] - max);
            for (var outX = 0; outX < embed; outX++)
            {
                var result = 0f;
                for (var col = 0; col < keyLength; col++)
                {
                    result += Mathf.Exp(scores[col] - max) / sum
                        * value[Index(embed, keyLength, 1, heads, outX, col, 0, head)];
                }
                output[Index(embed, queryLength, 1, heads, outX, row, 0, head)] = result;
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
