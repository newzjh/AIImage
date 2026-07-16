#if UNITY_EDITOR
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using AIImage.Inference.Core;
using NcnnCompute;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

[Serializable]
public sealed class NcnnInt8WeightOnlyRegressionReport
{
    public string reportVersion = "aiimage.int8-selective-regression/v1";
    public string quantizationVersion = "aiimage.int8-selective/v1";
    public string calibrationMethod = "weights-absmax-per-output-channel-plus-manifest-activation-scale";
    public string accumulation = "FP32";
    public string activationStorage = "FP16 Pack4 texture";
    public int inputSeed = 20260716;
    public string executionBackend = "deterministic-cpu-oracle-errors-plus-command-buffer-submit-readback-timing";
    public NcnnInt8WeightOnlyRegressionCase[] cases;
}

[Serializable]
public sealed class NcnnInt8WeightOnlyRegressionCase
{
    public string fixture;
    public string operatorName;
    public string int8Mode;
    public float activationScale;
    public int activationZeroPoint;
    public int outputChannels;
    public int valuesPerOutputChannel;
    public int fp32WeightBytes;
    public int fp16WeightBytes;
    public int int8WeightBytes;
    public long fp32RtBytes;
    public long fp16RtBytes;
    public long int8SelectiveRtBytes;
    public long int8WeightOnlyRtBytes;
    public NcnnInt8WeightOnlyGpuTiming gpuSubmitAndReadback;
    public NcnnInt8WeightOnlyError fp16VsFp32;
    public NcnnInt8WeightOnlyError int8SelectiveVsFp32;
    public NcnnInt8WeightOnlyError int8WeightOnlyVsFp32;
}

[Serializable]
public sealed class NcnnInt8WeightOnlyError
{
    public float maxAbsoluteError;
    public float meanAbsoluteError;
    public float rootMeanSquareError;
}

[Serializable]
public sealed class NcnnInt8WeightOnlyGpuTiming
{
    public bool available;
    public string status;
    public string measurement = "CommandBuffer submit through final output AsyncGPUReadback completion";
    public int warmupIterations;
    public int measuredIterations;
    public string graphicsDeviceName;
    public string graphicsDeviceType;
    public string graphicsDeviceVersion;
    public string unityVersion;
    public double fp32Milliseconds;
    public double fp16Milliseconds;
    public double int8SelectiveMilliseconds;
    public double int8WeightOnlyMilliseconds;
}

// Numeric errors use deterministic CPU oracle arithmetic. GPU timings use the actual
// CommandBuffer texture path and are annotated with their device/runtime metadata.
public static class NcnnInt8WeightOnlyRegressionTool
{
    private const int InputSeed = 20260716;
    private const int GpuWarmupIterations = 2;
    private const int GpuMeasuredIterations = 5;

    [MenuItem("AIImage/Inference/Export INT8 Selective Regression")]
    public static void ExportFromMenu()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var path = Path.Combine(root, "output", "int8-selective-regression.json");
        Export(path);
        UnityEngine.Debug.Log("[INT8Selective] wrote " + path);
    }

    public static void ExportFromBatch()
    {
        ExportFromMenu();
    }

    public static void Export(string outputPath)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("An output path is required.", nameof(outputPath));

        var report = Run();
        var directory = Path.GetDirectoryName(Path.GetFullPath(outputPath));
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllText(outputPath, JsonUtility.ToJson(report, true) + "\n");
    }

    public static NcnnInt8WeightOnlyRegressionReport Run()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        return new NcnnInt8WeightOnlyRegressionReport
        {
            cases = new[]
            {
                RunMattingConv(Path.Combine(root, "Assets", "StreamingAssets", "Matting")),
                RunYoloConv(Path.Combine(root, "Assets", "StreamingAssets", "Yolo")),
                RunMobileClipGemm(Path.Combine(root, "Assets", "StreamingAssets", "Clip", "mobileclip_s0_export"))
            }
        };
    }

    private static NcnnInt8WeightOnlyRegressionCase RunMattingConv(string modelDirectory)
    {
        const string layerName = "Conv_236";
        var paramText = File.ReadAllText(Path.Combine(modelDirectory, "matting.param"));
        var param = NcnnParamParser.Parse(paramText);
        var layer = param.FindByName(layerName) ?? throw new InvalidOperationException("Matting " + layerName + " was not found.");
        var outChannels = layer.GetInt(0, 0);
        var kernel = layer.GetInt(1, 0);
        var stride = layer.GetInt(3, 1);
        var pad = layer.GetInt(4, 0);
        var weightCount = layer.GetInt(6, 0);
        var inputChannels = weightCount / Math.Max(1, outChannels * kernel * kernel);
        var weights = new float[weightCount];
        var bias = new float[outChannels];
        using (var ops = new NcnnOps())
        using (var session = new NcnnRepro(ops))
        using (var stream = File.OpenRead(Path.Combine(modelDirectory, "matting.bin")))
        using (var reader = new NcnnBinReader(stream))
        {
            session.LoadModel(paramText, reader);
            if (!session._conv.TryGetValue(layerName, out var pack)
                || pack.rawWeight == null
                || pack.rawBias == null)
            {
                throw new InvalidOperationException("Matting " + layerName + " raw weights were not loaded.");
            }
            pack.rawWeight.GetData(weights);
            pack.rawBias.GetData(bias);
        }

        const int inputWidth = 32;
        const int inputHeight = 32;
        var input = CreateValues(inputWidth * inputHeight * inputChannels, InputSeed);
        var outputWidth = (inputWidth + pad * 2 - kernel) / stride + 1;
        var outputHeight = (inputHeight + pad * 2 - kernel) / stride + 1;
        var fp32 = RunConv(input, weights, bias, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outChannels, kernel, stride, pad, false, null, null, 0f);
        var fp16 = RunConv(input, weights, bias, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outChannels, kernel, stride, pad, true, null, null, 0f);
        Quantize(weights, outChannels, weightCount / outChannels, true, out var quantized, out var scales);
        const float activationScale = 0f;
        var int8 = RunConv(input, weights, bias, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outChannels, kernel, stride, pad, true, quantized, scales, activationScale);

        var result = CreateCase(
            "matting/" + layerName,
            "Convolution",
            "W8",
            activationScale,
            outChannels,
            weightCount / outChannels,
            outputWidth,
            outputHeight,
            fp32,
            fp16,
            int8);
        result.gpuSubmitAndReadback = MeasureMattingGpuTiming(
            weights,
            bias,
            input,
            inputWidth,
            inputHeight,
            inputChannels,
            outputWidth,
            outputHeight,
            outChannels,
            kernel,
            stride,
            pad,
            activationScale);
        return result;
    }

    private static NcnnInt8WeightOnlyRegressionCase RunYoloConv(string modelDirectory)
    {
        var param = NcnnParamParser.Parse(File.ReadAllText(Path.Combine(modelDirectory, "yolov8n_seg.ncnn.param")));
        var layer = param.FindByName("conv_0") ?? throw new InvalidOperationException("YOLO conv_0 was not found.");
        var outChannels = layer.GetInt(0, 0);
        var kernel = layer.GetInt(1, 0);
        var weightCount = layer.GetInt(6, 0);
        var inputChannels = weightCount / Math.Max(1, outChannels * kernel * kernel);
        using var stream = File.OpenRead(Path.Combine(modelDirectory, "yolov8n_seg.ncnn.bin"));
        using var reader = new NcnnBinReader(stream);
        var weights = NcnnRepro.ReadPackedOrRawWeightArray(reader, weightCount, layer.name);
        var bias = layer.GetInt(5, 0) != 0 ? reader.ReadFloat32Array(outChannels) : new float[outChannels];

        const int inputWidth = 64;
        const int inputHeight = 64;
        var stride = layer.GetInt(3, 1);
        var pad = layer.GetInt(4, 0);
        var input = CreateValues(inputWidth * inputHeight * inputChannels, InputSeed + 1);
        var outputWidth = (inputWidth + pad * 2 - kernel) / stride + 1;
        var outputHeight = (inputHeight + pad * 2 - kernel) / stride + 1;
        var fp32 = RunConv(input, weights, bias, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outChannels, kernel, stride, pad, false, null, null, 0f);
        var fp16 = RunConv(input, weights, bias, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outChannels, kernel, stride, pad, true, null, null, 0f);
        Quantize(weights, outChannels, weightCount / outChannels, true, out var quantized, out var scales);
        const float activationScale = 0.0078125f;
        var int8WeightOnly = RunConv(input, weights, bias, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outChannels, kernel, stride, pad, true, quantized, scales, 0f);
        var int8 = RunConv(input, weights, bias, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outChannels, kernel, stride, pad, true, quantized, scales, activationScale);

        var result = CreateCase(
            "yolo-seg/conv_0",
            "Convolution",
            "W8A8",
            activationScale,
            outChannels,
            weightCount / outChannels,
            outputWidth,
            outputHeight,
            fp32,
            fp16,
            int8,
            int8WeightOnly);
        result.gpuSubmitAndReadback = MeasureMattingGpuTiming(
            weights,
            bias,
            input,
            inputWidth,
            inputHeight,
            inputChannels,
            outputWidth,
            outputHeight,
            outChannels,
            kernel,
            stride,
            pad,
            activationScale);
        return result;
    }

    private static NcnnInt8WeightOnlyRegressionCase RunMobileClipGemm(string modelDirectory)
    {
        var paramText = File.ReadAllText(Path.Combine(modelDirectory, "image_encoder.ncnn.param"));
        using var ops = new NcnnOps();
        using var session = new NcnnRepro(ops);
        using var stream = File.OpenRead(Path.Combine(modelDirectory, "image_encoder.ncnn.bin"));
        using var reader = new NcnnBinReader(stream);
        session.LoadModel(paramText, reader);

        if (!session._gemm.TryGetValue("gemm_0", out var pack) || pack.bDataCpu == null)
            throw new InvalidOperationException("MobileCLIP gemm_0 was not loaded.");
        var weights = pack.bDataCpu;
        var bias = pack.cDataCpu != null && pack.cDataCpu.Length == pack.constantN
            ? pack.cDataCpu
            : new float[pack.constantN];
        var input = CreateValues(pack.constantK, InputSeed + 2);
        var fp32 = RunGemm(input, weights, bias, pack.constantN, pack.constantK, false, null, null, 0f);
        var fp16 = RunGemm(input, weights, bias, pack.constantN, pack.constantK, true, null, null, 0f);
        Quantize(weights, pack.constantN, pack.constantK, pack.transB, out var quantized, out var scales);
        const float activationScale = 0.03125f;
        var int8WeightOnly = RunGemm(input, weights, bias, pack.constantN, pack.constantK, true, quantized, scales, 0f);
        var int8 = RunGemm(input, weights, bias, pack.constantN, pack.constantK, true, quantized, scales, activationScale);

        var result = CreateCase(
            "mobileclip-s0/gemm_0",
            "Gemm",
            "W8A8",
            activationScale,
            pack.constantN,
            pack.constantK,
            (pack.constantN + 3) / 4,
            1,
            fp32,
            fp16,
            int8,
            int8WeightOnly,
            rtPackCountOverride: 1);
        result.gpuSubmitAndReadback = MeasureFrozenClipGpuTiming(weights, bias, input, pack.constantN, pack.constantK, activationScale);
        return result;
    }

    private static NcnnInt8WeightOnlyRegressionCase CreateCase(
        string fixture,
        string operatorName,
        string int8Mode,
        float activationScale,
        int outputChannels,
        int valuesPerOutputChannel,
        int outputWidth,
        int outputHeight,
        float[] fp32,
        float[] fp16,
        float[] int8,
        float[] int8WeightOnly = null,
        int rtPackCountOverride = 0)
    {
        int8WeightOnly ??= int8;
        var packCount = rtPackCountOverride > 0 ? rtPackCountOverride : (outputChannels + 3) / 4;
        var fp32RtBytes = (long)outputWidth * outputHeight * packCount * 16;
        var fp16RtBytes = (long)outputWidth * outputHeight * packCount * 8;
        var elements = outputChannels * valuesPerOutputChannel;
        return new NcnnInt8WeightOnlyRegressionCase
        {
            fixture = fixture,
            operatorName = operatorName,
            int8Mode = int8Mode,
            activationScale = activationScale,
            activationZeroPoint = 0,
            outputChannels = outputChannels,
            valuesPerOutputChannel = valuesPerOutputChannel,
            fp32WeightBytes = elements * sizeof(float),
            fp16WeightBytes = elements * sizeof(ushort),
            int8WeightBytes = ((elements + 3) / 4) * sizeof(uint) + outputChannels * sizeof(float),
            fp32RtBytes = fp32RtBytes,
            fp16RtBytes = fp16RtBytes,
            int8SelectiveRtBytes = fp16RtBytes,
            int8WeightOnlyRtBytes = fp16RtBytes,
            fp16VsFp32 = Compare(fp32, fp16),
            int8SelectiveVsFp32 = Compare(fp32, int8),
            int8WeightOnlyVsFp32 = Compare(fp32, int8WeightOnly)
        };
    }

    private static NcnnInt8WeightOnlyGpuTiming MeasureMattingGpuTiming(
        float[] weights,
        float[] bias,
        float[] input,
        int inputWidth,
        int inputHeight,
        int inputChannels,
        int outputWidth,
        int outputHeight,
        int outputChannels,
        int kernel,
        int stride,
        int pad,
        float activationScale)
    {
        if (!CanRunGpuTiming(out var unavailableReason))
            return CreateUnavailableGpuTiming(unavailableReason);

        NcnnRepro.Int8WeightOnlyUpload int8 = null;
        ComputeBuffer fp16 = null;
        try
        {
            var inputPacks = (inputChannels + 3) / 4;
            var outputPacks = (outputChannels + 3) / 4;
            var packedWeights = NcnnRepro.PackWeightsToO4I4K(weights, outputChannels, inputChannels, kernel, outputPacks, inputPacks);
            var packedBias = NcnnRepro.PackBiasToO4(bias, outputChannels, outputPacks);
            using var ops = new NcnnOps();
            using var repro = new NcnnRepro(ops);
            using var packedWeightBuffer = new ComputeBuffer(packedWeights.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            using var packedBiasBuffer = new ComputeBuffer(packedBias.Length, sizeof(float) * 4, ComputeBufferType.Structured);
            using var rawBiasBuffer = NewFloatBuffer(bias);
            packedWeightBuffer.SetData(packedWeights);
            packedBiasBuffer.SetData(packedBias);
            fp16 = NcnnRepro.NewFp16Vector4Buffer(packedWeights, "NcnnInt8WeightOnlyRegressionTool.MattingFp16");
            int8 = NcnnRepro.NewInt8WeightOnlyUpload(
                weights,
                outputChannels,
                weights.Length / outputChannels,
                outputChannelsAreContiguous: true,
                "NcnnInt8WeightOnlyRegressionTool.MattingInt8");
            return CreateAvailableGpuTiming(
                MeasureMattingGpuMode(repro, ops, input, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outputChannels, kernel, stride, pad, packedWeightBuffer, packedBiasBuffer, rawBiasBuffer, fp16, int8, NcnnPrecisionMode.FP32),
                MeasureMattingGpuMode(repro, ops, input, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outputChannels, kernel, stride, pad, packedWeightBuffer, packedBiasBuffer, rawBiasBuffer, fp16, int8, NcnnPrecisionMode.FP16),
                MeasureMattingGpuMode(repro, ops, input, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outputChannels, kernel, stride, pad, packedWeightBuffer, packedBiasBuffer, rawBiasBuffer, fp16, int8, NcnnPrecisionMode.INT8Selective, activationScale),
                MeasureMattingGpuMode(repro, ops, input, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outputChannels, kernel, stride, pad, packedWeightBuffer, packedBiasBuffer, rawBiasBuffer, fp16, int8, NcnnPrecisionMode.INT8WeightOnly));
        }
        catch (Exception exception)
        {
            return CreateUnavailableGpuTiming("GPU timing failed: " + exception.Message);
        }
        finally
        {
            if (int8 != null && int8.packedWeights != null)
            {
                NcnnGpuResourceTracker.ReleaseBuffer(int8.packedWeights, "NcnnInt8WeightOnlyRegressionTool.MattingInt8");
                int8.packedWeights.Dispose();
            }
            if (int8 != null && int8.scales != null)
            {
                NcnnGpuResourceTracker.ReleaseBuffer(int8.scales, "NcnnInt8WeightOnlyRegressionTool.MattingInt8");
                int8.scales.Dispose();
            }
            if (fp16 != null)
            {
                NcnnGpuResourceTracker.ReleaseBuffer(fp16, "NcnnInt8WeightOnlyRegressionTool.MattingFp16");
                fp16.Dispose();
            }
        }
    }

    private static double MeasureMattingGpuMode(
        NcnnRepro repro,
        NcnnOps ops,
        float[] input,
        int inputWidth,
        int inputHeight,
        int inputChannels,
        int outputWidth,
        int outputHeight,
        int outputChannels,
        int kernel,
        int stride,
        int pad,
        ComputeBuffer packedWeights,
        ComputeBuffer packedBias,
        ComputeBuffer rawBias,
        ComputeBuffer fp16Weights,
        NcnnRepro.Int8WeightOnlyUpload int8Weights,
        NcnnPrecisionMode precision,
        float activationScale = 0f)
    {
        var format = precision == NcnnPrecisionMode.FP32 ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
        var inputPacks = (inputChannels + 3) / 4;
        var outputPacks = (outputChannels + 3) / 4;
        var inputTexture = CreatePackedChwInputTexture(input, inputWidth, inputHeight, inputChannels, format);
        var persistentOutputs = CreatePersistentOutputs(outputPacks, outputWidth, outputHeight, format);
        try
        {
            var useInt8 = precision == NcnnPrecisionMode.INT8WeightOnly || precision == NcnnPrecisionMode.INT8Selective;
            ops.SetFp16ConvWeights(precision == NcnnPrecisionMode.FP16 ? fp16Weights : null);
            ops.SetInt8ConvWeights(useInt8 ? int8Weights.packedWeights : null, useInt8 ? int8Weights.scales : null);
            ops.SetInt8ActivationQuantization(precision == NcnnPrecisionMode.INT8Selective && activationScale > 0f
                ? new QuantizedNodePlan
                {
                    mode = QuantizedNodeMode.Int8W8A8,
                    activationScale = activationScale,
                    activationZeroPoint = 0
                }
                : null);
            return MeasureGpuSubmitAndReadback(() => DispatchMattingConv(
                repro,
                ops,
                inputTexture,
                inputPacks,
                inputWidth,
                inputHeight,
                inputChannels,
                outputWidth,
                outputHeight,
                outputChannels,
                kernel,
                stride,
                pad,
                format,
                packedWeights,
                packedBias,
                useInt8 ? int8Weights.packedWeights : packedWeights,
                rawBias,
                useInt8,
                persistentOutputs));
        }
        finally
        {
            ReleasePersistentOutputs(persistentOutputs);
            UnityEngine.Object.DestroyImmediate(inputTexture);
        }
    }

    private static NcnnInt8WeightOnlyGpuTiming MeasureFrozenClipGpuTiming(
        float[] weights,
        float[] bias,
        float[] input,
        int outputFeatures,
        int inputFeatures,
        float activationScale)
    {
        if (!CanRunGpuTiming(out var unavailableReason))
            return CreateUnavailableGpuTiming(unavailableReason);

        NcnnRepro.Int8WeightOnlyUpload int8 = null;
        ComputeBuffer fp16 = null;
        try
        {
            using var ops = new NcnnOps();
            using var repro = new NcnnRepro(ops);
            using var fp32 = NewFloatBuffer(weights);
            using var biasBuffer = NewFloatBuffer(bias);
            using var unusedWeightBinding = new ComputeBuffer(1, sizeof(float), ComputeBufferType.Structured);
            fp16 = NcnnRepro.NewFp16Buffer(weights, "NcnnInt8WeightOnlyRegressionTool.ClipFp16");
            int8 = NcnnRepro.NewInt8WeightOnlyUpload(
                weights,
                outputFeatures,
                inputFeatures,
                outputChannelsAreContiguous: true,
                "NcnnInt8WeightOnlyRegressionTool.ClipInt8");

            var fp32Ms = MeasureFrozenClipGpuMode(
                repro, ops, input, outputFeatures, inputFeatures, RenderTextureFormat.ARGBFloat,
                fp32, biasBuffer, fp16, int8, unusedWeightBinding, NcnnPrecisionMode.FP32);
            var fp16Ms = MeasureFrozenClipGpuMode(
                repro, ops, input, outputFeatures, inputFeatures, RenderTextureFormat.ARGBHalf,
                fp32, biasBuffer, fp16, int8, unusedWeightBinding, NcnnPrecisionMode.FP16);
            var int8Ms = MeasureFrozenClipGpuMode(
                repro, ops, input, outputFeatures, inputFeatures, RenderTextureFormat.ARGBHalf,
                fp32, biasBuffer, fp16, int8, unusedWeightBinding, NcnnPrecisionMode.INT8Selective, activationScale);
            var int8WeightOnlyMs = MeasureFrozenClipGpuMode(
                repro, ops, input, outputFeatures, inputFeatures, RenderTextureFormat.ARGBHalf,
                fp32, biasBuffer, fp16, int8, unusedWeightBinding, NcnnPrecisionMode.INT8WeightOnly);
            return CreateAvailableGpuTiming(fp32Ms, fp16Ms, int8Ms, int8WeightOnlyMs);
        }
        catch (Exception exception)
        {
            return CreateUnavailableGpuTiming("GPU timing failed: " + exception.Message);
        }
        finally
        {
            if (int8 != null && int8.packedWeights != null)
            {
                NcnnGpuResourceTracker.ReleaseBuffer(int8.packedWeights, "NcnnInt8WeightOnlyRegressionTool.ClipInt8");
                int8.packedWeights.Dispose();
            }
            if (int8 != null && int8.scales != null)
            {
                NcnnGpuResourceTracker.ReleaseBuffer(int8.scales, "NcnnInt8WeightOnlyRegressionTool.ClipInt8");
                int8.scales.Dispose();
            }
            if (fp16 != null)
            {
                NcnnGpuResourceTracker.ReleaseBuffer(fp16, "NcnnInt8WeightOnlyRegressionTool.ClipFp16");
                fp16.Dispose();
            }
        }
    }

    private static double MeasureFrozenClipGpuMode(
        NcnnRepro repro,
        NcnnOps ops,
        float[] input,
        int outputFeatures,
        int inputFeatures,
        RenderTextureFormat format,
        ComputeBuffer fp32Weights,
        ComputeBuffer bias,
        ComputeBuffer fp16Weights,
        NcnnRepro.Int8WeightOnlyUpload int8Weights,
        ComputeBuffer unusedWeightBinding,
        NcnnPrecisionMode precision,
        float activationScale = 0f)
    {
        var inputTexture = CreatePackedLinearInputTexture(input, format);
        var persistentOutputs = CreatePersistentOutputs(1, (outputFeatures + 3) / 4, 1, format);
        try
        {
            var useInt8 = precision == NcnnPrecisionMode.INT8WeightOnly || precision == NcnnPrecisionMode.INT8Selective;
            ops.SetFp16GemmWeights(precision == NcnnPrecisionMode.FP16 ? fp16Weights : null);
            ops.SetInt8GemmWeights(useInt8 ? int8Weights.packedWeights : null, useInt8 ? int8Weights.scales : null);
            ops.SetInt8ActivationQuantization(precision == NcnnPrecisionMode.INT8Selective
                ? new QuantizedNodePlan
                {
                    mode = QuantizedNodeMode.Int8W8A8,
                    activationScale = activationScale,
                    activationZeroPoint = 0
                }
                : null);
            return MeasureGpuSubmitAndReadback(() => DispatchFrozenClipProjection(
                repro,
                ops,
                inputTexture,
                inputFeatures,
                outputFeatures,
                format,
                useInt8 ? unusedWeightBinding : fp32Weights,
                bias,
                persistentOutputs));
        }
        finally
        {
            ReleasePersistentOutputs(persistentOutputs);
            UnityEngine.Object.DestroyImmediate(inputTexture);
        }
    }

    private static double DispatchMattingConv(
        NcnnRepro repro,
        NcnnOps ops,
        Texture2DArray input,
        int inputPacks,
        int inputWidth,
        int inputHeight,
        int inputChannels,
        int outputWidth,
        int outputHeight,
        int outputChannels,
        int kernel,
        int stride,
        int pad,
        RenderTextureFormat format,
        ComputeBuffer packedWeights,
        ComputeBuffer packedBias,
        ComputeBuffer int8WeightBinding,
        ComputeBuffer rawBias,
        bool useInt8,
        RenderTexture[] persistentOutputs)
    {
        using var commandBuffer = new CommandBuffer { name = "NcnnInt8WeightOnlyRegression.MattingConv" };
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, inputWidth, inputHeight, inputPacks, format);
            CopyTextureArrayToComputeTexture(commandBuffer, input, source, inputPacks);
            output = repro.RentTempArray(commandBuffer, outputWidth, outputHeight, (outputChannels + 3) / 4, format);
            if (useInt8)
            {
                ops.Conv2dGroupPack4(
                    commandBuffer, source, int8WeightBinding, rawBias, inputChannels, outputChannels, 1,
                    kernel, kernel, stride, stride, pad, pad, 1, 1, 0, 0f, output);
            }
            else
            {
                ops.ConvPack4General(
                    commandBuffer, source, inputPacks, packedWeights, packedBias, (outputChannels + 3) / 4, outputChannels,
                    kernel, kernel, stride, stride, pad, pad, 1, 1, 0, 0f, output);
            }
            CopyComputeTextureToPersistentOutputs(commandBuffer, output, persistentOutputs);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, source);
            output = null;
            source = null;

            var stopwatch = Stopwatch.StartNew();
            Graphics.ExecuteCommandBuffer(commandBuffer);
            WaitForReadback(persistentOutputs);
            stopwatch.Stop();
            return stopwatch.Elapsed.TotalMilliseconds;
        }
        finally
        {
            if (output != null)
                repro.ReturnTempArray(commandBuffer, output);
            if (source != null)
                repro.ReturnTempArray(commandBuffer, source);
        }
    }

    private static double DispatchFrozenClipProjection(
        NcnnRepro repro,
        NcnnOps ops,
        Texture2DArray input,
        int inputFeatures,
        int outputFeatures,
        RenderTextureFormat format,
        ComputeBuffer weights,
        ComputeBuffer bias,
        RenderTexture[] persistentOutputs)
    {
        using var commandBuffer = new CommandBuffer { name = "NcnnInt8WeightOnlyRegression.FrozenClipProjection" };
        ComputeTexture source = null;
        ComputeTexture output = null;
        try
        {
            source = repro.RentTempArray(commandBuffer, (inputFeatures + 3) / 4, 1, 1, format);
            output = repro.RentTempArray(commandBuffer, (outputFeatures + 3) / 4, 1, 1, format);
            CopyTextureArrayToComputeTexture(commandBuffer, input, source, 1);
            ops.Gemm2DPack4LinearTextureA(
                commandBuffer,
                source,
                true,
                weights,
                bias,
                m: 1,
                n: outputFeatures,
                k: inputFeatures,
                transB: true,
                alpha: 1f,
                beta: 1f,
                useC: true,
                broadcastTypeC: 4,
                output);
            CopyComputeTextureToPersistentOutputs(commandBuffer, output, persistentOutputs);
            repro.ReturnTempArray(commandBuffer, output);
            repro.ReturnTempArray(commandBuffer, source);
            output = null;
            source = null;

            var stopwatch = Stopwatch.StartNew();
            Graphics.ExecuteCommandBuffer(commandBuffer);
            WaitForReadback(persistentOutputs);
            stopwatch.Stop();
            return stopwatch.Elapsed.TotalMilliseconds;
        }
        finally
        {
            if (output != null)
                repro.ReturnTempArray(commandBuffer, output);
            if (source != null)
                repro.ReturnTempArray(commandBuffer, source);
        }
    }

    private static double MeasureGpuSubmitAndReadback(Func<double> executeOnce)
    {
        for (var index = 0; index < GpuWarmupIterations; index++)
            executeOnce();

        var total = 0d;
        for (var index = 0; index < GpuMeasuredIterations; index++)
            total += executeOnce();
        return total / GpuMeasuredIterations;
    }

    private static bool CanRunGpuTiming(out string reason)
    {
        if (!SystemInfo.supportsComputeShaders)
        {
            reason = "SystemInfo.supportsComputeShaders is false.";
            return false;
        }
        if (!SystemInfo.supportsAsyncGPUReadback)
        {
            reason = "SystemInfo.supportsAsyncGPUReadback is false.";
            return false;
        }
        reason = null;
        return true;
    }

    private static NcnnInt8WeightOnlyGpuTiming CreateAvailableGpuTiming(double fp32Ms, double fp16Ms, double int8Ms, double int8WeightOnlyMs)
    {
        return new NcnnInt8WeightOnlyGpuTiming
        {
            available = true,
            status = "ok",
            warmupIterations = GpuWarmupIterations,
            measuredIterations = GpuMeasuredIterations,
            graphicsDeviceName = SystemInfo.graphicsDeviceName,
            graphicsDeviceType = SystemInfo.graphicsDeviceType.ToString(),
            graphicsDeviceVersion = SystemInfo.graphicsDeviceVersion,
            unityVersion = Application.unityVersion,
            fp32Milliseconds = fp32Ms,
            fp16Milliseconds = fp16Ms,
            int8SelectiveMilliseconds = int8Ms,
            int8WeightOnlyMilliseconds = int8WeightOnlyMs
        };
    }

    private static NcnnInt8WeightOnlyGpuTiming CreateUnavailableGpuTiming(string reason)
    {
        return new NcnnInt8WeightOnlyGpuTiming
        {
            available = false,
            status = reason ?? "GPU timing is unavailable.",
            warmupIterations = GpuWarmupIterations,
            measuredIterations = GpuMeasuredIterations,
            graphicsDeviceName = SystemInfo.graphicsDeviceName,
            graphicsDeviceType = SystemInfo.graphicsDeviceType.ToString(),
            graphicsDeviceVersion = SystemInfo.graphicsDeviceVersion,
            unityVersion = Application.unityVersion
        };
    }

    private static ComputeBuffer NewFloatBuffer(float[] values)
    {
        var buffer = new ComputeBuffer(values.Length, sizeof(float), ComputeBufferType.Structured);
        buffer.SetData(values);
        return buffer;
    }

    private static Texture2DArray CreatePackedChwInputTexture(float[] values, int width, int height, int channels, RenderTextureFormat format)
    {
        var packCount = (channels + 3) / 4;
        var texture = new Texture2DArray(width, height, packCount, ToTextureFormat(format), false, true);
        for (var pack = 0; pack < packCount; pack++)
        {
            var pixels = new Color[width * height];
            for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
            {
                var pixel = y * width + x;
                var channel = pack * 4;
                pixels[pixel] = new Color(
                    channel < channels ? values[(channel * height + y) * width + x] : 0f,
                    channel + 1 < channels ? values[((channel + 1) * height + y) * width + x] : 0f,
                    channel + 2 < channels ? values[((channel + 2) * height + y) * width + x] : 0f,
                    channel + 3 < channels ? values[((channel + 3) * height + y) * width + x] : 0f);
            }
            texture.SetPixels(pixels, pack, 0);
        }
        texture.Apply(false, true);
        return texture;
    }

    private static Texture2DArray CreatePackedLinearInputTexture(float[] values, RenderTextureFormat format)
    {
        var packCount = (values.Length + 3) / 4;
        var texture = new Texture2DArray(packCount, 1, 1, ToTextureFormat(format), false, true);
        var pixels = new Color[packCount];
        for (var pack = 0; pack < packCount; pack++)
        {
            var index = pack * 4;
            pixels[pack] = new Color(
                index < values.Length ? values[index] : 0f,
                index + 1 < values.Length ? values[index + 1] : 0f,
                index + 2 < values.Length ? values[index + 2] : 0f,
                index + 3 < values.Length ? values[index + 3] : 0f);
        }
        texture.SetPixels(pixels, 0, 0);
        texture.Apply(false, true);
        return texture;
    }

    private static TextureFormat ToTextureFormat(RenderTextureFormat format)
    {
        return format == RenderTextureFormat.ARGBHalf ? TextureFormat.RGBAHalf : TextureFormat.RGBAFloat;
    }

    private static RenderTexture[] CreatePersistentOutputs(int count, int width, int height, RenderTextureFormat format)
    {
        var outputs = new RenderTexture[count];
        for (var index = 0; index < outputs.Length; index++)
        {
            outputs[index] = new RenderTexture(new RenderTextureDescriptor(width, height, format, 0)
            {
                dimension = TextureDimension.Tex2D,
                volumeDepth = 1,
                enableRandomWrite = true,
                msaaSamples = 1
            });
            outputs[index].Create();
        }
        return outputs;
    }

    private static void ReleasePersistentOutputs(RenderTexture[] outputs)
    {
        foreach (var output in outputs ?? Array.Empty<RenderTexture>())
        {
            if (output == null)
                continue;
            if (output.IsCreated())
                output.Release();
            UnityEngine.Object.DestroyImmediate(output);
        }
    }

    private static void CopyTextureArrayToComputeTexture(CommandBuffer commandBuffer, Texture2DArray source, ComputeTexture destination, int slices)
    {
        for (var slice = 0; slice < slices; slice++)
            commandBuffer.CopyTexture(source, slice, 0, destination.nameID, slice, 0);
    }

    private static void CopyComputeTextureToPersistentOutputs(CommandBuffer commandBuffer, ComputeTexture source, RenderTexture[] outputs)
    {
        for (var slice = 0; slice < outputs.Length; slice++)
            commandBuffer.CopyTexture(source.nameID, slice, 0, outputs[slice], 0, 0);
    }

    private static void WaitForReadback(RenderTexture[] outputs)
    {
        foreach (var output in outputs)
        {
            var request = AsyncGPUReadback.Request(output, 0);
            request.WaitForCompletion();
            if (request.hasError)
                throw new InvalidOperationException("Final output AsyncGPUReadback failed.");
        }
    }

    private static float[] RunConv(float[] input, float[] weights, float[] bias, int inW, int inH, int inC, int outW, int outH, int outC, int kernel, int stride, int pad, bool fp16, sbyte[] quantized, float[] scales, float activationScale)
    {
        var output = new float[outW * outH * outC];
        for (var outputChannel = 0; outputChannel < outC; outputChannel++)
        for (var y = 0; y < outH; y++)
        for (var x = 0; x < outW; x++)
        {
            var sum = bias[outputChannel];
            for (var inputChannel = 0; inputChannel < inC; inputChannel++)
            for (var ky = 0; ky < kernel; ky++)
            for (var kx = 0; kx < kernel; kx++)
            {
                var srcX = x * stride - pad + kx;
                var srcY = y * stride - pad + ky;
                if (srcX < 0 || srcX >= inW || srcY < 0 || srcY >= inH)
                    continue;
                var inputValue = fp16
                    ? ToHalf(input[(inputChannel * inH + srcY) * inW + srcX])
                    : input[(inputChannel * inH + srcY) * inW + srcX];
                if (activationScale > 0f)
                    inputValue = QuantizeActivation(inputValue, activationScale, 0);
                var weightIndex = ((outputChannel * inC + inputChannel) * kernel + ky) * kernel + kx;
                var weightValue = quantized == null ? weights[weightIndex] : quantized[weightIndex] * scales[outputChannel];
                sum += inputValue * (fp16 && quantized == null ? ToHalf(weightValue) : weightValue);
            }
            output[(outputChannel * outH + y) * outW + x] = sum;
        }
        return output;
    }

    private static float[] RunGemm(float[] input, float[] weights, float[] bias, int outFeatures, int inFeatures, bool fp16, sbyte[] quantized, float[] scales, float activationScale)
    {
        var output = new float[outFeatures];
        for (var outputChannel = 0; outputChannel < outFeatures; outputChannel++)
        {
            var sum = bias[outputChannel];
            var baseIndex = outputChannel * inFeatures;
            for (var inputIndex = 0; inputIndex < inFeatures; inputIndex++)
            {
                var inputValue = fp16 ? ToHalf(input[inputIndex]) : input[inputIndex];
                if (activationScale > 0f)
                    inputValue = QuantizeActivation(inputValue, activationScale, 0);
                var weightValue = quantized == null ? weights[baseIndex + inputIndex] : quantized[baseIndex + inputIndex] * scales[outputChannel];
                sum += inputValue * (fp16 && quantized == null ? ToHalf(weightValue) : weightValue);
            }
            output[outputChannel] = sum;
        }
        return output;
    }

    private static float QuantizeActivation(float value, float scale, int zeroPoint)
    {
        var quantized = Mathf.Clamp(Mathf.RoundToInt(value / scale) + zeroPoint, -128, 127);
        return (quantized - zeroPoint) * scale;
    }

    private static void Quantize(float[] source, int outputChannels, int valuesPerOutputChannel, bool contiguous, out sbyte[] quantized, out float[] scales)
    {
        quantized = new sbyte[source.Length];
        scales = new float[outputChannels];
        for (var outputChannel = 0; outputChannel < outputChannels; outputChannel++)
        {
            var maxAbs = 0f;
            for (var index = 0; index < valuesPerOutputChannel; index++)
            {
                var sourceIndex = contiguous ? outputChannel * valuesPerOutputChannel + index : index * outputChannels + outputChannel;
                maxAbs = Mathf.Max(maxAbs, Mathf.Abs(source[sourceIndex]));
            }
            scales[outputChannel] = maxAbs > 0f ? maxAbs / 127f : 1f;
            for (var index = 0; index < valuesPerOutputChannel; index++)
            {
                var sourceIndex = contiguous ? outputChannel * valuesPerOutputChannel + index : index * outputChannels + outputChannel;
                quantized[sourceIndex] = (sbyte)Mathf.Clamp(Mathf.RoundToInt(source[sourceIndex] / scales[outputChannel]), -127, 127);
            }
        }
    }

    private static NcnnInt8WeightOnlyError Compare(float[] expected, float[] actual)
    {
        var absolute = 0d;
        var squared = 0d;
        var maximum = 0f;
        for (var index = 0; index < expected.Length; index++)
        {
            var error = Mathf.Abs(expected[index] - actual[index]);
            absolute += error;
            squared += error * error;
            maximum = Mathf.Max(maximum, error);
        }
        return new NcnnInt8WeightOnlyError
        {
            maxAbsoluteError = maximum,
            meanAbsoluteError = (float)(absolute / expected.Length),
            rootMeanSquareError = (float)Math.Sqrt(squared / expected.Length)
        };
    }

    private static float[] CreateValues(int count, int seed)
    {
        var values = new float[count];
        uint state = unchecked((uint)seed);
        for (var index = 0; index < values.Length; index++)
        {
            state = state * 1664525u + 1013904223u;
            values[index] = ((state >> 8) & 0xffff) / 32767.5f - 1f;
        }
        return values;
    }

    private static float ToHalf(float value)
    {
        return NcnnNumericUtils.ToHalfRoundedFloat(value);
    }
}
#endif
