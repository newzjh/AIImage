using System;
using System.IO;
using AIImage.Inference.Core;
using UnityEngine;

namespace NcnnCompute
{
    // Keep Auto stable as new quantized variants are added after FP16.
    public enum NcnnPrecisionMode
    {
        Auto = 0,
        FP32 = 1,
        FP16 = 2,
        INT8Selective = 3,
        INT4Selective = 4
    }

    [Serializable]
    internal sealed class NcnnModelManifestDocument
    {
        public string schemaVersion;
        public string modelId;
        public NcnnModelManifestPrecision precision;
        public NcnnModelManifestQuantization quantization;
    }

    [Serializable]
    internal sealed class NcnnModelManifestPrecision
    {
        public string activationDtype;
        public string weightDtype;
        public string sensitiveOutputDtype;
        public bool requireStrictTexturePlan = true;
    }

    [Serializable]
    internal sealed class NcnnModelManifestQuantization
    {
        public string quantizationVersion;
        public string calibrationVersion;
        public string calibrationMethod;
        public string weightScheme;
        public int outputChannelAxis;
        public bool symmetric = true;
        public int zeroPoint;
        public string accumulationDtype;
        public bool activationQuantized;
        public NcnnModelManifestQuantizedNodePlan[] nodePlans;
        public string[] quantizedOperators;
        public string unquantizedWeightDtype;
    }

    [Serializable]
    internal sealed class NcnnModelManifestQuantizedNodePlan
    {
        public string layerName;
        public string operatorName;
        public string mode;
        public float activationScale = 1f;
        public int activationZeroPoint;
    }

    // Parsing stays in the Unity backend: core owns only the durable contract and does
    // not take a dependency on UnityEngine.JsonUtility or StreamingAssets APIs.
    public static class NcnnModelManifestLoader
    {
        public const string ManifestEnvironmentVariable = "AIIMAGE_INFERENCE_MODEL_MANIFEST";

        public static ModelManifest LoadFromFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("A model manifest path is required.", nameof(path));
            var fullPath = Path.GetFullPath(path);
            if (!File.Exists(fullPath))
                throw new FileNotFoundException("Model manifest was not found.", fullPath);
            return LoadFromJson(File.ReadAllText(fullPath), fullPath);
        }

        public static ModelManifest LoadFromJson(string json, string source = "manifest")
        {
            if (string.IsNullOrWhiteSpace(json))
                throw new InferenceContractException("Model manifest is empty: " + source);

            var document = JsonUtility.FromJson<NcnnModelManifestDocument>(json);
            if (document == null)
                throw new InferenceContractException("Model manifest could not be parsed: " + source);

            var weightDataType = ParseWeightType(document.precision?.weightDtype, "weightDtype", source);
            var manifest = new ModelManifest
            {
                schemaVersion = string.IsNullOrWhiteSpace(document.schemaVersion) ? ModelManifest.Contract : document.schemaVersion,
                modelId = document.modelId ?? string.Empty,
                precision = new ModelPrecisionContract
                {
                    activationDataType = ParseActivationType(document.precision?.activationDtype, "activationDtype", source),
                    weightDataType = weightDataType,
                    sensitiveOutputDataType = ParseActivationType(document.precision?.sensitiveOutputDtype, "sensitiveOutputDtype", source),
                    requireStrictTexturePlan = document.precision == null || document.precision.requireStrictTexturePlan
                },
                // JsonUtility may materialize a nested serializable object even when the
                // JSON omitted it. FP32/FP16 manifests therefore ignore that default
                // object; only quantized weights opt into the D2 quantization contract.
                quantization = weightDataType == TensorDataType.Int8
                    || weightDataType == TensorDataType.Int4
                    ? ParseQuantization(document.quantization, source)
                    : null
            };
            manifest.Validate();
            return manifest;
        }

        public static ModelManifest TryLoadFromEnvironment()
        {
            var path = Environment.GetEnvironmentVariable(ManifestEnvironmentVariable);
            return string.IsNullOrWhiteSpace(path) ? null : LoadFromFile(path);
        }

        // Explicit process configuration wins; runners use this only for a model-specific
        // shipping default when no manifest was selected by the host.
        public static ModelManifest LoadFromEnvironmentOrStreamingAssets(string defaultManifestFileName)
        {
            var configured = TryLoadFromEnvironment();
            if (configured != null)
                return configured;
            if (string.IsNullOrWhiteSpace(defaultManifestFileName))
                throw new ArgumentException("A default manifest file name is required.", nameof(defaultManifestFileName));

            var path = Path.Combine(Application.streamingAssetsPath, "InferenceManifests", defaultManifestFileName);
            return LoadFromFile(path);
        }

        public static NcnnPrecisionMode ResolveAutoPrecision(string modelId)
        {
            return string.Equals(modelId, "mobileclip_s0_export", StringComparison.Ordinal)
                || string.Equals(modelId, "realesrgan-x4plus", StringComparison.Ordinal)
                ? NcnnPrecisionMode.FP16
                : NcnnPrecisionMode.FP32;
        }

        public static ModelManifest ResolveRunnerManifest(string modelId, NcnnPrecisionMode precisionMode)
        {
            // An Inspector selection is an explicit runner contract. Only Auto may inherit
            // the process-wide manifest override used by batch validation.
            if (precisionMode == NcnnPrecisionMode.Auto)
            {
                var configured = TryLoadFromEnvironment();
                if (configured != null
                    && string.Equals(configured.modelId, modelId, StringComparison.OrdinalIgnoreCase))
                    return configured;
            }

            var effectiveMode = precisionMode == NcnnPrecisionMode.Auto
                ? ResolveAutoPrecision(modelId)
                : precisionMode;
            if (!TryResolveManifestFileName(modelId, effectiveMode, out var manifestFileName))
            {
                if (effectiveMode == NcnnPrecisionMode.FP16
                    || effectiveMode == NcnnPrecisionMode.INT8Selective
                    || effectiveMode == NcnnPrecisionMode.INT4Selective)
                {
                    throw new InferenceContractException(
                        effectiveMode + " was requested but this runner has no verified model manifest"
                        + " | model=" + (modelId ?? string.Empty));
                }

                // FP32 is still explicitly enforced by the session factory for runners
                // that have not yet received a packaged model manifest.
                return null;
            }

            return LoadFromFile(Path.Combine(Application.streamingAssetsPath, "InferenceManifests", manifestFileName));
        }

        public static NcnnPrecisionMode ResolveAppliedPrecision(
            string modelId,
            NcnnPrecisionMode requestedMode,
            ModelManifest manifest)
        {
            if (requestedMode != NcnnPrecisionMode.Auto)
                return requestedMode;

            if (manifest?.precision != null)
            {
                return manifest.IsInt4WeightOnly
                    ? NcnnPrecisionMode.INT4Selective
                    : manifest.IsInt8WeightOnly
                    ? NcnnPrecisionMode.INT8Selective
                    : manifest.precision.activationDataType == TensorDataType.Float16
                    && manifest.precision.weightDataType == TensorDataType.Float16
                    ? NcnnPrecisionMode.FP16
                    : NcnnPrecisionMode.FP32;
            }

            return ResolveAutoPrecision(modelId);
        }

        private static bool TryResolveManifestFileName(string modelId, NcnnPrecisionMode precisionMode, out string manifestFileName)
        {
            manifestFileName = null;
            if (string.Equals(modelId, "mobileclip_s0_export", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.INT4Selective ? "clip-mobileclip-s0.int4.model.json" : precisionMode == NcnnPrecisionMode.INT8Selective ? "clip-mobileclip-s0.int8.model.json" : precisionMode == NcnnPrecisionMode.FP16 ? "clip-mobileclip-s0.fp16.model.json" : "clip-mobileclip-s0.fp32.model.json";
            else if (string.Equals(modelId, "realesrgan-x4plus", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "esrgan-realesrgan-x4plus.fp16.model.json" : "esrgan-realesrgan-x4plus.fp32.model.json";
            else if (string.Equals(modelId, "matting.ncnn", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.INT4Selective ? "matting.int4.model.json" : precisionMode == NcnnPrecisionMode.INT8Selective ? "matting.int8.model.json" : precisionMode == NcnnPrecisionMode.FP16 ? "matting.fp16.model.json" : "matting.fp32.model.json";
            else if (string.Equals(modelId, "codeformer", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "codeformer.fp16.model.json" : "codeformer.fp32.model.json";
            else if (string.Equals(modelId, "gfpgan", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "gfpgan.fp16.model.json" : "gfpgan.fp32.model.json";
            else if (string.Equals(modelId, "yolo-seg", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.INT4Selective ? "yolo-seg.int4.model.json" : precisionMode == NcnnPrecisionMode.INT8Selective ? "yolo-seg.int8.model.json" : precisionMode == NcnnPrecisionMode.FP16 ? "yolo-seg.fp16.model.json" : null;
            else if (string.Equals(modelId, "sd-inpainting", StringComparison.Ordinal)
                && precisionMode == NcnnPrecisionMode.FP16)
                manifestFileName = "sd-inpainting.fp16.model.json";
            else if (string.Equals(modelId, "wholeBrain probe", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "wholebrain-probe.fp16.model.json" : "wholebrain-probe.fp32.model.json";
            return !string.IsNullOrWhiteSpace(manifestFileName);
        }

        private static TensorDataType ParseActivationType(string value, string field, string source)
        {
            if (string.Equals(value, "FP16", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Float16", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.Float16;
            if (string.Equals(value, "FP32", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Float32", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.Float32;
            throw new InferenceContractException("Model manifest " + field + " must be FP16 or FP32: " + source);
        }

        private static TensorDataType ParseWeightType(string value, string field, string source)
        {
            if (string.Equals(value, "INT8", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Int8", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.Int8;
            if (string.Equals(value, "INT4", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Int4", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.Int4;
            return ParseActivationType(value, field, source);
        }

        private static ModelQuantizationContract ParseQuantization(NcnnModelManifestQuantization document, string source)
        {
            if (document == null)
                return null;

            var isInt8 = string.Equals(document.weightScheme, "INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC", StringComparison.OrdinalIgnoreCase);
            var isInt4 = string.Equals(document.weightScheme, "INT4_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC", StringComparison.OrdinalIgnoreCase);
            if (!isInt8 && !isInt4)
            {
                throw new InferenceContractException(
                    "Model manifest quantization.weightScheme must be INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC or INT4_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC: " + source);
            }

            return new ModelQuantizationContract
            {
                quantizationVersion = document.quantizationVersion ?? string.Empty,
                calibrationVersion = document.calibrationVersion ?? string.Empty,
                calibrationMethod = document.calibrationMethod ?? string.Empty,
                weightScheme = isInt4
                    ? WeightQuantizationScheme.Int4WeightOnlyPerOutputChannelSymmetric
                    : WeightQuantizationScheme.Int8WeightOnlyPerOutputChannelSymmetric,
                outputChannelAxis = document.outputChannelAxis,
                symmetric = document.symmetric,
                zeroPoint = document.zeroPoint,
                accumulationDataType = ParseAccumulationType(document.accumulationDtype, source),
                activationQuantized = document.activationQuantized,
                nodePlans = ParseNodePlans(document.nodePlans, source),
                quantizedOperators = document.quantizedOperators ?? Array.Empty<string>(),
                unquantizedWeightDataType = string.IsNullOrWhiteSpace(document.unquantizedWeightDtype)
                    ? TensorDataType.Float32
                    : ParseActivationType(document.unquantizedWeightDtype, "unquantizedWeightDtype", source)
            };
        }

        private static QuantizedNodePlan[] ParseNodePlans(NcnnModelManifestQuantizedNodePlan[] document, string source)
        {
            var sourcePlans = document ?? Array.Empty<NcnnModelManifestQuantizedNodePlan>();
            var plans = new QuantizedNodePlan[sourcePlans.Length];
            for (var index = 0; index < sourcePlans.Length; index++)
            {
                var plan = sourcePlans[index] ?? throw new InferenceContractException("Quantization node plan is null: " + source);
                plans[index] = new QuantizedNodePlan
                {
                    layerName = plan.layerName ?? string.Empty,
                    operatorName = plan.operatorName ?? string.Empty,
                    mode = ParseNodeMode(plan.mode, source),
                    activationScale = plan.activationScale,
                    activationZeroPoint = plan.activationZeroPoint
                };
            }
            return plans;
        }

        private static QuantizedNodeMode ParseNodeMode(string value, string source)
        {
            if (string.Equals(value, "W8A8", StringComparison.OrdinalIgnoreCase))
                return QuantizedNodeMode.Int8W8A8;
            if (string.Equals(value, "W8", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "WeightOnly", StringComparison.OrdinalIgnoreCase))
                return QuantizedNodeMode.Int8WeightOnly;
            if (string.Equals(value, "W4", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "INT4", StringComparison.OrdinalIgnoreCase))
                return QuantizedNodeMode.Int4WeightOnly;
            if (string.Equals(value, "Float", StringComparison.OrdinalIgnoreCase))
                return QuantizedNodeMode.Float;
            throw new InferenceContractException("Unsupported quantization node mode " + (value ?? string.Empty) + ": " + source);
        }

        private static TensorDataType ParseAccumulationType(string value, string source)
        {
            if (string.Equals(value, "FP32", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Float32", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.Float32;
            throw new InferenceContractException("Model manifest quantization.accumulationDtype must be FP32: " + source);
        }
    }
}
