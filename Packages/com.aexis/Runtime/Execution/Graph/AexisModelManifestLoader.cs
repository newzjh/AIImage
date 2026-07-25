using System;
using System.IO;
using Aexis;
using UnityEngine;

namespace Aexis.Execution
{
    // Keep Auto stable as new quantized variants are added after FP16.
    public enum AexisPrecisionMode
    {
        Auto = 0,
        FP32 = 1,
        FP16 = 2,
        INT8Selective = 3,
        INT4Selective = 4,
        BF16 = 5
    }

    [Serializable]
    internal sealed class AexisModelManifestDocument
    {
        public string schemaVersion;
        public string modelId;
        public AexisModelManifestPrecision precision;
        public AexisModelManifestQuantization quantization;
        public AexisModelManifestMixedPrecision mixedPrecision;
        public AexisModelManifestPrecisionGate precisionGate;
    }

    [Serializable]
    internal sealed class AexisModelManifestPrecision
    {
        public string activationDtype;
        public string weightDtype;
        public string sensitiveOutputDtype;
        public bool requireStrictTexturePlan = true;
    }

    [Serializable]
    internal sealed class AexisModelManifestQuantization
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
        public AexisModelManifestQuantizedNodePlan[] nodePlans;
        public string[] quantizedOperators;
        public string unquantizedWeightDtype;
    }

    [Serializable]
    internal sealed class AexisModelManifestQuantizedNodePlan
    {
        public string layerName;
        public string operatorName;
        public string mode;
        public float activationScale = 1f;
        public int activationZeroPoint;
    }

    [Serializable]
    internal sealed class AexisModelManifestMixedPrecision
    {
        public string planVersion;
        public AexisModelManifestMixedPrecisionNode[] nodePlans;
        public AexisModelManifestActivationPlan[] activationPlans;
    }

    [Serializable]
    internal sealed class AexisModelManifestMixedPrecisionNode
    {
        public string layerName;
        public string operatorName;
        public string activationDtype;
        public string weightDtype;
        public string accumulationDtype;
        public float maximumAbsoluteError = float.PositiveInfinity;
        public float minimumCosineSimilarity = -1f;
    }

    [Serializable]
    internal sealed class AexisModelManifestActivationPlan
    {
        public string layerName;
        public string operatorName;
        public string packing;
        public bool dequantizeOutput = true;
        public AexisModelManifestCalibrationRange calibration;
    }

    [Serializable]
    internal sealed class AexisModelManifestCalibrationRange
    {
        public string layerName;
        public string tensorName;
        public float minimum;
        public float maximum;
        public int sampleCount;
        public string method;
    }

    [Serializable]
    internal sealed class AexisModelManifestPrecisionGate
    {
        public string gateVersion;
        public float maximumAbsoluteError = float.PositiveInfinity;
        public float maximumMeanAbsoluteError = float.PositiveInfinity;
        public float minimumCosineSimilarity = -1f;
        public AexisModelManifestPrecisionMeasurement[] baseline;
    }

    [Serializable]
    internal sealed class AexisModelManifestPrecisionMeasurement
    {
        public string outputName;
        public float maximumAbsoluteError;
        public float meanAbsoluteError;
        public float cosineSimilarity = 1f;
    }

    // Parsing stays in the Unity backend: core owns only the durable contract and does
    // not take a dependency on UnityEngine.JsonUtility or StreamingAssets APIs.
    public static class AexisModelManifestLoader
    {
        public const string ManifestEnvironmentVariable = "AIIMAGE_INFERENCE_MODEL_MANIFEST";

        // Hosts may provide an editor-only sample fallback without making the engine
        // depend on a package path, AssetDatabase, or an application assembly.
        public static Func<string, string> DefaultStreamingAssetsPathResolver { get; set; }

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

            var document = JsonUtility.FromJson<AexisModelManifestDocument>(json);
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
                    : null,
                mixedPrecision = ParseMixedPrecision(document.mixedPrecision, source),
                precisionGate = ParsePrecisionGate(document.precisionGate, source)
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

            return LoadFromFile(ResolveDefaultStreamingAssetsManifestPath(defaultManifestFileName));
        }

        public static AexisPrecisionMode ResolveAutoPrecision(string modelId)
        {
            return string.Equals(modelId, "mobileclip_s0_export", StringComparison.Ordinal)
                || IsRealEsrganX4Model(modelId)
                ? AexisPrecisionMode.FP16
                : AexisPrecisionMode.FP32;
        }

        public static ModelManifest ResolveRunnerManifest(string modelId, AexisPrecisionMode precisionMode)
        {
            // An Inspector selection is an explicit runner contract. Only Auto may inherit
            // the process-wide manifest override used by batch validation.
            if (precisionMode == AexisPrecisionMode.Auto)
            {
                var configured = TryLoadFromEnvironment();
                if (configured != null
                    && string.Equals(configured.modelId, modelId, StringComparison.OrdinalIgnoreCase))
                    return configured;
            }

            var effectiveMode = precisionMode == AexisPrecisionMode.Auto
                ? ResolveAutoPrecision(modelId)
                : precisionMode;
            if (!TryResolveManifestFileName(modelId, effectiveMode, out var manifestFileName))
            {
                if (effectiveMode == AexisPrecisionMode.FP16
                    || effectiveMode == AexisPrecisionMode.INT8Selective
                    || effectiveMode == AexisPrecisionMode.INT4Selective)
                {
                    throw new InferenceContractException(
                        effectiveMode + " was requested but this runner has no verified model manifest"
                        + " | model=" + (modelId ?? string.Empty));
                }

                // FP32 is still explicitly enforced by the session factory for runners
                // that have not yet received a packaged model manifest.
                return null;
            }

            return LoadFromFile(ResolveDefaultStreamingAssetsManifestPath(manifestFileName));
        }

        private static string ResolveDefaultStreamingAssetsManifestPath(string manifestFileName)
        {
            var relativePath = Path.Combine("InferenceManifests", manifestFileName);
            var playerPath = Path.Combine(Application.streamingAssetsPath, relativePath);
            if (File.Exists(playerPath))
                return playerPath;

            var resolver = DefaultStreamingAssetsPathResolver;
            if (resolver != null)
            {
                var resolvedPath = resolver(relativePath);
                if (!string.IsNullOrWhiteSpace(resolvedPath) && File.Exists(resolvedPath))
                    return resolvedPath;
            }

            // Preserve the player path in the error when a build omitted its required
            // manifest instead of exposing an editor-only package fallback.
            return playerPath;
        }

        public static AexisPrecisionMode ResolveAppliedPrecision(
            string modelId,
            AexisPrecisionMode requestedMode,
            ModelManifest manifest)
        {
            if (requestedMode != AexisPrecisionMode.Auto)
                return requestedMode;

            if (manifest?.precision != null)
            {
                return manifest.IsInt4WeightOnly
                    ? AexisPrecisionMode.INT4Selective
                    : manifest.IsInt8WeightOnly
                    ? AexisPrecisionMode.INT8Selective
                    : manifest.precision.activationDataType == TensorDataType.Float16
                    && manifest.precision.weightDataType == TensorDataType.Float16
                    ? AexisPrecisionMode.FP16
                    : manifest.precision.activationDataType == TensorDataType.BFloat16
                    && manifest.precision.weightDataType == TensorDataType.BFloat16
                    ? AexisPrecisionMode.BF16
                    : AexisPrecisionMode.FP32;
            }

            return ResolveAutoPrecision(modelId);
        }

        private static bool TryResolveManifestFileName(string modelId, AexisPrecisionMode precisionMode, out string manifestFileName)
        {
            manifestFileName = null;
            if (string.Equals(modelId, "mobileclip_s0_export", StringComparison.Ordinal))
                manifestFileName = precisionMode == AexisPrecisionMode.INT4Selective ? "clip-mobileclip-s0.int4.model.json" : precisionMode == AexisPrecisionMode.INT8Selective ? "clip-mobileclip-s0.int8.model.json" : precisionMode == AexisPrecisionMode.FP16 ? "clip-mobileclip-s0.fp16.model.json" : "clip-mobileclip-s0.fp32.model.json";
            else if (IsRealEsrganX4Model(modelId))
                manifestFileName = precisionMode == AexisPrecisionMode.FP16 ? "esrgan-realesrgan-x4plus.fp16.model.json" : "esrgan-realesrgan-x4plus.fp32.model.json";
            else if (string.Equals(modelId, "matting.ncnn", StringComparison.Ordinal))
                manifestFileName = precisionMode == AexisPrecisionMode.INT4Selective ? "matting.int4.model.json" : precisionMode == AexisPrecisionMode.INT8Selective ? "matting.int8.model.json" : precisionMode == AexisPrecisionMode.FP16 ? "matting.fp16.model.json" : "matting.fp32.model.json";
            else if (string.Equals(modelId, "codeformer", StringComparison.Ordinal))
                manifestFileName = precisionMode == AexisPrecisionMode.FP16 ? "codeformer.fp16.model.json" : "codeformer.fp32.model.json";
            else if (string.Equals(modelId, "gfpgan", StringComparison.Ordinal))
                manifestFileName = precisionMode == AexisPrecisionMode.FP16 ? "gfpgan.fp16.model.json" : "gfpgan.fp32.model.json";
            else if (string.Equals(modelId, "yolo-seg", StringComparison.Ordinal))
                manifestFileName = precisionMode == AexisPrecisionMode.INT4Selective ? "yolo-seg.int4.model.json" : precisionMode == AexisPrecisionMode.INT8Selective ? "yolo-seg.int8.model.json" : precisionMode == AexisPrecisionMode.FP16 ? "yolo-seg.fp16.model.json" : null;
            else if (string.Equals(modelId, "sd-inpainting", StringComparison.Ordinal)
                && precisionMode == AexisPrecisionMode.FP16)
                manifestFileName = "sd-inpainting.fp16.model.json";
            else if (string.Equals(modelId, "wholeBrain probe", StringComparison.Ordinal))
                manifestFileName = precisionMode == AexisPrecisionMode.FP16 ? "wholebrain-probe.fp16.model.json" : "wholebrain-probe.fp32.model.json";
            return !string.IsNullOrWhiteSpace(manifestFileName);
        }

        private static bool IsRealEsrganX4Model(string modelId)
        {
            return string.Equals(modelId, "realesrgan-x4plus", StringComparison.Ordinal)
                || string.Equals(modelId, "realesrgan-x4plus-anime", StringComparison.Ordinal);
        }

        private static TensorDataType ParseActivationType(string value, string field, string source)
        {
            if (string.Equals(value, "FP16", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Float16", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.Float16;
            if (string.Equals(value, "BF16", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "BFloat16", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.BFloat16;
            if (string.Equals(value, "FP32", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Float32", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.Float32;
            throw new InferenceContractException("Model manifest " + field + " must be FP16, BF16, or FP32: " + source);
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

        private static ModelMixedPrecisionContract ParseMixedPrecision(AexisModelManifestMixedPrecision document, string source)
        {
            // JsonUtility can materialize a nested serializable object even when an
            // older manifest did not contain this P1 field. Treat that completely
            // empty shell as absent so existing FP32/FP16 manifests remain valid.
            if (document == null
                || (string.IsNullOrWhiteSpace(document.planVersion)
                    && document.nodePlans == null
                    && document.activationPlans == null))
                return null;

            var sourceNodes = document.nodePlans ?? Array.Empty<AexisModelManifestMixedPrecisionNode>();
            var nodePlans = new MixedPrecisionNodePlan[sourceNodes.Length];
            for (var index = 0; index < sourceNodes.Length; index++)
            {
                var node = sourceNodes[index] ?? throw new InferenceContractException("Mixed precision node plan is null: " + source);
                nodePlans[index] = new MixedPrecisionNodePlan
                {
                    layerName = node.layerName ?? string.Empty,
                    operatorName = node.operatorName ?? string.Empty,
                    activationDataType = ParseActivationType(node.activationDtype, "mixedPrecision.nodePlans.activationDtype", source),
                    weightDataType = ParseWeightType(node.weightDtype, "mixedPrecision.nodePlans.weightDtype", source),
                    accumulationDataType = ParseActivationType(node.accumulationDtype, "mixedPrecision.nodePlans.accumulationDtype", source),
                    maximumAbsoluteError = node.maximumAbsoluteError,
                    minimumCosineSimilarity = node.minimumCosineSimilarity
                };
            }

            var sourceActivations = document.activationPlans ?? Array.Empty<AexisModelManifestActivationPlan>();
            var activationPlans = new QuantizedActivationPlan[sourceActivations.Length];
            for (var index = 0; index < sourceActivations.Length; index++)
            {
                var plan = sourceActivations[index] ?? throw new InferenceContractException("Quantized activation plan is null: " + source);
                var calibration = plan.calibration ?? throw new InferenceContractException("Quantized activation calibration is missing: " + source);
                activationPlans[index] = new QuantizedActivationPlan
                {
                    layerName = plan.layerName ?? string.Empty,
                    operatorName = plan.operatorName ?? string.Empty,
                    packing = ParseActivationPacking(plan.packing, source),
                    dequantizeOutput = plan.dequantizeOutput,
                    calibration = new ActivationCalibrationRange
                    {
                        layerName = calibration.layerName ?? plan.layerName ?? string.Empty,
                        tensorName = calibration.tensorName ?? string.Empty,
                        minimum = calibration.minimum,
                        maximum = calibration.maximum,
                        sampleCount = calibration.sampleCount,
                        method = ParseCalibrationMethod(calibration.method, source)
                    }
                };
            }

            return new ModelMixedPrecisionContract
            {
                planVersion = document.planVersion ?? string.Empty,
                nodePlans = nodePlans,
                activationPlans = activationPlans
            };
        }

        private static ModelPrecisionGateContract ParsePrecisionGate(AexisModelManifestPrecisionGate document, string source)
        {
            // Keep pre-P1 manifests forward-compatible for the same JsonUtility
            // nested-object behavior handled by ParseMixedPrecision above.
            if (document == null
                || (string.IsNullOrWhiteSpace(document.gateVersion)
                    && document.baseline == null))
                return null;
            var sourceBaseline = document.baseline ?? Array.Empty<AexisModelManifestPrecisionMeasurement>();
            var baseline = new PrecisionGateMeasurement[sourceBaseline.Length];
            for (var index = 0; index < sourceBaseline.Length; index++)
            {
                var measurement = sourceBaseline[index] ?? throw new InferenceContractException("Precision gate baseline measurement is null: " + source);
                baseline[index] = new PrecisionGateMeasurement
                {
                    outputName = measurement.outputName ?? string.Empty,
                    maximumAbsoluteError = measurement.maximumAbsoluteError,
                    meanAbsoluteError = measurement.meanAbsoluteError,
                    cosineSimilarity = measurement.cosineSimilarity
                };
            }
            return new ModelPrecisionGateContract
            {
                gateVersion = document.gateVersion ?? string.Empty,
                maximumAbsoluteError = document.maximumAbsoluteError,
                maximumMeanAbsoluteError = document.maximumMeanAbsoluteError,
                minimumCosineSimilarity = document.minimumCosineSimilarity,
                baseline = baseline
            };
        }

        private static ActivationQuantizationPacking ParseActivationPacking(string value, string source)
        {
            if (string.Equals(value, "PACK4_INT8", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Pack4SignedInt8", StringComparison.OrdinalIgnoreCase))
                return ActivationQuantizationPacking.Pack4SignedInt8;
            if (string.Equals(value, "PACK4_UINT8", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Pack4UnsignedInt8", StringComparison.OrdinalIgnoreCase))
                return ActivationQuantizationPacking.Pack4UnsignedInt8;
            throw new InferenceContractException("Unsupported INT8 activation packing " + (value ?? string.Empty) + ": " + source);
        }

        private static CalibrationMethod ParseCalibrationMethod(string value, string source)
        {
            if (string.Equals(value, "minmax", StringComparison.OrdinalIgnoreCase))
                return CalibrationMethod.MinMax;
            if (string.Equals(value, "percentile", StringComparison.OrdinalIgnoreCase))
                return CalibrationMethod.Percentile;
            if (string.Equals(value, "entropy", StringComparison.OrdinalIgnoreCase))
                return CalibrationMethod.Entropy;
            throw new InferenceContractException("Unsupported calibration method " + (value ?? string.Empty) + ": " + source);
        }

        private static ModelQuantizationContract ParseQuantization(AexisModelManifestQuantization document, string source)
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

        private static QuantizedNodePlan[] ParseNodePlans(AexisModelManifestQuantizedNodePlan[] document, string source)
        {
            var sourcePlans = document ?? Array.Empty<AexisModelManifestQuantizedNodePlan>();
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
