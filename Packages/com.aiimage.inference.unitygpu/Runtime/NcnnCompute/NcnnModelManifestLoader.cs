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
        FP16 = 2
    }

    [Serializable]
    internal sealed class NcnnModelManifestDocument
    {
        public string schemaVersion;
        public string modelId;
        public NcnnModelManifestPrecision precision;
    }

    [Serializable]
    internal sealed class NcnnModelManifestPrecision
    {
        public string activationDtype;
        public string weightDtype;
        public string sensitiveOutputDtype;
        public bool requireStrictTexturePlan = true;
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

            var manifest = new ModelManifest
            {
                schemaVersion = string.IsNullOrWhiteSpace(document.schemaVersion) ? ModelManifest.Contract : document.schemaVersion,
                modelId = document.modelId ?? string.Empty,
                precision = new ModelPrecisionContract
                {
                    activationDataType = ParseFloatingType(document.precision?.activationDtype, "activationDtype", source),
                    weightDataType = ParseFloatingType(document.precision?.weightDtype, "weightDtype", source),
                    sensitiveOutputDataType = ParseFloatingType(document.precision?.sensitiveOutputDtype, "sensitiveOutputDtype", source),
                    requireStrictTexturePlan = document.precision == null || document.precision.requireStrictTexturePlan
                }
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
                if (effectiveMode == NcnnPrecisionMode.FP16)
                {
                    throw new InferenceContractException(
                        "FP16 was requested but this runner has no verified FP16 model manifest"
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
                return manifest.precision.activationDataType == TensorDataType.Float16
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
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "clip-mobileclip-s0.fp16.model.json" : "clip-mobileclip-s0.fp32.model.json";
            else if (string.Equals(modelId, "realesrgan-x4plus", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "esrgan-realesrgan-x4plus.fp16.model.json" : "esrgan-realesrgan-x4plus.fp32.model.json";
            else if (string.Equals(modelId, "matting.ncnn", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "matting.fp16.model.json" : "matting.fp32.model.json";
            else if (string.Equals(modelId, "codeformer", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "codeformer.fp16.model.json" : "codeformer.fp32.model.json";
            else if (string.Equals(modelId, "gfpgan", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "gfpgan.fp16.model.json" : "gfpgan.fp32.model.json";
            else if (string.Equals(modelId, "wholeBrain probe", StringComparison.Ordinal))
                manifestFileName = precisionMode == NcnnPrecisionMode.FP16 ? "wholebrain-probe.fp16.model.json" : "wholebrain-probe.fp32.model.json";
            return !string.IsNullOrWhiteSpace(manifestFileName);
        }

        private static TensorDataType ParseFloatingType(string value, string field, string source)
        {
            if (string.Equals(value, "FP16", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Float16", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.Float16;
            if (string.Equals(value, "FP32", StringComparison.OrdinalIgnoreCase)
                || string.Equals(value, "Float32", StringComparison.OrdinalIgnoreCase))
                return TensorDataType.Float32;
            throw new InferenceContractException("Model manifest " + field + " must be FP16 or FP32: " + source);
        }
    }
}
