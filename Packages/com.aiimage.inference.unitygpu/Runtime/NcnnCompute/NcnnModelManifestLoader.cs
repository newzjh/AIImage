using System;
using System.IO;
using AIImage.Inference.Core;
using UnityEngine;

namespace NcnnCompute
{
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

        internal static ModelManifest TryLoadFromEnvironment()
        {
            var path = Environment.GetEnvironmentVariable(ManifestEnvironmentVariable);
            return string.IsNullOrWhiteSpace(path) ? null : LoadFromFile(path);
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
