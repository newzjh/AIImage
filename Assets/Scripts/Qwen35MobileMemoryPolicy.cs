using System;
using System.IO;
using Newtonsoft.Json.Linq;

namespace AIImage.Qwen35
{
    public sealed class Qwen35MobileMemoryPolicy
    {
        public long WeightBytes { get; private set; }
        public bool Quantized { get; private set; }
        public bool BroadMobileSupported => Quantized && WeightBytes <= 1610612736L;
        public readonly string[] UnsupportedReasons;

        private Qwen35MobileMemoryPolicy(string[] reasons) { UnsupportedReasons = reasons; }

        public static Qwen35MobileMemoryPolicy Evaluate(Qwen35ModelContract contract, string quantizationManifestPath = null)
        {
            long bytes = contract.MobileAssets != null ? contract.MobileAssets.StoredWeightBytes : 0;
            if (contract.MobileAssets == null)
            {
                foreach (var pair in contract.Files)
                    if (pair.Key.EndsWith(".bin", StringComparison.OrdinalIgnoreCase)) bytes += new FileInfo(pair.Value).Length;
            }
            var quantized = contract.MobileAssets != null && contract.MobileAssets.WeightOnly;
            if (!string.IsNullOrWhiteSpace(quantizationManifestPath) && File.Exists(quantizationManifestPath))
                quantized = (bool?)JObject.Parse(File.ReadAllText(quantizationManifestPath))["weight_only"] ?? false;
            var reasons = quantized && bytes <= 1610612736L
                ? Array.Empty<string>()
                : new[] { "unquantized or oversized asset set", "Android/iOS deployment requires int8 weight-only (int4 optional) artifacts", "Editor FP32 validation must not be advertised as mobile availability" };
            return new Qwen35MobileMemoryPolicy(reasons) { WeightBytes = bytes, Quantized = quantized };
        }

        public JObject ToJson()
        {
            return new JObject { ["weight_bytes"] = WeightBytes, ["weight_gib"] = WeightBytes / (1024.0 * 1024.0 * 1024.0), ["quantized"] = Quantized, ["broad_mobile_supported"] = BroadMobileSupported, ["unsupported_reasons"] = new JArray(UnsupportedReasons) };
        }
    }
}
