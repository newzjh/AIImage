using System;
using System.IO;
using Aexis.Samples.Json.Linq;

namespace AIImage.Qwen35
{
    public sealed class Qwen35MobileMemoryPolicy
    {
        public const long MaximumDeliveryWeightBytes = 1610612736L;
        public long WeightBytes { get; private set; }
        public bool Quantized { get; private set; }
        public bool DeliveryEligible => Quantized && WeightBytes <= MaximumDeliveryWeightBytes;

        // A compact asset package alone does not prove that the complete texture
        // runtime fits broadly on mobile. Q8 still needs a device-memory gate and
        // on-device Vulkan/Metal evidence before that claim can be made.
        public bool BroadMobileSupported => false;
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
            var deliveryEligible = quantized && bytes <= MaximumDeliveryWeightBytes;
            var reasons = deliveryEligible
                ? new[]
                {
                    "Q8 asset delivery is eligible, but broad mobile runtime support is not proven.",
                    "The runtime enforces a 12288 MiB system-memory floor until Vulkan/Metal device measurements qualify a lower tier.",
                    "Editor validation and packaged asset size must not be advertised as broad mobile availability."
                }
                : new[]
                {
                    "unquantized or oversized asset set",
                    "Android/iOS deployment requires int8 weight-only (int4 optional) artifacts",
                    "Editor FP32 validation must not be advertised as mobile availability"
                };
            return new Qwen35MobileMemoryPolicy(reasons) { WeightBytes = bytes, Quantized = quantized };
        }

        public JObject ToJson()
        {
            return new JObject
            {
                ["weight_bytes"] = WeightBytes,
                ["weight_gib"] = WeightBytes / (1024.0 * 1024.0 * 1024.0),
                ["quantized"] = Quantized,
                ["delivery_eligible"] = DeliveryEligible,
                ["broad_mobile_supported"] = BroadMobileSupported,
                ["runtime_device_memory_floor_mb"] = Qwen35DeviceCompatibility.MinimumSystemMemoryMb,
                ["unsupported_reasons"] = new JArray(UnsupportedReasons)
            };
        }
    }
}
