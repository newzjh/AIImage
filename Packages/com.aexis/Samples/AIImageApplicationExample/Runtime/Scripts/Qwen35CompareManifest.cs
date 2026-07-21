using System;
using System.Collections.Generic;
using System.IO;
using Aexis.Samples.Json.Linq;

namespace AIImage.Qwen35
{
    public sealed class Qwen35CompareManifest
    {
        public int TotalLayers { get; private set; }
        public int TotalCheckpoints { get; private set; }
        public readonly Dictionary<string, int> NetworkCheckpoints = new Dictionary<string, int>(StringComparer.Ordinal);
        public readonly List<string> Errors = new List<string>();
        public bool IsContractValid => Errors.Count == 0 && TotalLayers == 1156 && TotalCheckpoints == 1562;

        public static Qwen35CompareManifest Load(string path)
        {
            var result = new Qwen35CompareManifest();
            if (!File.Exists(path)) { result.Errors.Add("compare manifest missing: " + path); return result; }
            var root = JObject.Parse(File.ReadAllText(path));
            result.TotalLayers = (int?)root["total_layers"] ?? 0;
            result.TotalCheckpoints = (int?)root["total_checkpoints"] ?? 0;
            var networks = root["networks"] as JObject;
            if (networks != null)
                foreach (var property in networks.Properties())
                {
                    var checkpointCount = (int?)property.Value["checkpoint_count"] ?? 0;
                    result.NetworkCheckpoints[property.Name] = checkpointCount;
                    var layers = property.Value["layers"] as JArray;
                    if (layers == null || layers.Count == 0) result.Errors.Add("manifest network has no layers: " + property.Name);
                }
            if (result.TotalLayers != 1156) result.Errors.Add("manifest total_layers mismatch: " + result.TotalLayers);
            if (result.TotalCheckpoints != 1562) result.Errors.Add("manifest total_checkpoints mismatch: " + result.TotalCheckpoints);
            if (!result.NetworkCheckpoints.ContainsKey("decoder")) result.Errors.Add("manifest decoder network missing");
            return result;
        }

        public JObject ToJson()
        {
            var networks = new JObject();
            foreach (var kv in NetworkCheckpoints) networks[kv.Key] = kv.Value;
            return new JObject { ["total_layers"] = TotalLayers, ["total_checkpoints"] = TotalCheckpoints, ["network_checkpoints"] = networks, ["contract_valid"] = IsContractValid, ["unity_value_compare"] = "pending", ["errors"] = new JArray(Errors) };
        }
    }
}
