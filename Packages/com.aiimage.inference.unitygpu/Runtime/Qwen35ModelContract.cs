using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using Newtonsoft.Json.Linq;

namespace NcnnCompute
{
    public sealed class Qwen35ModelContract
    {
        public const string ModelType = "qwen3.5";
        public readonly string ModelDirectory;
        public readonly Dictionary<string, string> Files = new Dictionary<string, string>(StringComparer.Ordinal);
        public readonly List<string> Errors = new List<string>();
        public Qwen35MobileAssetSet MobileAssets { get; private set; }
        public int DecoderLayerCount { get; private set; }
        public int DecoderBlobCount { get; private set; }
        public int ShortConvCount { get; private set; }
        public int GatedDeltaRuleCount { get; private set; }
        public bool IsValid => Errors.Count == 0;

        private Qwen35ModelContract(string directory) { ModelDirectory = directory; }

        public static Qwen35ModelContract Validate(
            string directory,
            bool requireWeights = true,
            Action<Qwen35Progress> onProgress = null,
            CancellationToken cancellationToken = default)
        {
            var contract = new Qwen35ModelContract(Path.GetFullPath(directory ?? string.Empty));
            if (!Directory.Exists(contract.ModelDirectory)) { contract.Errors.Add("model directory missing: " + contract.ModelDirectory); return contract; }
            onProgress?.Invoke(new Qwen35Progress("validating_assets", "Reading mobile asset manifest", 0f));
            try
            {
                contract.MobileAssets = Qwen35MobileAssetSet.TryLoad(
                    contract.ModelDirectory,
                    verifyHashes: true,
                    (completed, total, file) =>
                    {
                        var progress = total > 0 ? (float)((double)completed / total) : 1f;
                        onProgress?.Invoke(new Qwen35Progress(
                            "validating_assets",
                            "Verifying " + file,
                            progress * 0.9f,
                            completed,
                            total));
                    },
                    cancellationToken);
            }
            catch (Exception error) { contract.Errors.Add("mobile asset manifest invalid: " + error.Message); }
            cancellationToken.ThrowIfCancellationRequested();
            onProgress?.Invoke(new Qwen35Progress("validating_contract", "Parsing model contract", 0.92f));
            var modelJsonPath = Path.Combine(contract.ModelDirectory, "model.json");
            if (!File.Exists(modelJsonPath)) { contract.Errors.Add("model.json missing"); return contract; }
            var root = JObject.Parse(File.ReadAllText(modelJsonPath));
            if (!string.Equals((string)root["type"], ModelType, StringComparison.Ordinal)) contract.Errors.Add("model.json type is not qwen3.5");
            var files = new[] { "model.json", "vocab.txt", "merges.txt", "qwen3.5_decoder.ncnn.param", "qwen3.5_decoder.ncnn.bin", "qwen3.5_embed_token.ncnn.param", "qwen3.5_embed_token.ncnn.bin", "qwen3.5_proj_out.ncnn.param", "qwen3.5_vision_embed_patch.ncnn.param", "qwen3.5_vision_embed_patch.ncnn.bin", "qwen3.5_vision_embed_pos.ncnn.param", "qwen3.5_vision_embed_pos.ncnn.bin", "qwen3.5_vision_encoder.ncnn.param", "qwen3.5_vision_encoder.ncnn.bin" };
            foreach (var file in files)
            {
                var path = Path.Combine(contract.ModelDirectory, file);
                if (file.EndsWith(".bin", StringComparison.Ordinal) && contract.MobileAssets != null && contract.MobileAssets.Contains(file))
                {
                    contract.Files[file] = contract.MobileAssets.ManifestPath;
                    continue;
                }
                if (!File.Exists(path)) { if (requireWeights || !file.EndsWith(".bin", StringComparison.Ordinal)) contract.Errors.Add("missing asset: " + file); continue; }
                contract.Files[file] = path;
                if (ExpectedSizes.TryGetValue(file, out var expectedBytes) && new FileInfo(path).Length != expectedBytes)
                    contract.Errors.Add("asset size mismatch: " + file + " expected=" + expectedBytes + " actual=" + new FileInfo(path).Length);
            }
            if (contract.Files.ContainsKey("qwen3.5_embed_token.ncnn.bin") && contract.Files.ContainsKey("qwen3.5_proj_out.ncnn.param"))
            {
                var projText = File.ReadAllText(contract.Files["qwen3.5_proj_out.ncnn.param"]);
                if (!projText.Contains("Embed", StringComparison.Ordinal) && !projText.Contains("Gemm", StringComparison.Ordinal)) contract.Errors.Add("proj_out param has no supported projection layer");
            }
            if (contract.Files.TryGetValue("qwen3.5_decoder.ncnn.param", out var decoderParam))
            {
                var model = NcnnParamParser.Parse(File.ReadAllText(decoderParam));
                contract.DecoderLayerCount = model.layers.Count;
                contract.DecoderBlobCount = model.blobCount;
                for (var i = 0; i < model.layers.Count; i++)
                {
                    if (model.layers[i].type == NcnnLayerTypes.ShortConv) contract.ShortConvCount++;
                    if (model.layers[i].type == NcnnLayerTypes.GatedDeltaRule) contract.GatedDeltaRuleCount++;
                }
                if (contract.DecoderLayerCount != 869) contract.Errors.Add("decoder layer count mismatch: " + contract.DecoderLayerCount);
                if (contract.DecoderBlobCount != 1181) contract.Errors.Add("decoder blob count mismatch: " + contract.DecoderBlobCount);
                if (contract.ShortConvCount != 18) contract.Errors.Add("ShortConv count mismatch: " + contract.ShortConvCount);
                if (contract.GatedDeltaRuleCount != 18) contract.Errors.Add("GDR count mismatch: " + contract.GatedDeltaRuleCount);
            }
            cancellationToken.ThrowIfCancellationRequested();
            onProgress?.Invoke(new Qwen35Progress("validating_contract", "Model contract ready", 1f));
            return contract;
        }

        private static readonly Dictionary<string, long> ExpectedSizes = new Dictionary<string, long>(StringComparer.Ordinal)
        {
            ["qwen3.5_decoder.ncnn.bin"] = 1992454120L,
            ["qwen3.5_embed_token.ncnn.bin"] = 1017118724L,
            ["qwen3.5_vision_embed_patch.ncnn.bin"] = 4721668L,
            ["qwen3.5_vision_embed_pos.ncnn.bin"] = 7077888L,
            ["qwen3.5_vision_encoder.ncnn.bin"] = 390572432L,
        };

        public JObject ToJson(bool includeHashes = true)
        {
            var files = new JObject();
            foreach (var kv in Files)
            {
                if (kv.Key.EndsWith(".bin", StringComparison.Ordinal) && MobileAssets != null && MobileAssets.Contains(kv.Key))
                {
                    files[kv.Key] = new JObject
                    {
                        ["path"] = MobileAssets.ManifestPath,
                        ["bytes"] = MobileAssets.GetStoredBytes(kv.Key),
                        ["mobile_q8_sharded"] = true,
                        ["hashes_verified"] = true
                    };
                    continue;
                }
                var item = new JObject { ["path"] = kv.Value, ["bytes"] = new FileInfo(kv.Value).Length };
                if (includeHashes) using (var sha = SHA256.Create()) using (var stream = File.OpenRead(kv.Value)) item["sha256"] = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty).ToLowerInvariant();
                files[kv.Key] = item;
            }
            return new JObject { ["schema"] = "qwen35.unity.contract/v1", ["model_directory"] = ModelDirectory, ["valid"] = IsValid, ["decoder_layers"] = DecoderLayerCount, ["decoder_blobs"] = DecoderBlobCount, ["shortconv"] = ShortConvCount, ["gdr"] = GatedDeltaRuleCount, ["files"] = files, ["errors"] = new JArray(Errors) };
        }
    }
}
