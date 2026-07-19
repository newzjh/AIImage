using System;
using System.Collections.Generic;
using System.IO;

namespace NcnnCompute
{
    public sealed class Qwen35NetworkAsset
    {
        public string Name;
        public string ParamPath;
        public string BinPath;
        public bool SharesTokenEmbedding;
    }

    public sealed class Qwen35NetworkAssetCatalog
    {
        public readonly Qwen35NetworkAsset[] Networks;
        public readonly string SharedTokenEmbeddingBin;

        private Qwen35NetworkAssetCatalog(Qwen35NetworkAsset[] networks, string sharedBin)
        {
            Networks = networks;
            SharedTokenEmbeddingBin = sharedBin;
        }

        public static Qwen35NetworkAssetCatalog Create(Qwen35ModelContract contract)
        {
            var root = contract.ModelDirectory;
            var shared = Path.Combine(root, "qwen3.5_embed_token.ncnn.bin");
            var networks = new[]
            {
                Asset(root, "embed_token", "qwen3.5_embed_token.ncnn.param", shared, false),
                Asset(root, "decoder", "qwen3.5_decoder.ncnn.param", Path.Combine(root, "qwen3.5_decoder.ncnn.bin"), false),
                Asset(root, "proj_out", "qwen3.5_proj_out.ncnn.param", shared, true),
                Asset(root, "vision_embed_patch", "qwen3.5_vision_embed_patch.ncnn.param", Path.Combine(root, "qwen3.5_vision_embed_patch.ncnn.bin"), false),
                Asset(root, "vision_embed_pos", "qwen3.5_vision_embed_pos.ncnn.param", Path.Combine(root, "qwen3.5_vision_embed_pos.ncnn.bin"), false),
                Asset(root, "vision_encoder", "qwen3.5_vision_encoder.ncnn.param", Path.Combine(root, "qwen3.5_vision_encoder.ncnn.bin"), false),
            };
            return new Qwen35NetworkAssetCatalog(networks, shared);
        }

        public bool HasExactlySixNetworks => Networks.Length == 6;
        public bool UsesSingleTokenEmbeddingBin
        {
            get
            {
                var sharedCount = 0;
                for (var i = 0; i < Networks.Length; i++) if (Networks[i].SharesTokenEmbedding) sharedCount++;
                return sharedCount == 1 && string.Equals(Networks[0].BinPath, SharedTokenEmbeddingBin, StringComparison.Ordinal) && string.Equals(Networks[2].BinPath, SharedTokenEmbeddingBin, StringComparison.Ordinal);
            }
        }

        public List<string> ValidateFiles()
        {
            var errors = new List<string>();
            for (var i = 0; i < Networks.Length; i++)
            {
                if (!File.Exists(Networks[i].ParamPath)) errors.Add("missing param: " + Networks[i].ParamPath);
                if (!File.Exists(Networks[i].BinPath)) errors.Add("missing bin: " + Networks[i].BinPath);
            }
            if (!HasExactlySixNetworks) errors.Add("Qwen3.5 requires six network assets");
            if (!UsesSingleTokenEmbeddingBin) errors.Add("token embedding/LM head must share one bin path");
            return errors;
        }

        private static Qwen35NetworkAsset Asset(string root, string name, string param, string bin, bool shared)
        {
            return new Qwen35NetworkAsset { Name = name, ParamPath = Path.Combine(root, param), BinPath = bin, SharesTokenEmbedding = shared };
        }
    }
}
