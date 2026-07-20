using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Aexis.Ncnn;
using UnityEngine;

namespace AIImage.Qwen35
{
    public sealed class Qwen35NetworkLoadResult
    {
        public string Name;
        public int LayerCount;
        public int BlobCount;
        public long WeightBytes;
        public long ElapsedMilliseconds;
        public string Error;
        public bool Success => string.IsNullOrEmpty(Error);
    }

    /// Loads one Qwen ncnn graph through the existing managed NCNN runtime.
    /// The loader intentionally scopes each graph to a using block so the
    /// 1 GiB token matrix is never duplicated during asset validation.
    public sealed class Qwen35NetworkLoader : IDisposable
    {
        private readonly NcnnOps _ops;
        private bool _ownsOps;

        public Qwen35NetworkLoader(NcnnOps ops = null)
        {
            _ops = ops ?? new NcnnOps();
            _ownsOps = ops == null;
        }

        public Qwen35NetworkLoadResult LoadAndRelease(Qwen35NetworkAsset asset, Action<NcnnGraphSession.LoadProgress> progress = null)
        {
            if (asset == null) throw new ArgumentNullException(nameof(asset));
            var result = new Qwen35NetworkLoadResult { Name = asset.Name };
            var watch = Stopwatch.StartNew();
            try
            {
                var paramText = File.ReadAllText(asset.ParamPath);
                using (var stream = asset.OpenBinRead())
                using (var reader = new NcnnBinReader(stream))
                using (var repro = new NcnnGraphSession(_ops))
                {
                    Qwen35ModelAssetResolver.ApplyMobilePrecisionManifest(repro, asset.ModelDirectory);
                    Qwen35SharedTokenEmbeddingWeights shared = null;
                    try
                    {
                        if (string.Equals(asset.LogicalBinName, "qwen3.5_embed_token.ncnn.bin", StringComparison.Ordinal))
                        {
                            shared = Qwen35SharedTokenEmbeddingWeights.LoadModelAsset(asset.ModelDirectory);
                            shared.Attach(repro);
                        }
                        repro.LoadModel(paramText, reader, progress);
                    }
                    finally
                    {
                        shared?.Dispose();
                    }
                    result.LayerCount = repro.Model != null ? repro.Model.layers.Count : 0;
                    result.BlobCount = repro.Model != null ? repro.Model.blobCount : 0;
                    result.WeightBytes = asset.StoredWeightBytes;
                }
            }
            catch (Exception error)
            {
                result.Error = error.ToString();
            }
            finally
            {
                watch.Stop();
                result.ElapsedMilliseconds = watch.ElapsedMilliseconds;
            }
            return result;
        }

        public IReadOnlyList<Qwen35NetworkLoadResult> ValidateAllSequential(Qwen35NetworkAssetCatalog catalog, Action<string, NcnnGraphSession.LoadProgress> progress = null)
        {
            if (catalog == null) throw new ArgumentNullException(nameof(catalog));
            var results = new List<Qwen35NetworkLoadResult>(catalog.Networks.Length);
            for (var i = 0; i < catalog.Networks.Length; i++)
            {
                var asset = catalog.Networks[i];
                results.Add(LoadAndRelease(asset, p => progress?.Invoke(asset.Name, p)));
            }
            return results;
        }

        public void Dispose()
        {
            if (!_ownsOps) return;
            try { _ops.Dispose(); } catch { }
            _ownsOps = false;
        }
    }
}
