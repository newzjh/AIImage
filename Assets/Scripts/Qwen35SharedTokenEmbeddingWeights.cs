using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;

namespace AIImage.Qwen35
{
    public sealed class Qwen35SharedTokenEmbeddingWeights : IDisposable
    {
        public const int ExpectedElementCount = 254279680;

        public ComputeBuffer Buffer { get; private set; }
        public ComputeBuffer Int8PackedBuffer { get; private set; }
        public ComputeBuffer Int8ScalesBuffer { get; private set; }
        public int ElementCount { get; private set; }
        public bool IsInt8 => Int8PackedBuffer != null;
        public long ByteCount => IsInt8
            ? (long)Int8PackedBuffer.count * sizeof(uint) + (long)Int8ScalesBuffer.count * sizeof(float)
            : (long)ElementCount * sizeof(float);
        public long LoadMilliseconds { get; private set; }

        private Qwen35SharedTokenEmbeddingWeights()
        {
        }

        public static Qwen35SharedTokenEmbeddingWeights Load(string binPath, int expectedElementCount = ExpectedElementCount)
        {
            if (string.IsNullOrWhiteSpace(binPath))
                throw new ArgumentException("Token embedding bin path is empty.", nameof(binPath));
            if (!File.Exists(binPath))
                throw new FileNotFoundException("Token embedding bin was not found.", binPath);
            if (expectedElementCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(expectedElementCount));

            using (var stream = File.OpenRead(binPath))
                return Load(stream, expectedElementCount);
        }

        public static Qwen35SharedTokenEmbeddingWeights LoadModelAsset(
            string modelDirectory,
            string logicalName = "qwen3.5_embed_token.ncnn.bin",
            int expectedElementCount = ExpectedElementCount)
        {
            if (string.IsNullOrWhiteSpace(modelDirectory)) throw new ArgumentException("Model directory is empty.", nameof(modelDirectory));
            if (string.IsNullOrWhiteSpace(logicalName)) throw new ArgumentException("Logical token embedding asset name is empty.", nameof(logicalName));
            using (var stream = Qwen35ModelAssetResolver.OpenBin(modelDirectory, logicalName))
                return Load(stream, expectedElementCount);
        }

        public static async UniTask<Qwen35SharedTokenEmbeddingWeights> LoadModelAssetAsync(
            string modelDirectory,
            string logicalName = "qwen3.5_embed_token.ncnn.bin",
            int expectedElementCount = ExpectedElementCount,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(modelDirectory)) throw new ArgumentException("Model directory is empty.", nameof(modelDirectory));
            if (string.IsNullOrWhiteSpace(logicalName)) throw new ArgumentException("Logical token embedding asset name is empty.", nameof(logicalName));
            cancellationToken.ThrowIfCancellationRequested();
            var stopwatch = Stopwatch.StartNew();
            var payload = await UniTask.RunOnThreadPool(
                () =>
                {
                    using (var stream = Qwen35ModelAssetResolver.OpenBin(modelDirectory, logicalName))
                        return ReadCpuPayload(stream, expectedElementCount);
                },
                cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            var result = Upload(payload);
            stopwatch.Stop();
            result.LoadMilliseconds = stopwatch.ElapsedMilliseconds;
            return result;
        }

        private static Qwen35SharedTokenEmbeddingWeights Load(Stream stream, int expectedElementCount)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            if (expectedElementCount <= 0) throw new ArgumentOutOfRangeException(nameof(expectedElementCount));
            var stopwatch = Stopwatch.StartNew();
            var result = Upload(ReadCpuPayload(stream, expectedElementCount));
            stopwatch.Stop();
            result.LoadMilliseconds = stopwatch.ElapsedMilliseconds;
            return result;
        }

        private static CpuPayload ReadCpuPayload(Stream stream, int expectedElementCount)
        {
            using (var reader = new NcnnBinReader(stream))
            {
                if (reader.IsQ8Archive)
                {
                    var packed = reader.ReadQ8NcnnMatPacked(expectedElementCount, Qwen35DecoderSession.HiddenSize);
                    return new CpuPayload { Packed = packed, ElementCount = packed.ElementCount };
                }

                var values = reader.ReadNcnnMatAsFloat32(expectedElementCount, 0, 0, 0, 0);
                return new CpuPayload { Values = values, ElementCount = values.Length };
            }
        }

        private static Qwen35SharedTokenEmbeddingWeights Upload(CpuPayload payload)
        {
            if (payload == null) throw new ArgumentNullException(nameof(payload));
            if (payload.Packed == null)
            {
                return new Qwen35SharedTokenEmbeddingWeights
                {
                    Buffer = NcnnRepro.UploadImmutableFloatConstants(
                        payload.Values,
                        "Qwen35SharedTokenEmbeddingWeights.Fp32"),
                    ElementCount = payload.ElementCount
                };
            }

            var result = new Qwen35SharedTokenEmbeddingWeights { ElementCount = payload.ElementCount };
            try
            {
                result.Int8PackedBuffer = new ComputeBuffer(payload.Packed.PackedValues.Length, sizeof(uint), ComputeBufferType.Structured);
                result.Int8ScalesBuffer = new ComputeBuffer(payload.Packed.Scales.Length, sizeof(float), ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(result.Int8PackedBuffer, payload.Packed.PackedValues.Length, sizeof(uint), "Qwen35SharedTokenEmbeddingWeights.Int8Packed");
                NcnnGpuResourceTracker.RegisterBuffer(result.Int8ScalesBuffer, payload.Packed.Scales.Length, sizeof(float), "Qwen35SharedTokenEmbeddingWeights.Int8Scales");
                result.Int8PackedBuffer.SetData(payload.Packed.PackedValues);
                result.Int8ScalesBuffer.SetData(payload.Packed.Scales);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        private sealed class CpuPayload
        {
            public NcnnQ8PackedArray Packed;
            public float[] Values;
            public int ElementCount;
        }

        public void Attach(NcnnRepro repro)
        {
            if (repro == null)
                throw new ArgumentNullException(nameof(repro));
            if (Buffer == null && Int8PackedBuffer == null)
                throw new ObjectDisposedException(nameof(Qwen35SharedTokenEmbeddingWeights));
            repro.SharedTokenEmbeddingWeights = Buffer;
            repro.SharedTokenEmbeddingWeightsInt8Packed = Int8PackedBuffer;
            repro.SharedTokenEmbeddingWeightsInt8Scales = Int8ScalesBuffer;
            repro.SharedTokenEmbeddingElementCount = ElementCount;
        }

        public void Dispose()
        {
            var buffer = Buffer;
            var int8Packed = Int8PackedBuffer;
            var int8Scales = Int8ScalesBuffer;
            Buffer = null;
            Int8PackedBuffer = null;
            Int8ScalesBuffer = null;
            ElementCount = 0;
            try { NcnnGpuResourceTracker.ReleaseBuffer(buffer, "Qwen35SharedTokenEmbeddingWeights.Dispose"); buffer?.Dispose(); } catch { }
            try { NcnnGpuResourceTracker.ReleaseBuffer(int8Packed, "Qwen35SharedTokenEmbeddingWeights.Dispose.Int8Packed"); int8Packed?.Dispose(); } catch { }
            try { NcnnGpuResourceTracker.ReleaseBuffer(int8Scales, "Qwen35SharedTokenEmbeddingWeights.Dispose.Int8Scales"); int8Scales?.Dispose(); } catch { }
        }
    }
}
