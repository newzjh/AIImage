using System;
using System.Diagnostics;
using System.IO;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class Qwen35SharedTokenEmbeddingWeights : IDisposable
    {
        public const int ExpectedElementCount = 254279680;

        public ComputeBuffer Buffer { get; private set; }
        public int ElementCount { get; private set; }
        public long ByteCount => (long)ElementCount * sizeof(float);
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

            var stopwatch = Stopwatch.StartNew();
            float[] values;
            using (var stream = File.OpenRead(binPath))
            using (var reader = new NcnnBinReader(stream))
                values = reader.ReadNcnnMatAsFloat32(expectedElementCount, 0, 0, 0, 0);

            var shared = new Qwen35SharedTokenEmbeddingWeights
            {
                Buffer = NcnnRepro.NewBuffer(values),
                ElementCount = values.Length
            };
            stopwatch.Stop();
            shared.LoadMilliseconds = stopwatch.ElapsedMilliseconds;
            return shared;
        }

        public void Attach(NcnnRepro repro)
        {
            if (repro == null)
                throw new ArgumentNullException(nameof(repro));
            if (Buffer == null)
                throw new ObjectDisposedException(nameof(Qwen35SharedTokenEmbeddingWeights));
            repro.SharedTokenEmbeddingWeights = Buffer;
        }

        public void Dispose()
        {
            var buffer = Buffer;
            Buffer = null;
            ElementCount = 0;
            if (buffer == null)
                return;
            try { NcnnGpuResourceTracker.ReleaseBuffer(buffer, "Qwen35SharedTokenEmbeddingWeights.Dispose"); } catch { }
            try { buffer.Dispose(); } catch { }
        }
    }
}
