using System;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnTensorBuffer : IDisposable
    {
        public int w { get; }
        public int h { get; }
        public int c { get; }
        public ComputeBuffer buffer { get; }

        public NcnnTensorBuffer(int w, int h, int c)
        {
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            if (c <= 0) throw new ArgumentOutOfRangeException(nameof(c));
            this.w = w;
            this.h = h;
            this.c = c;
            var count = checked(w * h * c);
            buffer = new ComputeBuffer(count, sizeof(float), ComputeBufferType.Structured);
        }

        public void Dispose()
        {
            try { buffer?.Dispose(); } catch { }
        }
    }
}

