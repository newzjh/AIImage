using System;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnTensorBuffer : IDisposable
    {
        public int dims { get; }
        public int w { get; }
        public int h { get; }
        public int d { get; }
        public int c { get; }
        public ComputeBuffer buffer { get; }
        public bool ownsBuffer { get; }

        public NcnnTensorBuffer(int w, int h, int c)
        {
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            if (c <= 0) throw new ArgumentOutOfRangeException(nameof(c));
            dims = 3;
            this.w = w;
            this.h = h;
            this.d = 1;
            this.c = c;
            ownsBuffer = true;
            var count = checked(w * h * c);
            buffer = new ComputeBuffer(count, sizeof(float), ComputeBufferType.Structured);
        }

        public NcnnTensorBuffer(int w)
        {
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            dims = 1;
            this.w = w;
            this.h = 1;
            this.d = 1;
            this.c = 1;
            ownsBuffer = true;
            buffer = new ComputeBuffer(w, sizeof(float), ComputeBufferType.Structured);
        }

        public NcnnTensorBuffer(int w, int h)
        {
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            dims = 2;
            this.w = w;
            this.h = h;
            this.d = 1;
            this.c = 1;
            ownsBuffer = true;
            buffer = new ComputeBuffer(checked(w * h), sizeof(float), ComputeBufferType.Structured);
        }

        public NcnnTensorBuffer(int w, int h, int d, int c)
        {
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            if (d <= 0) throw new ArgumentOutOfRangeException(nameof(d));
            if (c <= 0) throw new ArgumentOutOfRangeException(nameof(c));
            dims = 4;
            this.w = w;
            this.h = h;
            this.d = d;
            this.c = c;
            ownsBuffer = true;
            buffer = new ComputeBuffer(checked(w * h * d * c), sizeof(float), ComputeBufferType.Structured);
        }

        internal NcnnTensorBuffer(ComputeBuffer existing, int dims, int w, int h, int d, int c, bool ownsBuffer)
        {
            buffer = existing ?? throw new ArgumentNullException(nameof(existing));
            this.dims = dims;
            this.w = w;
            this.h = h;
            this.d = d;
            this.c = c;
            this.ownsBuffer = ownsBuffer;
        }

        public int elementCount => checked(w * h * d * c);

        public NcnnTensorBuffer View(int dims, int w, int h, int d, int c)
        {
            if (dims < 1 || dims > 4) throw new ArgumentOutOfRangeException(nameof(dims));
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (dims >= 2 && h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            if (dims >= 3 && c <= 0) throw new ArgumentOutOfRangeException(nameof(c));
            if (dims == 4 && d <= 0) throw new ArgumentOutOfRangeException(nameof(d));
            if (dims != 4) d = 1;
            if (dims <= 2) c = 1;
            var count = checked(w * h * d * c);
            if (count != buffer.count)
                throw new ArgumentOutOfRangeException(nameof(w), "view elementCount mismatch");
            return new NcnnTensorBuffer(buffer, dims, w, h, d, c, false);
        }

        public void Dispose()
        {
            if (!ownsBuffer) return;
            try { buffer?.Dispose(); } catch { }
        }
    }
}
