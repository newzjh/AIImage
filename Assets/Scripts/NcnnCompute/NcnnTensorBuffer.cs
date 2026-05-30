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
            NcnnGpuResourceTracker.RegisterBuffer(buffer, count, sizeof(float), "NcnnTensorBuffer(3d)");
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
            NcnnGpuResourceTracker.RegisterBuffer(buffer, w, sizeof(float), "NcnnTensorBuffer(1d)");
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
            NcnnGpuResourceTracker.RegisterBuffer(buffer, checked(w * h), sizeof(float), "NcnnTensorBuffer(2d)");
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
            NcnnGpuResourceTracker.RegisterBuffer(buffer, checked(w * h * d * c), sizeof(float), "NcnnTensorBuffer(4d)");
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

        public NcnnTensorBuffer Reshape(int newDims, int newW, int newH = 1, int newD = 1, int newC = 1)
        {
            return View(newDims, newW, newH, newD, newC);
        }

        public NcnnTensorBuffer ExpandDims(int axis)
        {
            var inDims = dims;
            if (inDims < 1 || inDims > 4)
                throw new InvalidOperationException("invalid dims: " + inDims);
            if (inDims == 4)
                throw new InvalidOperationException("ExpandDims would exceed dims=4");

            if (axis < 0) axis += (inDims + 1);
            if (axis < 0 || axis > inDims)
                throw new ArgumentOutOfRangeException(nameof(axis));

            int s0 = w;
            int s1 = inDims >= 2 ? h : 1;
            int s2 = inDims == 4 ? d : (inDims >= 3 ? c : 1);
            int s3 = inDims == 4 ? c : 1;

            var sizes = new[] { s0, s1, s2, s3 };
            var outSizes = new[] { 1, 1, 1, 1 };
            var outDims = inDims + 1;
            for (var i = 0; i < outDims; i++)
            {
                if (i < axis)
                    outSizes[i] = sizes[i];
                else if (i == axis)
                    outSizes[i] = 1;
                else
                    outSizes[i] = sizes[i - 1];
            }

            if (outDims == 2) return View(2, outSizes[0], outSizes[1], 1, 1);
            if (outDims == 3) return View(3, outSizes[0], outSizes[1], 1, outSizes[2]);
            return View(4, outSizes[0], outSizes[1], outSizes[2], outSizes[3]);
        }

        public void Dispose()
        {
            if (!ownsBuffer) return;
            NcnnGpuResourceTracker.ReleaseBuffer(buffer, "NcnnTensorBuffer.Dispose");
            try { buffer?.Dispose(); } catch { }
        }
    }
}
