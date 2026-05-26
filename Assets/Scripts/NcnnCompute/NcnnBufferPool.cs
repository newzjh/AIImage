using System;
using System.Collections.Generic;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnBufferPool : IDisposable
    {
        private readonly Stack<NcnnTensorBuffer> _freeBuffers = new Stack<NcnnTensorBuffer>();
        private readonly HashSet<NcnnTensorBuffer> _inUseBuffers = new HashSet<NcnnTensorBuffer>();
        private readonly object _lock = new object();
        private readonly int _maxFreeBuffers;
        private readonly int _sizeCompareRatio;

        public NcnnBufferPool(int maxFreeBuffers = 16, float sizeCompareRatio = 0.875f)
        {
            _maxFreeBuffers = maxFreeBuffers;
            _sizeCompareRatio = (int)(sizeCompareRatio * 256);
        }

        public NcnnTensorBuffer Rent(int w, int h, int c)
        {
            lock (_lock)
            {
                var targetSize = checked(w * h * c);
                Debug.Log($"[Pool] Rent w={w} h={h} c={c} targetSize={targetSize} free={_freeBuffers.Count} inUse={_inUseBuffers.Count}");

                NcnnTensorBuffer best = null;

                var e = _freeBuffers.GetEnumerator();
                while (e.MoveNext())
                {
                    var buf = e.Current;
                    var bufSize = buf.w * buf.h * buf.c;

                    if (bufSize == targetSize)
                    {
                        best = buf;
                        break;
                    }
                }

                NcnnTensorBuffer result = null;

                if (best != null)
                {
                    best.SetDimensions(w, h, c);
                    Debug.Log($"[Pool] Reuse exact buffer.count={best.buffer.count} req={w*h*c}");
                    _freeBuffers.Clear();
                    result = best;
                }

                if (result != null)
                {
                    _inUseBuffers.Add(result);
                    return result;
                }

                Debug.Log($"[Pool] Creating NEW buffer w={w} h={h} c={c} (no suitable buffer in pool)");
                var newBuf = new NcnnTensorBuffer(w, h, c) { PoolOwner = this };
                _inUseBuffers.Add(newBuf);
                return newBuf;
            }
        }

        public void Return(NcnnTensorBuffer buffer)
        {
            if (buffer == null)
                return;

            lock (_lock)
            {
                if (!_inUseBuffers.Contains(buffer))
                {
                    Debug.LogWarning($"[Pool] Return buffer not in inUseBuffers! w={buffer.w} h={buffer.h} c={buffer.c}");
                    return;
                }

                _inUseBuffers.Remove(buffer);

                if (_freeBuffers.Count >= _maxFreeBuffers)
                {
                    Debug.Log($"[Pool] Disposing buffer w={buffer.w} h={buffer.h} c={buffer.c} (pool full, free={_freeBuffers.Count})");
                    buffer.Dispose();
                    return;
                }

                _freeBuffers.Push(buffer);
                Debug.Log($"[Pool] Return w={buffer.w} h={buffer.h} c={buffer.c} free={_freeBuffers.Count} inUse={_inUseBuffers.Count}");
            }
        }

        public void Clear()
        {
            lock (_lock)
            {
                while (_freeBuffers.Count > 0)
                {
                    var buf = _freeBuffers.Pop();
                    buf?.Dispose();
                }

                foreach (var buf in _inUseBuffers)
                {
                    buf?.Dispose();
                }
                _inUseBuffers.Clear();
            }
        }

        public int FreeBufferCount
        {
            get { lock (_lock) { return _freeBuffers.Count; } }
        }

        public int InUseBufferCount
        {
            get { lock (_lock) { return _inUseBuffers.Count; } }
        }

        public void Dispose()
        {
            Clear();
        }
    }
}