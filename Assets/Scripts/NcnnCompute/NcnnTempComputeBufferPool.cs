using System;
using System.Collections.Generic;
using UnityEngine;

namespace NcnnCompute
{
    internal sealed class NcnnTempComputeBufferPool
    {
        private readonly struct BufferKey : IEquatable<BufferKey>
        {
            public readonly int count;
            public readonly int stride;

            public BufferKey(int count, int stride)
            {
                this.count = count;
                this.stride = stride;
            }

            public bool Equals(BufferKey other) => count == other.count && stride == other.stride;
            public override bool Equals(object obj) => obj is BufferKey other && Equals(other);

            public override int GetHashCode()
            {
                unchecked
                {
                    return (count * 397) ^ stride;
                }
            }
        }

        private sealed class BufferEntry
        {
            public ComputeBuffer buffer;
            public int lastUsedFrame;
        }

        private readonly Dictionary<BufferKey, Stack<BufferEntry>> _freeBuffers = new Dictionary<BufferKey, Stack<BufferEntry>>();
        private readonly HashSet<int> _pooledIds = new HashSet<int>();

        public bool Enabled { get; set; }
        public int MaxPooledPerShape { get; set; } = 2;
        public int GarbageCollectFrames { get; set; } = 2;

        public ComputeBuffer Rent(int count, int stride, ComputeBufferType type, string label)
        {
            count = Mathf.Max(1, count);
            stride = Mathf.Max(1, stride);
            GarbageCollect();

            var key = new BufferKey(count, stride);
            if (Enabled && _freeBuffers.TryGetValue(key, out var pool))
            {
                while (pool.Count > 0)
                {
                    var entry = pool.Pop();
                    var pooled = entry?.buffer;
                    if (pooled == null)
                        continue;

                    _pooledIds.Remove(pooled.GetHashCode());
                    NcnnGpuResourceTracker.ReuseBuffer(pooled, label + "|pool");
                    return pooled;
                }
            }

            var allocated = new ComputeBuffer(count, stride, type);
            NcnnGpuResourceTracker.RegisterBuffer(allocated, count, stride, label + "|new");
            return allocated;
        }

        public void Return(ComputeBuffer buffer, string label)
        {
            if (buffer == null)
                return;

            if (!Enabled || MaxPooledPerShape <= 0)
            {
                Release(buffer, label);
                return;
            }

            GarbageCollect();

            var key = new BufferKey(buffer.count, buffer.stride);
            if (!_freeBuffers.TryGetValue(key, out var pool))
            {
                pool = new Stack<BufferEntry>();
                _freeBuffers[key] = pool;
            }

            if (pool.Count >= MaxPooledPerShape)
            {
                Release(buffer, label + "(pool-full)");
                return;
            }

            var id = buffer.GetHashCode();
            if (!_pooledIds.Add(id))
            {
                Release(buffer, label + "(duplicate)");
                return;
            }

            pool.Push(new BufferEntry
            {
                buffer = buffer,
                lastUsedFrame = Time.frameCount
            });
        }

        public void Clear(string label)
        {
            foreach (var kv in _freeBuffers)
            {
                var pool = kv.Value;
                while (pool.Count > 0)
                {
                    var entry = pool.Pop();
                    if (entry?.buffer == null)
                        continue;

                    _pooledIds.Remove(entry.buffer.GetHashCode());
                    Release(entry.buffer, label);
                }
            }

            _freeBuffers.Clear();
            _pooledIds.Clear();
        }

        public void GarbageCollect()
        {
            if (GarbageCollectFrames < 0 || _freeBuffers.Count == 0)
                return;

            var currentFrame = Time.frameCount;
            var removeKeys = ListPool<BufferKey>.Get();
            try
            {
                foreach (var kv in _freeBuffers)
                {
                    var pool = kv.Value;
                    if (pool == null || pool.Count == 0)
                    {
                        removeKeys.Add(kv.Key);
                        continue;
                    }

                    var keep = new Stack<BufferEntry>(pool.Count);
                    while (pool.Count > 0)
                    {
                        var entry = pool.Pop();
                        if (entry?.buffer == null)
                            continue;

                        var frameDiff = currentFrame - entry.lastUsedFrame;
                        if (frameDiff >= 0 && frameDiff <= GarbageCollectFrames)
                        {
                            keep.Push(entry);
                            continue;
                        }

                        _pooledIds.Remove(entry.buffer.GetHashCode());
                        Release(entry.buffer, "NcnnTempComputeBufferPool.GarbageCollect");
                    }

                    while (keep.Count > 0)
                        pool.Push(keep.Pop());

                    if (pool.Count == 0)
                        removeKeys.Add(kv.Key);
                }

                for (var i = 0; i < removeKeys.Count; i++)
                    _freeBuffers.Remove(removeKeys[i]);
            }
            finally
            {
                ListPool<BufferKey>.Release(removeKeys);
            }
        }

        private static void Release(ComputeBuffer buffer, string label)
        {
            NcnnGpuResourceTracker.ReleaseBuffer(buffer, label);
            try { buffer.Dispose(); } catch { }
        }

        private static class ListPool<T>
        {
            private static readonly Stack<List<T>> Pool = new Stack<List<T>>();

            public static List<T> Get()
            {
                return Pool.Count > 0 ? Pool.Pop() : new List<T>();
            }

            public static void Release(List<T> list)
            {
                if (list == null)
                    return;
                list.Clear();
                Pool.Push(list);
            }
        }
    }
}
