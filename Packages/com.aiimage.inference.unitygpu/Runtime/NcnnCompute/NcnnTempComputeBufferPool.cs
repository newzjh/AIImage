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
            public int count;
            public int stride;
            public long bytes;
            public int id;
        }

        private readonly Dictionary<BufferKey, Stack<BufferEntry>> _freeBuffers = new Dictionary<BufferKey, Stack<BufferEntry>>();
        private readonly HashSet<int> _pooledIds = new HashSet<int>();
        private long _pooledBytes;

        public bool Enabled { get; set; }
        public int MaxPooledPerShape { get; set; } = 2;
        public int GarbageCollectFrames { get; set; } = 2;
        public long MaxSingleBufferBytes { get; set; } = 16L * 1024L * 1024L;
        public long MaxTotalPooledBytes { get; set; } = 128L * 1024L * 1024L;

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
                    {
                        if (entry != null)
                            _pooledBytes = Math.Max(0L, _pooledBytes - entry.bytes);
                        continue;
                    }

                    if (!IsBufferUsable(pooled))
                    {
                        _pooledIds.Remove(entry.id);
                        _pooledBytes = Math.Max(0L, _pooledBytes - entry.bytes);
                        ReleaseSilently(pooled, "NcnnTempComputeBufferPool.Rent(dead)");
                        continue;
                    }

                    _pooledIds.Remove(entry.id);
                    _pooledBytes = Math.Max(0L, _pooledBytes - entry.bytes);
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

            if (!TryGetBufferInfo(buffer, out var count, out var stride, out var bytes, out var id))
            {
                ReleaseSilently(buffer, label + "(invalid)");
                return;
            }

            GarbageCollect();

            var key = new BufferKey(count, stride);
            if (!_freeBuffers.TryGetValue(key, out var pool))
            {
                pool = new Stack<BufferEntry>();
                _freeBuffers[key] = pool;
            }

            if (pool.Count >= MaxPooledPerShape
                || bytes > MaxSingleBufferBytes
                || _pooledBytes + bytes > MaxTotalPooledBytes)
            {
                Release(buffer, label + "(pool-full)");
                return;
            }

            if (!_pooledIds.Add(id))
            {
                Release(buffer, label + "(duplicate)");
                return;
            }

            pool.Push(new BufferEntry
            {
                buffer = buffer,
                lastUsedFrame = Time.frameCount,
                count = count,
                stride = stride,
                bytes = bytes,
                id = id
            });
            _pooledBytes += bytes;
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

                    _pooledIds.Remove(entry.id);
                    _pooledBytes = Math.Max(0L, _pooledBytes - entry.bytes);
                    Release(entry.buffer, label);
                }
            }

            _freeBuffers.Clear();
            _pooledIds.Clear();
            _pooledBytes = 0L;
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
                        {
                            if (entry != null)
                                _pooledBytes = Math.Max(0L, _pooledBytes - entry.bytes);
                            continue;
                        }

                        var frameDiff = currentFrame - entry.lastUsedFrame;
                        if (frameDiff >= 0 && frameDiff <= GarbageCollectFrames && IsBufferUsable(entry.buffer))
                        {
                            keep.Push(entry);
                            continue;
                        }

                        _pooledIds.Remove(entry.id);
                        _pooledBytes = Math.Max(0L, _pooledBytes - entry.bytes);
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

        private static void ReleaseSilently(ComputeBuffer buffer, string label)
        {
            try { NcnnGpuResourceTracker.ReleaseBuffer(buffer, label); } catch { }
            try { buffer.Dispose(); } catch { }
        }

        private static long EstimateBytes(int count, int stride)
        {
            return Math.Max(0L, (long)count * Math.Max(1, stride));
        }

        private static bool TryGetBufferInfo(ComputeBuffer buffer, out int count, out int stride, out long bytes, out int id)
        {
            count = 0;
            stride = 0;
            bytes = 0L;
            id = 0;

            if (buffer == null)
                return false;

            try
            {
                count = buffer.count;
                stride = buffer.stride;
                id = buffer.GetHashCode();
                bytes = EstimateBytes(count, stride);
                return count > 0 && stride > 0;
            }
            catch
            {
                return false;
            }
        }

        private static bool IsBufferUsable(ComputeBuffer buffer)
        {
            if (buffer == null)
                return false;

            try
            {
                return buffer.count > 0 && buffer.stride > 0;
            }
            catch
            {
                return false;
            }
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
