using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public static class NcnnGpuResourceTracker
    {
        private sealed class BufferInfo
        {
            public int count;
            public int stride;
            public long bytes;
            public string label;
        }

        private sealed class TextureInfo
        {
            public int width;
            public int height;
            public int depth;
            public RenderTextureFormat format;
            public long bytes;
            public string label;
        }

        private static readonly Dictionary<int, BufferInfo> Buffers = new Dictionary<int, BufferInfo>();
        private static readonly Dictionary<int, TextureInfo> Textures = new Dictionary<int, TextureInfo>();
        private static readonly List<string> Timeline = new List<string>(2048);

        private static long _bufferBytes;
        private static long _textureBytes;
        private static long _peakBufferBytes;
        private static long _peakTextureBytes;
        private static long _peakTotalBytes;
        private static int _peakBufferCount;
        private static int _peakTextureCount;
        private static int _lowMemoryWarningCount;

        public static bool Enabled { get; set; }

        public static void Reset(string sessionLabel)
        {
            Buffers.Clear();
            Textures.Clear();
            Timeline.Clear();
            _bufferBytes = 0;
            _textureBytes = 0;
            _peakBufferBytes = 0;
            _peakTextureBytes = 0;
            _peakTotalBytes = 0;
            _peakBufferCount = 0;
            _peakTextureCount = 0;
            _lowMemoryWarningCount = 0;

            if (Enabled)
                Timeline.Add("session=" + (sessionLabel ?? ""));
        }

        public static void RegisterBuffer(ComputeBuffer buffer, int count, int stride, string label)
        {
            if (!Enabled || buffer == null)
                return;

            var id = buffer.GetHashCode();
            if (Buffers.ContainsKey(id))
                return;

            var bytes = Math.Max(0L, (long)count * stride);
            Buffers[id] = new BufferInfo
            {
                count = count,
                stride = stride,
                bytes = bytes,
                label = label ?? ""
            };
            _bufferBytes += bytes;
            RecordPeak();
            AddTimeline("alloc_buffer", label, bytes, count + "x" + stride);
        }

        public static void ReleaseBuffer(ComputeBuffer buffer, string label)
        {
            if (!Enabled || buffer == null)
                return;

            var id = buffer.GetHashCode();
            if (!Buffers.TryGetValue(id, out var info))
                return;

            Buffers.Remove(id);
            _bufferBytes -= info.bytes;
            AddTimeline("free_buffer", label ?? info.label, info.bytes, info.count + "x" + info.stride);
        }

        public static void RegisterTexture(RenderTexture texture, string label)
        {
            if (!Enabled || texture == null)
                return;

            var id = texture.GetHashCode();
            if (Textures.ContainsKey(id))
                return;

            var depth = Mathf.Max(1, texture.volumeDepth > 0 ? texture.volumeDepth : 1);
            var bytesPerPixel = EstimateBytesPerPixel(texture.format);
            var bytes = Math.Max(0L, (long)texture.width * texture.height * depth * bytesPerPixel);
            Textures[id] = new TextureInfo
            {
                width = texture.width,
                height = texture.height,
                depth = depth,
                format = texture.format,
                bytes = bytes,
                label = label ?? ""
            };
            _textureBytes += bytes;
            RecordPeak();
            AddTimeline("alloc_rt", label, bytes, texture.width + "x" + texture.height + "x" + depth + " " + texture.format);
        }

        public static void ReleaseTexture(RenderTexture texture, string label)
        {
            if (!Enabled || texture == null)
                return;

            var id = texture.GetHashCode();
            if (!Textures.TryGetValue(id, out var info))
                return;

            Textures.Remove(id);
            _textureBytes -= info.bytes;
            AddTimeline("free_rt", label ?? info.label, info.bytes, info.width + "x" + info.height + "x" + info.depth + " " + info.format);
        }

        public static void ReportLowMemoryWarning(string message)
        {
            if (!Enabled)
                return;
            _lowMemoryWarningCount++;
            if (Timeline.Count < 2048)
                Timeline.Add("warning=" + (message ?? ""));
        }

        public static string BuildSummary()
        {
            return "gpu_resources"
                + " | peak_total_mb=" + ToMb(_peakTotalBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | peak_buffers_mb=" + ToMb(_peakBufferBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | peak_rts_mb=" + ToMb(_peakTextureBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | live_buffers=" + Buffers.Count
                + " | live_rts=" + Textures.Count
                + " | peak_buffer_count=" + _peakBufferCount
                + " | peak_rt_count=" + _peakTextureCount
                + " | low_memory_warnings=" + _lowMemoryWarningCount;
        }

        public static void WriteReport(string directoryPath, string fileName = "gpu_resource_stats.txt")
        {
            if (!Enabled || string.IsNullOrWhiteSpace(directoryPath))
                return;

            try
            {
                Directory.CreateDirectory(directoryPath);
                using var sw = new StreamWriter(Path.Combine(directoryPath, fileName), false);
                sw.WriteLine(BuildSummary());
                sw.WriteLine("live_buffers:");
                foreach (var kv in Buffers)
                {
                    var info = kv.Value;
                    sw.WriteLine("  " + info.label + " | bytes=" + info.bytes + " | count=" + info.count + " | stride=" + info.stride);
                }
                sw.WriteLine("live_rendertextures:");
                foreach (var kv in Textures)
                {
                    var info = kv.Value;
                    sw.WriteLine("  " + info.label + " | bytes=" + info.bytes + " | shape=" + info.width + "x" + info.height + "x" + info.depth + " | format=" + info.format);
                }
                sw.WriteLine("timeline:");
                for (var i = 0; i < Timeline.Count; i++)
                    sw.WriteLine("  " + Timeline[i]);
            }
            catch
            {
            }
        }

        private static void AddTimeline(string op, string label, long bytes, string detail)
        {
            if (!Enabled || Timeline.Count >= 2048)
                return;
            Timeline.Add(op
                + " | label=" + (label ?? "")
                + " | mb=" + ToMb(bytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | total_mb=" + ToMb(_bufferBytes + _textureBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | detail=" + detail);
        }

        private static void RecordPeak()
        {
            var total = _bufferBytes + _textureBytes;
            if (_bufferBytes > _peakBufferBytes)
                _peakBufferBytes = _bufferBytes;
            if (_textureBytes > _peakTextureBytes)
                _peakTextureBytes = _textureBytes;
            if (total > _peakTotalBytes)
                _peakTotalBytes = total;
            if (Buffers.Count > _peakBufferCount)
                _peakBufferCount = Buffers.Count;
            if (Textures.Count > _peakTextureCount)
                _peakTextureCount = Textures.Count;
        }

        private static double ToMb(long bytes)
        {
            return bytes / (1024.0 * 1024.0);
        }

        private static int EstimateBytesPerPixel(RenderTextureFormat format)
        {
            return format switch
            {
                RenderTextureFormat.RHalf => 2,
                RenderTextureFormat.RFloat => 4,
                RenderTextureFormat.ARGB32 => 4,
                RenderTextureFormat.ARGBHalf => 8,
                RenderTextureFormat.ARGBFloat => 16,
                RenderTextureFormat.R8 => 1,
                _ => 8
            };
        }
    }
}
