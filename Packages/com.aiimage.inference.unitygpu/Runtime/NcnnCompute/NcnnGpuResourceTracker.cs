using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public readonly struct NcnnTemporaryRtDescriptor
    {
        public NcnnTemporaryRtDescriptor(
            NcnnRepro.BufferShape logicalShape,
            NcnnRepro.BufferShape storageShape,
            RenderTextureDescriptor renderTextureDescriptor,
            string owner,
            string node,
            string label)
        {
            LogicalShape = logicalShape;
            StorageShape = storageShape;
            RenderTextureDescriptor = renderTextureDescriptor;
            Owner = owner ?? string.Empty;
            Node = node ?? string.Empty;
            Label = label ?? string.Empty;
        }

        public NcnnRepro.BufferShape LogicalShape { get; }
        public NcnnRepro.BufferShape StorageShape { get; }
        public RenderTextureDescriptor RenderTextureDescriptor { get; }
        public string Owner { get; }
        public string Node { get; }
        public string Label { get; }
        public int Width => Mathf.Max(1, RenderTextureDescriptor.width);
        public int Height => Mathf.Max(1, RenderTextureDescriptor.height);
        public int ArrayDepth => Mathf.Max(1, RenderTextureDescriptor.volumeDepth);
        public TextureDimension Dimension => RenderTextureDescriptor.dimension;
        public RenderTextureFormat Format => RenderTextureDescriptor.colorFormat;
        public UnityEngine.Experimental.Rendering.GraphicsFormat GraphicsFormat => RenderTextureDescriptor.graphicsFormat;
        public bool EnableRandomWrite => RenderTextureDescriptor.enableRandomWrite;
    }

    public static class NcnnGpuResourceTracker
    {
        public readonly struct StatsSnapshot
        {
            public readonly long currentBufferBytes;
            public readonly long currentTextureBytes;
            public readonly long peakBufferBytes;
            public readonly long peakTextureBytes;
            public readonly long peakTotalBytes;
            public readonly int liveBufferCount;
            public readonly int liveTextureCount;
            public readonly int peakBufferCount;
            public readonly int peakTextureCount;
            public readonly int lowMemoryWarningCount;
            public readonly long currentTemporaryTextureBytes;
            public readonly long peakTemporaryTextureBytes;
            public readonly int liveTemporaryTextureCount;

            public StatsSnapshot(
                long currentBufferBytes,
                long currentTextureBytes,
                long peakBufferBytes,
                long peakTextureBytes,
                long peakTotalBytes,
                int liveBufferCount,
                int liveTextureCount,
                int peakBufferCount,
                int peakTextureCount,
                int lowMemoryWarningCount,
                long currentTemporaryTextureBytes,
                long peakTemporaryTextureBytes,
                int liveTemporaryTextureCount)
            {
                this.currentBufferBytes = currentBufferBytes;
                this.currentTextureBytes = currentTextureBytes;
                this.peakBufferBytes = peakBufferBytes;
                this.peakTextureBytes = peakTextureBytes;
                this.peakTotalBytes = peakTotalBytes;
                this.liveBufferCount = liveBufferCount;
                this.liveTextureCount = liveTextureCount;
                this.peakBufferCount = peakBufferCount;
                this.peakTextureCount = peakTextureCount;
                this.lowMemoryWarningCount = lowMemoryWarningCount;
                this.currentTemporaryTextureBytes = currentTemporaryTextureBytes;
                this.peakTemporaryTextureBytes = peakTemporaryTextureBytes;
                this.liveTemporaryTextureCount = liveTemporaryTextureCount;
            }

            public double currentBufferMb => ToMb(currentBufferBytes);
            public double currentTextureMb => ToMb(currentTextureBytes);
            public double currentTotalMb => ToMb(currentBufferBytes + currentTextureBytes);
            public double peakBufferMb => ToMb(peakBufferBytes);
            public double peakTextureMb => ToMb(peakTextureBytes);
            public double peakTotalMb => ToMb(peakTotalBytes);
            public double currentTemporaryTextureMb => ToMb(currentTemporaryTextureBytes);
            public double peakTemporaryTextureMb => ToMb(peakTemporaryTextureBytes);
        }

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
            public bool temporary;
            public string owner;
            public string node;
            public NcnnRepro.BufferShape logicalShape;
            public NcnnRepro.BufferShape storageShape;
            public TextureDimension dimension;
            public bool enableRandomWrite;
            public UnityEngine.Experimental.Rendering.GraphicsFormat graphicsFormat;
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
        private static long _temporaryTextureBytes;
        private static long _peakTemporaryTextureBytes;
        private static int _temporaryTextureCount;

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
            _temporaryTextureBytes = 0;
            _peakTemporaryTextureBytes = 0;
            _temporaryTextureCount = 0;

            if (Enabled)
                Timeline.Add("session=" + (sessionLabel ?? ""));
        }

        public static void RegisterBuffer(ComputeBuffer buffer, int count, int stride, string label)
        {
            if (buffer == null)
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
            if (buffer == null)
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
            if (texture == null)
                return;

            RegisterTextureCore(texture.GetHashCode(), texture.width, texture.height, Mathf.Max(1, texture.volumeDepth > 0 ? texture.volumeDepth : 1), texture.format, label);
        }

        public static void ReleaseTexture(RenderTexture texture, string label)
        {
            if (texture == null)
                return;

            ReleaseTextureCore(texture.GetHashCode(), label);
        }

        public static void UpdateTextureLabel(RenderTexture texture, string label)
        {
            if (texture == null)
                return;

            var id = texture.GetHashCode();
            if (!Textures.TryGetValue(id, out var info) || info == null)
                return;

            info.label = label ?? "";
        }

        public static void RegisterTextureHandle(int handle, int width, int height, int depth, RenderTextureFormat format, string label)
        {
            if (handle == 0)
                return;

            RegisterTextureCore(ToVirtualTextureKey(handle), width, height, depth, format, label);
        }

        public static void ReleaseTextureHandle(int handle, string label)
        {
            if (handle == 0)
                return;

            ReleaseTextureCore(ToVirtualTextureKey(handle), label);
        }

        public static void ReportLowMemoryWarning(string message)
        {
            _lowMemoryWarningCount++;
            if (Timeline.Count < 2048)
                Timeline.Add("warning=" + (message ?? ""));
        }

        public static string BuildSummary()
        {
            return "gpu_resources"
                + " | current_total_mb=" + ToMb(_bufferBytes + _textureBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | live_buffers_mb=" + ToMb(_bufferBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | live_rts_mb=" + ToMb(_textureBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | peak_total_mb=" + ToMb(_peakTotalBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | peak_buffers_mb=" + ToMb(_peakBufferBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | peak_rts_mb=" + ToMb(_peakTextureBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | live_buffers=" + Buffers.Count
                + " | live_rts=" + Textures.Count
                + " | peak_buffer_count=" + _peakBufferCount
                + " | peak_rt_count=" + _peakTextureCount
                + " | live_temp_rts=" + _temporaryTextureCount
                + " | live_temp_rts_mb=" + ToMb(_temporaryTextureBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | peak_temp_rts_mb=" + ToMb(_peakTemporaryTextureBytes).ToString("F3", CultureInfo.InvariantCulture)
                + " | low_memory_warnings=" + _lowMemoryWarningCount;
        }

        public static StatsSnapshot GetStatsSnapshot()
        {
            return new StatsSnapshot(
                _bufferBytes,
                _textureBytes,
                _peakBufferBytes,
                _peakTextureBytes,
                _peakTotalBytes,
                Buffers.Count,
                Textures.Count,
                _peakBufferCount,
                _peakTextureCount,
                _lowMemoryWarningCount,
                _temporaryTextureBytes,
                _peakTemporaryTextureBytes,
                _temporaryTextureCount);
        }

        public static void RegisterTemporaryTexture(RenderTexture texture, NcnnTemporaryRtDescriptor descriptor)
        {
            if (texture == null)
                return;
            RegisterTextureCore(texture.GetHashCode(), descriptor, texture.format);
        }

        public static void RegisterTemporaryTextureHandle(int handle, NcnnTemporaryRtDescriptor descriptor)
        {
            if (handle == 0)
                return;
            RegisterTextureCore(ToVirtualTextureKey(handle), descriptor, descriptor.Format);
        }

        public static void EnsureTemporaryTextureBudget(long requestedBytes, long budgetBytes, string node)
        {
            if (budgetBytes <= 0 || _temporaryTextureBytes + requestedBytes <= budgetBytes)
                return;

            throw new InvalidOperationException(
                "temporary RT budget exceeded"
                + " | required_bytes=" + requestedBytes.ToString(CultureInfo.InvariantCulture)
                + " | current_peak_bytes=" + _peakTemporaryTextureBytes.ToString(CultureInfo.InvariantCulture)
                + " | current_live_bytes=" + _temporaryTextureBytes.ToString(CultureInfo.InvariantCulture)
                + " | budget_bytes=" + budgetBytes.ToString(CultureInfo.InvariantCulture)
                + " | node=" + (node ?? string.Empty));
        }

        public static long EstimateTemporaryTextureBytes(NcnnTemporaryRtDescriptor descriptor)
        {
            return EstimateTextureBytes(descriptor.Width, descriptor.Height, descriptor.ArrayDepth, descriptor.Format);
        }

        public static string[] GetTimelineSnapshot()
        {
            return Timeline.ToArray();
        }

        public static string[] GetUnreleasedTemporaryTextureDiagnostics()
        {
            var diagnostics = new List<string>();
            foreach (var info in Textures.Values)
            {
                if (info == null || !info.temporary)
                    continue;
                diagnostics.Add(
                    "owner=" + info.owner
                    + " | node=" + info.node
                    + " | logical=" + TensorDescriptor.FormatShape(info.logicalShape)
                    + " | storage=" + TensorDescriptor.FormatShape(info.storageShape)
                    + " | bytes=" + info.bytes.ToString(CultureInfo.InvariantCulture));
            }
            return diagnostics.ToArray();
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
                    sw.WriteLine("  " + info.label + " | bytes=" + info.bytes + " | shape=" + info.width + "x" + info.height + "x" + info.depth + " | format=" + info.format
                        + (info.temporary
                            ? " | temporary=true | owner=" + info.owner + " | node=" + info.node + " | logical=" + TensorDescriptor.FormatShape(info.logicalShape)
                                + " | storage=" + TensorDescriptor.FormatShape(info.storageShape) + " | graphics_format=" + info.graphicsFormat + " | random_write=" + info.enableRandomWrite
                            : string.Empty));
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

        private static void RegisterTextureCore(int id, int width, int height, int depth, RenderTextureFormat format, string label)
        {
            if (Textures.ContainsKey(id))
                return;

            depth = Mathf.Max(1, depth);
            var bytesPerPixel = EstimateBytesPerPixel(format);
            var bytes = Math.Max(0L, (long)Mathf.Max(1, width) * Mathf.Max(1, height) * depth * bytesPerPixel);
            Textures[id] = new TextureInfo
            {
                width = Mathf.Max(1, width),
                height = Mathf.Max(1, height),
                depth = depth,
                format = format,
                bytes = bytes,
                label = label ?? ""
            };
            _textureBytes += bytes;
            RecordPeak();
            AddTimeline("alloc_rt", label, bytes, width + "x" + height + "x" + depth + " " + format);
        }

        private static void RegisterTextureCore(int id, NcnnTemporaryRtDescriptor descriptor, RenderTextureFormat format)
        {
            if (Textures.ContainsKey(id))
                return;

            var bytes = EstimateTemporaryTextureBytes(descriptor);
            var info = new TextureInfo
            {
                width = descriptor.Width,
                height = descriptor.Height,
                depth = descriptor.ArrayDepth,
                format = format,
                bytes = bytes,
                label = descriptor.Label,
                temporary = true,
                owner = descriptor.Owner,
                node = descriptor.Node,
                logicalShape = descriptor.LogicalShape,
                storageShape = descriptor.StorageShape,
                dimension = descriptor.Dimension,
                enableRandomWrite = descriptor.EnableRandomWrite,
                graphicsFormat = descriptor.GraphicsFormat
            };
            Textures[id] = info;
            _textureBytes += bytes;
            _temporaryTextureBytes += bytes;
            _temporaryTextureCount++;
            if (_temporaryTextureBytes > _peakTemporaryTextureBytes)
                _peakTemporaryTextureBytes = _temporaryTextureBytes;
            RecordPeak();
            AddTimeline(
                "alloc_temp_rt",
                descriptor.Label,
                bytes,
                "owner=" + descriptor.Owner
                + " node=" + descriptor.Node
                + " logical=" + TensorDescriptor.FormatShape(descriptor.LogicalShape)
                + " storage=" + TensorDescriptor.FormatShape(descriptor.StorageShape)
                + " dimension=" + descriptor.Dimension
                + " graphics_format=" + descriptor.GraphicsFormat
                + " random_write=" + descriptor.EnableRandomWrite);
        }

        private static void ReleaseTextureCore(int id, string label)
        {
            if (!Textures.TryGetValue(id, out var info))
                return;

            Textures.Remove(id);
            _textureBytes -= info.bytes;
            if (info.temporary)
            {
                _temporaryTextureBytes = Math.Max(0L, _temporaryTextureBytes - info.bytes);
                _temporaryTextureCount = Math.Max(0, _temporaryTextureCount - 1);
                AddTimeline("free_temp_rt", label ?? info.label, info.bytes, "owner=" + info.owner + " node=" + info.node);
                return;
            }
            AddTimeline("free_rt", label ?? info.label, info.bytes, info.width + "x" + info.height + "x" + info.depth + " " + info.format);
        }

        private static int ToVirtualTextureKey(int handle)
        {
            return unchecked(handle ^ int.MinValue);
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

        private static long EstimateTextureBytes(int width, int height, int depth, RenderTextureFormat format)
        {
            return Math.Max(0L,
                (long)Mathf.Max(1, width)
                * Mathf.Max(1, height)
                * Mathf.Max(1, depth)
                * EstimateBytesPerPixel(format));
        }
    }
}
