using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using Newtonsoft.Json.Linq;

namespace NcnnCompute
{
    public sealed class Qwen35MobileAssetSet
    {
        public const string ManifestFileName = "qwen3.5_mobile_q8_assets.json";
        public const string ManifestSchema = "qwen35.mobile-q8-assets/v1";
        public const string PrecisionManifestFileName = "qwen3.5_mobile_q8.model.json";

        private readonly Dictionary<string, Entry> _entries = new Dictionary<string, Entry>(StringComparer.Ordinal);

        private Qwen35MobileAssetSet(string modelDirectory, string manifestPath)
        {
            ModelDirectory = modelDirectory;
            ManifestPath = manifestPath;
        }

        public string ModelDirectory { get; }
        public string ManifestPath { get; }
        public long StoredWeightBytes { get; private set; }
        public bool WeightOnly { get; private set; }

        public static Qwen35MobileAssetSet TryLoad(
            string modelDirectory,
            bool verifyHashes = false,
            Action<long, long, string> onHashProgress = null,
            CancellationToken cancellationToken = default)
        {
            var root = Path.GetFullPath(modelDirectory ?? string.Empty);
            var path = Path.Combine(root, ManifestFileName);
            if (!File.Exists(path)) return null;

            var document = JObject.Parse(File.ReadAllText(path));
            if (!string.Equals((string)document["schema"], ManifestSchema, StringComparison.Ordinal))
                throw new InvalidDataException("Unsupported Qwen3.5 mobile asset manifest: " + path);
            var result = new Qwen35MobileAssetSet(root, path)
            {
                WeightOnly = (bool?)document["weight_only"] ?? false
            };
            var logicalFiles = document["logical_files"] as JObject
                ?? throw new InvalidDataException("Qwen3.5 mobile asset manifest has no logical_files object: " + path);
            long totalHashBytes = 0;
            if (verifyHashes)
            {
                foreach (var property in logicalFiles.Properties())
                {
                    if (!(property.Value is JObject item) || !(item["parts"] is JArray parts))
                        continue;
                    foreach (var token in parts)
                        totalHashBytes = checked(totalHashBytes + ((long?)token?["bytes"] ?? 0));
                }
            }
            long completedHashBytes = 0;
            foreach (var property in logicalFiles.Properties())
            {
                cancellationToken.ThrowIfCancellationRequested();
                var item = property.Value as JObject
                    ?? throw new InvalidDataException("Invalid logical asset entry: " + property.Name);
                var parts = item["parts"] as JArray
                    ?? throw new InvalidDataException("Logical asset has no parts: " + property.Name);
                if (parts.Count == 0) throw new InvalidDataException("Logical asset has an empty part list: " + property.Name);
                var entry = new Entry { LogicalName = property.Name };
                foreach (var token in parts)
                {
                    var part = token as JObject ?? throw new InvalidDataException("Invalid shard entry: " + property.Name);
                    var relative = (string)part["file"];
                    if (string.IsNullOrWhiteSpace(relative)) throw new InvalidDataException("Shard file name is empty: " + property.Name);
                    var fullPath = ResolveContainedPath(root, relative);
                    if (!File.Exists(fullPath)) throw new FileNotFoundException("Qwen3.5 mobile shard is missing.", fullPath);
                    var actualBytes = new FileInfo(fullPath).Length;
                    var expectedBytes = (long?)part["bytes"] ?? -1;
                    if (actualBytes != expectedBytes)
                        throw new InvalidDataException("Qwen3.5 mobile shard size mismatch: " + relative + " expected=" + expectedBytes + " actual=" + actualBytes);
                    var expectedHash = ((string)part["sha256"] ?? string.Empty).ToLowerInvariant();
                    if (expectedHash.Length != 64) throw new InvalidDataException("Qwen3.5 shard SHA-256 is missing: " + relative);
                    if (verifyHashes)
                    {
                        var hashBase = completedHashBytes;
                        var actualHash = ComputeSha256(
                            fullPath,
                            bytes => onHashProgress?.Invoke(hashBase + bytes, totalHashBytes, relative),
                            cancellationToken);
                        if (!string.Equals(actualHash, expectedHash, StringComparison.Ordinal))
                            throw new InvalidDataException("Qwen3.5 mobile shard SHA-256 mismatch: " + relative);
                        completedHashBytes = checked(completedHashBytes + actualBytes);
                    }
                    entry.Parts.Add(new Part { Path = fullPath, Bytes = actualBytes, Sha256 = expectedHash });
                    entry.StoredBytes += actualBytes;
                }
                var declaredStoredBytes = (long?)item["stored_bytes"] ?? -1;
                if (declaredStoredBytes != entry.StoredBytes)
                    throw new InvalidDataException("Qwen3.5 logical stored byte count mismatch: " + property.Name);
                result._entries.Add(property.Name, entry);
                result.StoredWeightBytes += entry.StoredBytes;
            }
            return result;
        }

        public bool Contains(string logicalName) => !string.IsNullOrWhiteSpace(logicalName) && _entries.ContainsKey(logicalName);

        public Stream OpenRead(string logicalName)
        {
            if (!_entries.TryGetValue(logicalName, out var entry))
                throw new FileNotFoundException("Qwen3.5 logical mobile asset is missing from the manifest.", logicalName);
            return new Qwen35ShardReadStream(entry.Parts);
        }

        public long GetStoredBytes(string logicalName)
        {
            if (!_entries.TryGetValue(logicalName, out var entry)) return 0;
            return entry.StoredBytes;
        }

        internal sealed class Part
        {
            public string Path;
            public long Bytes;
            public string Sha256;
        }

        private sealed class Entry
        {
            public string LogicalName;
            public long StoredBytes;
            public readonly List<Part> Parts = new List<Part>();
        }

        private static string ResolveContainedPath(string root, string relative)
        {
            if (Path.IsPathRooted(relative)) throw new InvalidDataException("Qwen3.5 shard paths must be relative: " + relative);
            var full = Path.GetFullPath(Path.Combine(root, relative));
            var prefix = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            if (!full.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Qwen3.5 shard path escapes the model directory: " + relative);
            return full;
        }

        private static string ComputeSha256(
            string path,
            Action<long> onProgress,
            CancellationToken cancellationToken)
        {
            using (var sha = SHA256.Create())
            using (var stream = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                4 * 1024 * 1024,
                FileOptions.SequentialScan))
            {
                var buffer = new byte[4 * 1024 * 1024];
                long completed = 0;
                while (true)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var read = stream.Read(buffer, 0, buffer.Length);
                    if (read <= 0) break;
                    sha.TransformBlock(buffer, 0, read, null, 0);
                    completed += read;
                    onProgress?.Invoke(completed);
                }
                sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
                return BitConverter.ToString(sha.Hash).Replace("-", string.Empty).ToLowerInvariant();
            }
        }
    }

    public static class Qwen35ModelAssetResolver
    {
        public static Stream OpenBin(string modelDirectory, string logicalName)
        {
            var mobile = Qwen35MobileAssetSet.TryLoad(modelDirectory);
            if (mobile != null && mobile.Contains(logicalName)) return mobile.OpenRead(logicalName);
            return File.OpenRead(Path.Combine(modelDirectory, logicalName));
        }

        public static long GetStoredBytes(string modelDirectory, string logicalName)
        {
            var mobile = Qwen35MobileAssetSet.TryLoad(modelDirectory);
            if (mobile != null && mobile.Contains(logicalName)) return mobile.GetStoredBytes(logicalName);
            var path = Path.Combine(modelDirectory, logicalName);
            return File.Exists(path) ? new FileInfo(path).Length : 0;
        }

        public static void ApplyMobilePrecisionManifest(NcnnRepro repro, string modelDirectory)
        {
            if (repro == null) throw new ArgumentNullException(nameof(repro));
            var mobile = Qwen35MobileAssetSet.TryLoad(modelDirectory);
            if (mobile == null) return;
            var path = Path.Combine(modelDirectory, Qwen35MobileAssetSet.PrecisionManifestFileName);
            if (!File.Exists(path)) throw new FileNotFoundException("Qwen3.5 mobile precision manifest is missing.", path);
            repro.ApplyModelManifest(NcnnModelManifestLoader.LoadFromFile(path));
        }
    }

    internal sealed class Qwen35ShardReadStream : Stream
    {
        private readonly IReadOnlyList<Qwen35MobileAssetSet.Part> _parts;
        private readonly long[] _offsets;
        private readonly long _length;
        private FileStream _current;
        private int _currentPart = -1;
        private long _position;

        public Qwen35ShardReadStream(IReadOnlyList<Qwen35MobileAssetSet.Part> parts)
        {
            _parts = parts ?? throw new ArgumentNullException(nameof(parts));
            if (parts.Count == 0) throw new ArgumentException("At least one shard is required.", nameof(parts));
            _offsets = new long[parts.Count + 1];
            for (var i = 0; i < parts.Count; i++) _offsets[i + 1] = checked(_offsets[i] + parts[i].Bytes);
            _length = _offsets[parts.Count];
        }

        public override bool CanRead => true;
        public override bool CanSeek => true;
        public override bool CanWrite => false;
        public override long Length => _length;
        public override long Position { get => _position; set => Seek(value, SeekOrigin.Begin); }

        public override int Read(byte[] buffer, int offset, int count)
        {
            if (buffer == null) throw new ArgumentNullException(nameof(buffer));
            if (offset < 0 || count < 0 || offset + count > buffer.Length) throw new ArgumentOutOfRangeException(nameof(offset));
            if (_position >= _length || count == 0) return 0;
            var total = 0;
            while (count > 0 && _position < _length)
            {
                var partIndex = FindPart(_position);
                EnsurePart(partIndex);
                var local = _position - _offsets[partIndex];
                _current.Position = local;
                var available = (int)Math.Min(count, _parts[partIndex].Bytes - local);
                var read = _current.Read(buffer, offset, available);
                if (read <= 0) throw new EndOfStreamException("Unexpected end of Qwen3.5 mobile shard: " + _parts[partIndex].Path);
                offset += read;
                count -= read;
                total += read;
                _position += read;
            }
            return total;
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            long target;
            switch (origin)
            {
                case SeekOrigin.Begin: target = offset; break;
                case SeekOrigin.Current: target = checked(_position + offset); break;
                case SeekOrigin.End: target = checked(_length + offset); break;
                default: throw new ArgumentOutOfRangeException(nameof(origin));
            }
            if (target < 0 || target > _length) throw new IOException("Seek target is outside the Qwen3.5 logical asset.");
            _position = target;
            return target;
        }

        public override void Flush() { }
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing) _current?.Dispose();
            _current = null;
            _currentPart = -1;
            base.Dispose(disposing);
        }

        private int FindPart(long position)
        {
            if (position == _length) return _parts.Count - 1;
            var low = 0;
            var high = _parts.Count - 1;
            while (low <= high)
            {
                var mid = low + ((high - low) >> 1);
                if (position < _offsets[mid]) high = mid - 1;
                else if (position >= _offsets[mid + 1]) low = mid + 1;
                else return mid;
            }
            throw new IOException("Could not map Qwen3.5 logical stream position to a shard.");
        }

        private void EnsurePart(int index)
        {
            if (_currentPart == index && _current != null) return;
            _current?.Dispose();
            _current = new FileStream(_parts[index].Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan);
            _currentPart = index;
        }
    }
}
