using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using Aexis.Ncnn;
using Aexis.Samples.Json.Linq;
using Aexis.Execution;

namespace AIImage.Qwen35
{
    public sealed class Qwen35MobileAssetSet
    {
        public const string ManifestFileName = "qwen3.5_mobile_q8_assets.json";
        public const string ManifestSchema = "qwen35.mobile-q8-assets/v1";
        public const string PrecisionManifestFileName = "qwen3.5_mobile_q8.model.json";
        public const string Int4GpuPrecisionManifestFileName = "qwen3.5_mobile_q4gpu.model.json";
        public const string Q4ManifestFileName = "qwen3.5_mobile_q4_assets.json";
        public const string Q4ManifestSchema = "qwen35.mobile-q4-assets/v1";
        public const string Q4PrecisionManifestFileName = "qwen3.5_mobile_q4.model.json";
        public const string Q4ProjectionPrecisionManifestFileName = "qwen3.5_mobile_q4_projection.model.json";
        public const string RuntimePrecisionEnvironmentVariable = "AIIMAGE_QWEN35_RUNTIME_PRECISION";
        private const string ValidationCacheSchema = "qwen35.mobile-q8-validation-cache/v1";

        private static string _validationCacheRoot;

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
        public int QuantizationBits { get; private set; }
        public bool HashesVerifiedFromCache { get; private set; }

        public static void ConfigureValidationCacheRoot(string directory)
        {
            if (string.IsNullOrWhiteSpace(directory))
                throw new ArgumentException("Qwen3.5 validation cache directory is empty.", nameof(directory));
            _validationCacheRoot = Path.GetFullPath(directory);
        }

        public static Qwen35MobileAssetSet TryLoad(
            string modelDirectory,
            bool verifyHashes = false,
            Action<long, long, string> onHashProgress = null,
            CancellationToken cancellationToken = default)
        {
            var root = Path.GetFullPath(modelDirectory ?? string.Empty);
            var q4Path = Path.Combine(root, Q4ManifestFileName);
            var q8Path = Path.Combine(root, ManifestFileName);
            var path = File.Exists(q4Path) ? q4Path : q8Path;
            if (!File.Exists(path)) return null;

            var document = JObject.Parse(File.ReadAllText(path));
            var schema = (string)document["schema"];
            if (!string.Equals(schema, ManifestSchema, StringComparison.Ordinal)
                && !string.Equals(schema, Q4ManifestSchema, StringComparison.Ordinal))
                throw new InvalidDataException("Unsupported Qwen3.5 mobile asset manifest: " + path);
            var result = new Qwen35MobileAssetSet(root, path)
            {
                WeightOnly = (bool?)document["weight_only"] ?? false,
                QuantizationBits = (int?)document["quantization_bits"]
                    ?? (string.Equals(schema, Q4ManifestSchema, StringComparison.Ordinal) ? 4 : 8)
            };
            if (result.QuantizationBits != 4 && result.QuantizationBits != 8)
                throw new InvalidDataException("Unsupported Qwen3.5 mobile quantization bit width: " + result.QuantizationBits);
            var logicalFiles = document["logical_files"] as JObject
                ?? throw new InvalidDataException("Qwen3.5 mobile asset manifest has no logical_files object: " + path);
            var useValidationCache = verifyHashes && IsValidationCacheCurrent(root, path, logicalFiles);
            result.HashesVerifiedFromCache = useValidationCache;
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
                        if (useValidationCache)
                        {
                            onHashProgress?.Invoke(completedHashBytes + actualBytes, totalHashBytes, "cached:" + relative);
                        }
                        else
                        {
                            var hashBase = completedHashBytes;
                            var actualHash = ComputeSha256(
                                fullPath,
                                bytes => onHashProgress?.Invoke(hashBase + bytes, totalHashBytes, relative),
                                cancellationToken);
                            if (!string.Equals(actualHash, expectedHash, StringComparison.Ordinal))
                                throw new InvalidDataException("Qwen3.5 mobile shard SHA-256 mismatch: " + relative);
                        }
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
            if (verifyHashes && !useValidationCache)
                WriteValidationCache(root, path, logicalFiles);
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

        private static bool IsValidationCacheCurrent(string root, string manifestPath, JObject logicalFiles)
        {
            try
            {
                var cachePath = GetValidationCachePath(root);
                if (!File.Exists(cachePath)) return false;
                var cache = JObject.Parse(File.ReadAllText(cachePath));
                if (!string.Equals((string)cache["schema"], ValidationCacheSchema, StringComparison.Ordinal)
                    || !string.Equals((string)cache["manifest_sha256"], ComputeSha256(manifestPath, null, default), StringComparison.Ordinal)
                    || !(cache["parts"] is JObject cachedParts))
                {
                    return false;
                }

                foreach (var property in logicalFiles.Properties())
                {
                    if (!(property.Value is JObject item) || !(item["parts"] is JArray parts))
                        return false;
                    foreach (var token in parts)
                    {
                        if (!(token is JObject part)) return false;
                        var relative = (string)part["file"];
                        var expectedBytes = (long?)part["bytes"] ?? -1;
                        var expectedHash = ((string)part["sha256"] ?? string.Empty).ToLowerInvariant();
                        var fullPath = ResolveContainedPath(root, relative);
                        if (!(cachedParts[relative] is JObject cached)
                            || !File.Exists(fullPath)
                            || new FileInfo(fullPath).Length != expectedBytes
                            || (long?)cached["bytes"] != expectedBytes
                            || (long?)cached["last_write_utc_ticks"] != File.GetLastWriteTimeUtc(fullPath).Ticks
                            || !string.Equals((string)cached["sha256"], expectedHash, StringComparison.Ordinal))
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static void WriteValidationCache(string root, string manifestPath, JObject logicalFiles)
        {
            try
            {
                var partsDocument = new JObject();
                foreach (var property in logicalFiles.Properties())
                {
                    if (!(property.Value is JObject item) || !(item["parts"] is JArray parts))
                        continue;
                    foreach (var token in parts)
                    {
                        if (!(token is JObject part)) continue;
                        var relative = (string)part["file"];
                        var fullPath = ResolveContainedPath(root, relative);
                        partsDocument[relative] = new JObject
                        {
                            ["bytes"] = new FileInfo(fullPath).Length,
                            ["last_write_utc_ticks"] = File.GetLastWriteTimeUtc(fullPath).Ticks,
                            ["sha256"] = ((string)part["sha256"] ?? string.Empty).ToLowerInvariant()
                        };
                    }
                }

                var document = new JObject
                {
                    ["schema"] = ValidationCacheSchema,
                    ["manifest_sha256"] = ComputeSha256(manifestPath, null, default),
                    ["parts"] = partsDocument
                };
                var cachePath = GetValidationCachePath(root);
                Directory.CreateDirectory(Path.GetDirectoryName(cachePath));
                var temporaryPath = cachePath + ".tmp";
                File.WriteAllText(temporaryPath, document.ToString(Aexis.Samples.Json.Formatting.None));
                if (File.Exists(cachePath)) File.Delete(cachePath);
                File.Move(temporaryPath, cachePath);
            }
            catch
            {
                // A read-only or unavailable cache must never invalidate otherwise valid assets.
            }
        }

        private static string GetValidationCachePath(string root)
        {
            var cacheRoot = string.IsNullOrWhiteSpace(_validationCacheRoot)
                ? Path.Combine(Path.GetTempPath(), "AIImage", "Qwen35ValidationCache")
                : _validationCacheRoot;
            string key;
            using (var sha = SHA256.Create())
            {
                var bytes = Encoding.UTF8.GetBytes(Path.GetFullPath(root).ToUpperInvariant());
                key = BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", string.Empty).ToLowerInvariant();
            }
            return Path.Combine(cacheRoot, key + ".json");
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

        public static void ApplyMobilePrecisionManifest(AexisGraphSession repro, string modelDirectory)
        {
            if (repro == null) throw new ArgumentNullException(nameof(repro));
            var mobile = Qwen35MobileAssetSet.TryLoad(modelDirectory);
            if (mobile == null) return;
            var requestedPrecision = Environment.GetEnvironmentVariable(Qwen35MobileAssetSet.RuntimePrecisionEnvironmentVariable);
            string manifestName;
            if (mobile.QuantizationBits == 4)
            {
                if (!string.IsNullOrWhiteSpace(requestedPrecision)
                    && !string.Equals(requestedPrecision, "INT4", StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidOperationException(
                        Qwen35MobileAssetSet.RuntimePrecisionEnvironmentVariable
                        + " must be INT4 for a native Q4 model, but was " + requestedPrecision + ".");
                }
                manifestName = Qwen35MobileAssetSet.Q4PrecisionManifestFileName;
            }
            else if (string.IsNullOrWhiteSpace(requestedPrecision)
                || string.Equals(requestedPrecision, "INT8", StringComparison.OrdinalIgnoreCase))
            {
                manifestName = Qwen35MobileAssetSet.PrecisionManifestFileName;
            }
            else if (string.Equals(requestedPrecision, "INT4", StringComparison.OrdinalIgnoreCase))
            {
                manifestName = Qwen35MobileAssetSet.Int4GpuPrecisionManifestFileName;
            }
            else
            {
                throw new InvalidOperationException(
                    Qwen35MobileAssetSet.RuntimePrecisionEnvironmentVariable
                    + " must be INT8 or INT4, but was " + requestedPrecision + ".");
            }
            var path = Path.Combine(modelDirectory, manifestName);
            if (!File.Exists(path)) throw new FileNotFoundException("Qwen3.5 mobile precision manifest is missing.", path);
            repro.ApplyModelManifest(AexisModelManifestLoader.LoadFromFile(path));
        }

        public static void ApplyMobileProjectionPrecisionManifest(AexisGraphSession repro, string modelDirectory)
        {
            if (repro == null) throw new ArgumentNullException(nameof(repro));
            var mobile = Qwen35MobileAssetSet.TryLoad(modelDirectory, verifyHashes: false);
            var projectionPath = mobile?.QuantizationBits == 4
                ? Path.Combine(modelDirectory, Qwen35MobileAssetSet.Q4ProjectionPrecisionManifestFileName)
                : null;
            if (!string.IsNullOrWhiteSpace(projectionPath) && File.Exists(projectionPath))
            {
                repro.ApplyModelManifest(AexisModelManifestLoader.LoadFromFile(projectionPath));
                return;
            }

            ApplyMobilePrecisionManifest(repro, modelDirectory);
        }
    }

    public static class Qwen35ModelDirectoryResolver
    {
        public static string Resolve(string directoryOrCollectionRoot, string modelDirectoryName)
        {
            if (string.IsNullOrWhiteSpace(directoryOrCollectionRoot))
                return string.Empty;

            var root = Path.GetFullPath(directoryOrCollectionRoot);
            if (string.IsNullOrWhiteSpace(modelDirectoryName)
                || File.Exists(Path.Combine(root, "model.json")))
            {
                return root;
            }

            var candidate = Path.Combine(root, modelDirectoryName);
            return Directory.Exists(candidate) ? candidate : root;
        }
    }

    internal static class Qwen35RuntimeTuning
    {
        private const long DefaultMobileLoadGcIntervalBytes = 0L;
        // Qwen3.5 emits a short planning prefix before the image description.
        // A complete multi-person response needs room to state the count and
        // enumerate every visible person; 64 tokens cuts that answer off.
        private const int DefaultMobileMaximumNewTokens = 128;
        // 512 patches preserves the Q4 golden first token while reducing the vision
        // context, decoder KV textures, and transient Pack4 allocations on mobile.
        private const int DefaultMobileMaximumVisionPatches = 512;

        public static long ResolveManagedLoadGarbageCollectionIntervalBytes(string modelDirectory)
        {
            if (Qwen35MobileAssetSet.TryLoad(modelDirectory) == null)
                return 0L;

            var configured = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_LOAD_GC_INTERVAL_MB");
            if (long.TryParse(configured, out var megabytes) && megabytes >= 0)
                return checked(Math.Min(megabytes, 4096L) * 1024L * 1024L);
            return DefaultMobileLoadGcIntervalBytes;
        }

        public static int ResolveMaximumNewTokens(string modelDirectory, int requestedTokens)
        {
            var requested = Math.Max(1, Math.Min(requestedTokens, 4096));
            if (Qwen35MobileAssetSet.TryLoad(modelDirectory, verifyHashes: false) == null)
                return requested;

            var configured = ResolvePositiveEnvironmentValue(
                "AIIMAGE_QWEN35_MOBILE_MAX_NEW_TOKENS",
                DefaultMobileMaximumNewTokens,
                128);
            return Math.Min(requested, configured);
        }

        public static int ResolveMaximumVisionPatchCount(string modelDirectory)
        {
            if (Qwen35MobileAssetSet.TryLoad(modelDirectory, verifyHashes: false) == null)
                return int.MaxValue;

            return ResolvePositiveEnvironmentValue(
                "AIIMAGE_QWEN35_MOBILE_MAX_VISION_PATCHES",
                DefaultMobileMaximumVisionPatches,
                49152);
        }

        private static int ResolvePositiveEnvironmentValue(string name, int fallback, int maximum)
        {
            var configured = Environment.GetEnvironmentVariable(name);
            return int.TryParse(configured, out var value) && value > 0
                ? Math.Min(value, maximum)
                : fallback;
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
                if (_current.Position != local)
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
