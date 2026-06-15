using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;

// Stores CLIP classification results in one JSON file to avoid PlayerPrefs churn.
public static class ClipClassificationCache
{
    public sealed class CachedClipImageRecord
    {
        public int entryVersion;
        public string key;
        public string identityPath;
        public string filePath;
        public string bestLabel;
        public float bestProbability;
        public float[] imageEmbedding;
        public long updatedUtcTicks;
    }

    public sealed class CachedFileMetadata
    {
        public int entryVersion;
        public string key;
        public string identityPath;
        public string filePath;
        public DateTime? captureTime;
        public string locationText;
        public string cameraText;
        public string apertureText;
        public long updatedUtcTicks;
    }

    [Serializable]
    private sealed class CacheFile
    {
        public int version = CurrentVersion;
        public ClipClassificationCacheEntry[] entries;
    }

    [Serializable]
    private sealed class ClipClassificationCacheEntry
    {
        public int entryVersion;
        public string key;
        public string signature;
        public string identityPath;
        public string filePath;
        public string bestLabel;
        public float bestProbability;
        public ClipLabelScore[] scores;
        public float[] imageEmbedding;
        public bool hasCaptureTime;
        public long captureTimeBinary;
        public string locationText;
        public string cameraText;
        public string apertureText;
        public long updatedUtcTicks;
    }

    private const int CurrentVersion = 2;
    private const int CurrentEntryVersion = 2;
    private const int MaxEntries = 4096;
    private const string FileKeyPrefix = "file|";
    private const string TextureKeyPrefix = "texture|";
    private const string SessionKeyPrefix = "session|";
    private const string CacheDirectoryName = "Clip";
    private const string CacheFileName = "clip_classification_cache.json";

    private static readonly object Sync = new object();
    private static readonly Dictionary<string, ClipClassificationCacheEntry> Entries = new Dictionary<string, ClipClassificationCacheEntry>(StringComparer.Ordinal);

    private static bool _loaded;
    private static bool _dirty;
    private static bool _saveRunning;

    public static bool TryGet(ClipNcnnReproRunner runner, Texture2D texture, string filePath, bool preferFileIdentity, out ClipClassificationResult result)
    {
        if (preferFileIdentity && TryGetForFile(runner, filePath, out result))
            return true;

        return TryGetForTexture(runner, texture, out result);
    }

    public static bool TryGetForFile(ClipNcnnReproRunner runner, string filePath, out ClipClassificationResult result)
    {
        result = default;
        if (runner == null)
            return false;

        var key = BuildFileKey(filePath, out _);
        if (string.IsNullOrWhiteSpace(key))
            return false;

        return TryGetInternal(key, runner.ClassificationCacheSignature, out result);
    }

    public static bool TryGetMetadataForFile(string filePath, out CachedFileMetadata metadata)
    {
        metadata = null;

        var key = BuildFileKey(filePath, out _);
        if (string.IsNullOrWhiteSpace(key))
            return false;

        EnsureLoaded();
        lock (Sync)
        {
            if (!Entries.TryGetValue(key, out var entry) || !HasMetadata(entry))
                return false;

            metadata = BuildCachedMetadata(entry);
            return metadata != null;
        }
    }

    public static void StoreFileMetadata(
        string filePath,
        DateTime? captureTime,
        string locationText,
        string cameraText,
        string apertureText)
    {
        if (!captureTime.HasValue &&
            string.IsNullOrWhiteSpace(locationText) &&
            string.IsNullOrWhiteSpace(cameraText) &&
            string.IsNullOrWhiteSpace(apertureText))
        {
            return;
        }

        var key = BuildFileKey(filePath, out var normalizedPath);
        if (string.IsNullOrWhiteSpace(key) || string.IsNullOrWhiteSpace(normalizedPath))
            return;

        EnsureLoaded();

        var startSaveLoop = false;
        lock (Sync)
        {
            RemoveStaleEntriesForIdentityNoLock(key, normalizedPath);

            Entries.TryGetValue(key, out var entry);
            entry ??= new ClipClassificationCacheEntry
            {
                key = key
            };

            var changed = UpdateSharedEntryFields(entry, key, normalizedPath, normalizedPath);
            changed |= UpdateMetadataFields(entry, captureTime, locationText, cameraText, apertureText);
            if (!changed)
                return;

            entry.updatedUtcTicks = DateTime.UtcNow.Ticks;
            Entries[key] = entry;

            PruneIfNeededNoLock();

            _dirty = true;
            if (_saveRunning)
                return;

            _saveRunning = true;
            startSaveLoop = true;
        }

        if (startSaveLoop)
            PersistDirtyAsync().Forget();
    }

    public static bool NeedsEmbeddingUpgradeForFile(ClipNcnnReproRunner runner, string filePath)
    {
        if (runner == null)
            return false;

        var key = BuildFileKey(filePath, out _);
        if (string.IsNullOrWhiteSpace(key))
            return false;

        EnsureLoaded();
        var signature = runner.ClassificationCacheSignature;
        lock (Sync)
        {
            return Entries.TryGetValue(key, out var entry) &&
                   entry != null &&
                   string.Equals(entry.signature, signature, StringComparison.Ordinal) &&
                   (entry.imageEmbedding == null || entry.imageEmbedding.Length == 0);
        }
    }

    public static bool TryGetForTexture(ClipNcnnReproRunner runner, Texture2D texture, out ClipClassificationResult result)
    {
        result = default;
        if (runner == null || texture == null)
            return false;

        var key = BuildTextureKey(texture);
        if (string.IsNullOrWhiteSpace(key))
            return false;

        return TryGetInternal(key, runner.ClassificationCacheSignature, out result);
    }

    public static void Store(
        ClipNcnnReproRunner runner,
        ClipClassificationResult result,
        Texture2D texture,
        string filePath,
        bool preferFileIdentity)
    {
        if (runner == null)
            return;

        string key = null;
        string identityPath = null;

        if (preferFileIdentity)
            key = BuildFileKey(filePath, out identityPath);

        if (string.IsNullOrWhiteSpace(key))
            key = BuildTextureKey(texture);

        if (string.IsNullOrWhiteSpace(key))
            return;

        StoreSuccessfulResult(key, identityPath, filePath, runner.ClassificationCacheSignature, result);
    }

    public static List<CachedClipImageRecord> GetAllImageRecords(ClipNcnnReproRunner runner)
    {
        var list = new List<CachedClipImageRecord>();
        if (runner == null)
            return list;

        EnsureLoaded();
        var signature = runner.ClassificationCacheSignature;
        lock (Sync)
        {
            foreach (var entry in Entries.Values)
            {
                if (entry == null ||
                    string.IsNullOrWhiteSpace(entry.signature) ||
                    !string.Equals(entry.signature, signature, StringComparison.Ordinal) ||
                    entry.imageEmbedding == null ||
                    entry.imageEmbedding.Length == 0)
                {
                    continue;
                }

                list.Add(new CachedClipImageRecord
                {
                    entryVersion = GetEffectiveEntryVersion(entry),
                    key = entry.key,
                    identityPath = entry.identityPath,
                    filePath = entry.filePath,
                    bestLabel = entry.bestLabel,
                    bestProbability = entry.bestProbability,
                    updatedUtcTicks = entry.updatedUtcTicks,
                    imageEmbedding = CloneEmbedding(entry.imageEmbedding)
                });
            }
        }

        return list;
    }

    public static bool TryGetImageRecordForFile(ClipNcnnReproRunner runner, string filePath, out CachedClipImageRecord record)
    {
        record = null;
        if (runner == null)
            return false;

        var key = BuildFileKey(filePath, out _);
        if (string.IsNullOrWhiteSpace(key))
            return false;

        EnsureLoaded();
        var signature = runner.ClassificationCacheSignature;
        lock (Sync)
        {
            if (!Entries.TryGetValue(key, out var entry) ||
                entry == null ||
                !string.Equals(entry.signature, signature, StringComparison.Ordinal) ||
                entry.imageEmbedding == null ||
                entry.imageEmbedding.Length == 0)
            {
                return false;
            }

            record = new CachedClipImageRecord
            {
                entryVersion = GetEffectiveEntryVersion(entry),
                key = entry.key,
                identityPath = entry.identityPath,
                filePath = entry.filePath,
                bestLabel = entry.bestLabel,
                bestProbability = entry.bestProbability,
                updatedUtcTicks = entry.updatedUtcTicks,
                imageEmbedding = CloneEmbedding(entry.imageEmbedding)
            };
            return true;
        }
    }

    public static UniTask<ClipClassificationResult> GetOrClassifyAsync(
        ClipNcnnReproRunner runner,
        Texture2D texture,
        string filePath,
        bool preferFileIdentity,
        CancellationToken cancellationToken)
    {
        if (preferFileIdentity)
            return GetOrClassifyForFileAsync(runner, texture, filePath, cancellationToken);

        return GetOrClassifyForTextureAsync(runner, texture, cancellationToken);
    }

    public static UniTask<ClipClassificationResult> GetOrClassifyForFileAsync(
        ClipNcnnReproRunner runner,
        Texture2D texture,
        string filePath,
        CancellationToken cancellationToken)
    {
        return GetOrClassifyForFileAsync(runner, texture, filePath, cancellationToken, false);
    }

    public static UniTask<ClipClassificationResult> GetOrClassifyForFileAsync(
        ClipNcnnReproRunner runner,
        Texture2D texture,
        string filePath,
        CancellationToken cancellationToken,
        bool requireEmbedding)
    {
        if (runner == null || texture == null)
            return UniTask.FromResult(default(ClipClassificationResult));

        var key = BuildFileKey(filePath, out var normalizedPath);
        if (string.IsNullOrWhiteSpace(key))
            return GetOrClassifyForTextureAsync(runner, texture, cancellationToken);

        return GetOrClassifyInternalAsync(
            runner,
            texture,
            key,
            normalizedPath,
            runner.ClassificationCacheSignature,
            cancellationToken,
            requireEmbedding);
    }

    public static UniTask<ClipClassificationResult> GetOrClassifyForTextureAsync(
        ClipNcnnReproRunner runner,
        Texture2D texture,
        CancellationToken cancellationToken)
    {
        if (runner == null || texture == null)
            return UniTask.FromResult(default(ClipClassificationResult));

        var key = BuildTextureKey(texture);
        if (string.IsNullOrWhiteSpace(key))
            return UniTask.FromResult(default(ClipClassificationResult));

        return GetOrClassifyInternalAsync(
            runner,
            texture,
            key,
            null,
            runner.ClassificationCacheSignature,
            cancellationToken);
    }

    private static async UniTask<ClipClassificationResult> GetOrClassifyInternalAsync(
        ClipNcnnReproRunner runner,
        Texture2D texture,
        string key,
        string identityPath,
        string signature,
        CancellationToken cancellationToken,
        bool requireEmbedding = false)
    {
        if (TryGetInternal(key, signature, out var cached) &&
            (!requireEmbedding || (cached.imageEmbedding != null && cached.imageEmbedding.Length > 0)))
        {
            return cached;
        }

        var result = await runner.ProcessAsync(texture, cancellationToken);
        StoreSuccessfulResult(key, identityPath, identityPath, signature, result);
        return result;
    }

    private static bool TryGetInternal(string key, string signature, out ClipClassificationResult result)
    {
        result = default;
        if (string.IsNullOrWhiteSpace(key) || string.IsNullOrWhiteSpace(signature))
            return false;

        EnsureLoaded();
        lock (Sync)
        {
            if (!Entries.TryGetValue(key, out var entry) ||
                entry == null ||
                !string.Equals(entry.signature, signature, StringComparison.Ordinal))
            {
                return false;
            }

            result = new ClipClassificationResult
            {
                bestLabel = entry.bestLabel,
                bestProbability = entry.bestProbability,
                scores = CloneScores(entry.scores),
                imageEmbedding = CloneEmbedding(entry.imageEmbedding),
                error = null,
                elapsedMs = 0
            };
            return true;
        }
    }

    private static void StoreSuccessfulResult(string key, string identityPath, string filePath, string signature, ClipClassificationResult result)
    {
        if (!IsSuccessful(result) || string.IsNullOrWhiteSpace(key) || string.IsNullOrWhiteSpace(signature))
            return;

        EnsureLoaded();

        var shouldPersist = ShouldPersistKey(key);
        var startSaveLoop = false;
        lock (Sync)
        {
            RemoveStaleEntriesForIdentityNoLock(key, identityPath);

            Entries.TryGetValue(key, out var entry);
            entry ??= new ClipClassificationCacheEntry
            {
                key = key
            };

            UpdateSharedEntryFields(entry, key, identityPath, filePath);
            entry.signature = signature;
            entry.bestLabel = result.bestLabel;
            entry.bestProbability = result.bestProbability;
            entry.scores = CloneScores(result.scores);
            entry.imageEmbedding = CloneEmbedding(result.imageEmbedding);
            entry.updatedUtcTicks = DateTime.UtcNow.Ticks;
            Entries[key] = entry;

            PruneIfNeededNoLock();

            if (!shouldPersist)
                return;

            _dirty = true;
            if (_saveRunning)
                return;

            _saveRunning = true;
            startSaveLoop = true;
        }

        if (startSaveLoop)
            PersistDirtyAsync().Forget();
    }

    private static void EnsureLoaded()
    {
        lock (Sync)
        {
            if (_loaded)
                return;

            _loaded = true;
            var path = GetCacheFilePath();
            if (!File.Exists(path))
                return;

            try
            {
                var payload = File.ReadAllText(path);
                if (string.IsNullOrWhiteSpace(payload))
                    return;

                var cache = JsonUtility.FromJson<CacheFile>(payload);
                if (cache?.entries == null)
                    return;

                Entries.Clear();
                for (var i = 0; i < cache.entries.Length; i++)
                {
                    var entry = cache.entries[i];
                    if (entry == null || string.IsNullOrWhiteSpace(entry.key))
                        continue;

                    if (entry.entryVersion <= 0)
                        entry.entryVersion = 1;
                    Entries[entry.key] = entry;
                }

                PruneIfNeededNoLock();
            }
            catch (Exception e)
            {
                Debug.LogWarning("[CLIP-CACHE] Failed to load classification cache: " + e.Message);
            }
        }
    }

    private static async UniTaskVoid PersistDirtyAsync()
    {
        while (true)
        {
            string json;
            string path;
            lock (Sync)
            {
                if (!_dirty)
                {
                    _saveRunning = false;
                    return;
                }

                _dirty = false;
                path = GetCacheFilePath();
                json = BuildJsonNoLock();
            }

            try
            {
                var directory = Path.GetDirectoryName(path);
                if (!string.IsNullOrWhiteSpace(directory) && !Directory.Exists(directory))
                    Directory.CreateDirectory(directory);

                await UniTask.RunOnThreadPool(() => File.WriteAllText(path, json ?? string.Empty, Encoding.UTF8));
            }
            catch (Exception e)
            {
                Debug.LogWarning("[CLIP-CACHE] Failed to persist classification cache: " + e.Message);
                lock (Sync)
                {
                    _dirty = true;
                }

                await UniTask.DelayFrame(1);
            }
        }
    }

    private static string BuildJsonNoLock()
    {
        var list = new List<ClipClassificationCacheEntry>(Entries.Count);
        foreach (var entry in Entries.Values)
        {
            if (entry == null || !ShouldPersistKey(entry.key))
                continue;

            list.Add(entry);
        }

        list.Sort((a, b) => b.updatedUtcTicks.CompareTo(a.updatedUtcTicks));
        if (list.Count > MaxEntries)
            list.RemoveRange(MaxEntries, list.Count - MaxEntries);

        var cache = new CacheFile
        {
            version = CurrentVersion,
            entries = list.ToArray()
        };
        return JsonUtility.ToJson(cache);
    }

    private static void PruneIfNeededNoLock()
    {
        if (Entries.Count <= MaxEntries)
            return;

        var list = new List<ClipClassificationCacheEntry>(Entries.Values);
        list.Sort((a, b) => b.updatedUtcTicks.CompareTo(a.updatedUtcTicks));

        var keep = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < list.Count && i < MaxEntries; i++)
        {
            if (!string.IsNullOrWhiteSpace(list[i]?.key))
                keep.Add(list[i].key);
        }

        var staleKeys = new List<string>();
        foreach (var key in Entries.Keys)
        {
            if (!keep.Contains(key))
                staleKeys.Add(key);
        }

        for (var i = 0; i < staleKeys.Count; i++)
            Entries.Remove(staleKeys[i]);
    }

    private static bool IsSuccessful(ClipClassificationResult result)
    {
        return string.IsNullOrWhiteSpace(result.error) &&
               (!string.IsNullOrWhiteSpace(result.bestLabel) || (result.scores != null && result.scores.Length > 0));
    }

    private static ClipLabelScore[] CloneScores(ClipLabelScore[] scores)
    {
        if (scores == null || scores.Length == 0)
            return Array.Empty<ClipLabelScore>();

        var clone = new ClipLabelScore[scores.Length];
        Array.Copy(scores, clone, scores.Length);
        return clone;
    }

    private static float[] CloneEmbedding(float[] embedding)
    {
        if (embedding == null || embedding.Length == 0)
            return Array.Empty<float>();

        var clone = new float[embedding.Length];
        Array.Copy(embedding, clone, embedding.Length);
        return clone;
    }

    private static CachedFileMetadata BuildCachedMetadata(ClipClassificationCacheEntry entry)
    {
        if (!HasMetadata(entry))
            return null;

        return new CachedFileMetadata
        {
            entryVersion = GetEffectiveEntryVersion(entry),
            key = entry.key,
            identityPath = entry.identityPath,
            filePath = entry.filePath,
            captureTime = entry.hasCaptureTime ? DateTime.FromBinary(entry.captureTimeBinary) : (DateTime?)null,
            locationText = entry.locationText,
            cameraText = entry.cameraText,
            apertureText = entry.apertureText,
            updatedUtcTicks = entry.updatedUtcTicks
        };
    }

    private static bool HasMetadata(ClipClassificationCacheEntry entry)
    {
        return entry != null &&
               (entry.hasCaptureTime ||
                !string.IsNullOrWhiteSpace(entry.locationText) ||
                !string.IsNullOrWhiteSpace(entry.cameraText) ||
                !string.IsNullOrWhiteSpace(entry.apertureText));
    }

    private static int GetEffectiveEntryVersion(ClipClassificationCacheEntry entry)
    {
        return entry == null || entry.entryVersion <= 0 ? 1 : entry.entryVersion;
    }

    private static bool UpdateSharedEntryFields(
        ClipClassificationCacheEntry entry,
        string key,
        string identityPath,
        string filePath)
    {
        if (entry == null)
            return false;

        var normalizedFilePath = NormalizePath(filePath);
        var changed = entry.entryVersion != CurrentEntryVersion;
        changed |= !string.Equals(entry.key, key, StringComparison.Ordinal);
        changed |= !string.Equals(entry.identityPath, identityPath, StringComparison.OrdinalIgnoreCase);
        changed |= !string.Equals(entry.filePath, normalizedFilePath, StringComparison.OrdinalIgnoreCase);

        entry.entryVersion = CurrentEntryVersion;
        entry.key = key;
        entry.identityPath = identityPath;
        entry.filePath = normalizedFilePath;
        return changed;
    }

    private static bool UpdateMetadataFields(
        ClipClassificationCacheEntry entry,
        DateTime? captureTime,
        string locationText,
        string cameraText,
        string apertureText)
    {
        if (entry == null)
            return false;

        var normalizedLocation = NormalizeMetadataText(locationText);
        var normalizedCamera = NormalizeMetadataText(cameraText);
        var normalizedAperture = NormalizeMetadataText(apertureText);
        var hasCaptureTime = captureTime.HasValue;
        var captureTimeBinary = hasCaptureTime ? captureTime.Value.ToBinary() : 0L;

        var changed = entry.hasCaptureTime != hasCaptureTime ||
                      entry.captureTimeBinary != captureTimeBinary ||
                      !string.Equals(entry.locationText, normalizedLocation, StringComparison.Ordinal) ||
                      !string.Equals(entry.cameraText, normalizedCamera, StringComparison.Ordinal) ||
                      !string.Equals(entry.apertureText, normalizedAperture, StringComparison.Ordinal);

        entry.hasCaptureTime = hasCaptureTime;
        entry.captureTimeBinary = captureTimeBinary;
        entry.locationText = normalizedLocation;
        entry.cameraText = normalizedCamera;
        entry.apertureText = normalizedAperture;
        return changed;
    }

    private static string NormalizeMetadataText(string text)
    {
        return string.IsNullOrWhiteSpace(text) ? null : text.Trim();
    }

    private static void RemoveStaleEntriesForIdentityNoLock(string key, string identityPath)
    {
        if (string.IsNullOrWhiteSpace(identityPath))
            return;

        var staleKeys = new List<string>();
        foreach (var pair in Entries)
        {
            if (!string.Equals(pair.Key, key, StringComparison.Ordinal) &&
                string.Equals(pair.Value?.identityPath, identityPath, StringComparison.OrdinalIgnoreCase))
            {
                staleKeys.Add(pair.Key);
            }
        }

        for (var i = 0; i < staleKeys.Count; i++)
            Entries.Remove(staleKeys[i]);
    }

    private static string BuildFileKey(string filePath, out string normalizedPath)
    {
        normalizedPath = NormalizePath(filePath);
        if (string.IsNullOrWhiteSpace(normalizedPath))
            return null;

        FileInfo info;
        try
        {
            info = new FileInfo(normalizedPath);
            if (!info.Exists)
                return null;
        }
        catch
        {
            return null;
        }

        return FileKeyPrefix
            + normalizedPath
            + "|"
            + info.Length.ToString(CultureInfo.InvariantCulture)
            + "|"
            + info.LastWriteTimeUtc.Ticks.ToString(CultureInfo.InvariantCulture);
    }

    private static string BuildTextureKey(Texture2D texture)
    {
        if (texture == null)
            return null;

        try
        {
            var hash = ComputeTextureHash(texture);
            if (!string.IsNullOrWhiteSpace(hash))
                return TextureKeyPrefix + hash;
        }
        catch
        {
        }

        return SessionKeyPrefix + texture.GetInstanceID().ToString(CultureInfo.InvariantCulture);
    }

    private static string ComputeTextureHash(Texture2D texture)
    {
        Texture2D readable = null;
        try
        {
            readable = texture.isReadable ? texture : CreateReadableCopy(texture);
            if (readable == null)
                return null;

            var raw = readable.GetRawTextureData();
            var bytes = raw;
            using (var sha = SHA256.Create())
            {
                var header = Encoding.UTF8.GetBytes(
                    readable.width.ToString(CultureInfo.InvariantCulture)
                    + "x"
                    + readable.height.ToString(CultureInfo.InvariantCulture)
                    + "|"
                    + readable.format
                    + "|");

                sha.TransformBlock(header, 0, header.Length, null, 0);
                if (bytes.Length > 0)
                    sha.TransformBlock(bytes, 0, bytes.Length, null, 0);
                sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
                return BytesToHex(sha.Hash);
            }
        }
        finally
        {
            if (readable != null && !ReferenceEquals(readable, texture))
                UnityEngine.Object.Destroy(readable);
        }
    }

    private static Texture2D CreateReadableCopy(Texture2D texture)
    {
        if (texture == null)
            return null;

        var rt = RenderTexture.GetTemporary(texture.width, texture.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        var previous = RenderTexture.active;
        try
        {
            Graphics.Blit(texture, rt);
            RenderTexture.active = rt;
            var copy = new Texture2D(texture.width, texture.height, TextureFormat.RGBA32, false);
            copy.ReadPixels(new Rect(0f, 0f, texture.width, texture.height), 0, 0);
            copy.Apply(false, false);
            copy.name = texture.name;
            return copy;
        }
        finally
        {
            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(rt);
        }
    }

    private static bool ShouldPersistKey(string key)
    {
        return !string.IsNullOrWhiteSpace(key) &&
               !key.StartsWith(SessionKeyPrefix, StringComparison.Ordinal);
    }

    private static string NormalizePath(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return null;

        try
        {
            return Path.GetFullPath(filePath);
        }
        catch
        {
            return null;
        }
    }

    private static string GetCacheFilePath()
    {
        return Path.Combine(Application.persistentDataPath, CacheDirectoryName, CacheFileName);
    }

    private static string BytesToHex(byte[] bytes)
    {
        if (bytes == null || bytes.Length == 0)
            return string.Empty;

        var sb = new StringBuilder(bytes.Length * 2);
        for (var i = 0; i < bytes.Length; i++)
            sb.Append(bytes[i].ToString("x2", CultureInfo.InvariantCulture));
        return sb.ToString();
    }
}
