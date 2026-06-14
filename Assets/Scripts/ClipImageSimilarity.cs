using System;
using System.Collections.Generic;
using System.Linq;

public static class ClipImageSimilarity
{
    public sealed class SimilarImageMatch
    {
        public ClipClassificationCache.CachedClipImageRecord source;
        public ClipClassificationCache.CachedClipImageRecord target;
        public float cosineSimilarity;
    }

    public sealed class SimilarImageCluster
    {
        public int clusterId;
        public List<ClipClassificationCache.CachedClipImageRecord> members = new List<ClipClassificationCache.CachedClipImageRecord>();
    }

    public static float CosineSimilarity(float[] a, float[] b)
    {
        if (a == null || b == null || a.Length == 0 || b.Length == 0 || a.Length != b.Length)
            return 0f;

        var dot = 0f;
        var magA = 0f;
        var magB = 0f;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }

        if (magA <= 1e-12f || magB <= 1e-12f)
            return 0f;

        return dot / (float)(Math.Sqrt(magA) * Math.Sqrt(magB));
    }

    public static SimilarImageMatch FindBestMatch(
        ClipClassificationCache.CachedClipImageRecord source,
        IReadOnlyList<ClipClassificationCache.CachedClipImageRecord> candidates,
        Func<ClipClassificationCache.CachedClipImageRecord, bool> predicate = null)
    {
        if (source == null || candidates == null || source.imageEmbedding == null)
            return null;

        SimilarImageMatch best = null;
        for (var i = 0; i < candidates.Count; i++)
        {
            var target = candidates[i];
            if (target == null || ReferenceEquals(source, target) || target.imageEmbedding == null)
                continue;
            if (predicate != null && !predicate(target))
                continue;

            var similarity = CosineSimilarity(source.imageEmbedding, target.imageEmbedding);
            if (best == null || similarity > best.cosineSimilarity)
            {
                best = new SimilarImageMatch
                {
                    source = source,
                    target = target,
                    cosineSimilarity = similarity
                };
            }
        }

        return best;
    }

    public static List<SimilarImageMatch> FindMutualBestMatches(
        IReadOnlyList<ClipClassificationCache.CachedClipImageRecord> originals,
        IReadOnlyList<ClipClassificationCache.CachedClipImageRecord> edited,
        float minSimilarity = 0.90f)
    {
        var matches = new List<SimilarImageMatch>();
        if (originals == null || edited == null)
            return matches;

        var bestEditedForOriginal = new Dictionary<string, SimilarImageMatch>(StringComparer.Ordinal);
        var bestOriginalForEdited = new Dictionary<string, SimilarImageMatch>(StringComparer.Ordinal);

        for (var i = 0; i < originals.Count; i++)
        {
            var original = originals[i];
            var best = FindBestMatch(original, edited);
            if (best != null && best.cosineSimilarity >= minSimilarity && !string.IsNullOrWhiteSpace(original.key))
                bestEditedForOriginal[original.key] = best;
        }

        for (var i = 0; i < edited.Count; i++)
        {
            var edit = edited[i];
            var best = FindBestMatch(edit, originals);
            if (best != null && best.cosineSimilarity >= minSimilarity && !string.IsNullOrWhiteSpace(edit.key))
                bestOriginalForEdited[edit.key] = best;
        }

        foreach (var pair in bestEditedForOriginal)
        {
            var forward = pair.Value;
            if (forward?.target == null || string.IsNullOrWhiteSpace(forward.target.key))
                continue;

            if (!bestOriginalForEdited.TryGetValue(forward.target.key, out var backward))
                continue;
            if (backward?.target == null || !string.Equals(backward.target.key, forward.source.key, StringComparison.Ordinal))
                continue;

            matches.Add(forward);
        }

        return matches.OrderByDescending(m => m.cosineSimilarity).ToList();
    }

    public static List<SimilarImageCluster> BuildThresholdClusters(
        IReadOnlyList<ClipClassificationCache.CachedClipImageRecord> records,
        float minSimilarity = 0.92f)
    {
        var clusters = new List<SimilarImageCluster>();
        if (records == null || records.Count == 0)
            return clusters;

        var parent = new int[records.Count];
        for (var i = 0; i < parent.Length; i++)
            parent[i] = i;

        for (var i = 0; i < records.Count; i++)
        {
            var a = records[i];
            if (a?.imageEmbedding == null)
                continue;

            for (var j = i + 1; j < records.Count; j++)
            {
                var b = records[j];
                if (b?.imageEmbedding == null)
                    continue;

                if (CosineSimilarity(a.imageEmbedding, b.imageEmbedding) >= minSimilarity)
                    Union(parent, i, j);
            }
        }

        var clusterMap = new Dictionary<int, SimilarImageCluster>();
        for (var i = 0; i < records.Count; i++)
        {
            var root = Find(parent, i);
            if (!clusterMap.TryGetValue(root, out var cluster))
            {
                cluster = new SimilarImageCluster { clusterId = clusterMap.Count + 1 };
                clusterMap[root] = cluster;
                clusters.Add(cluster);
            }

            cluster.members.Add(records[i]);
        }

        return clusters.OrderByDescending(c => c.members.Count).ToList();
    }

    private static int Find(int[] parent, int index)
    {
        while (parent[index] != index)
        {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }

        return index;
    }

    private static void Union(int[] parent, int a, int b)
    {
        var pa = Find(parent, a);
        var pb = Find(parent, b);
        if (pa != pb)
            parent[pb] = pa;
    }
}
