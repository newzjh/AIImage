#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using Aexis.Ncnn;
using UnityEngine;

// This file is deliberately Editor-only. It is an Oracle/Debug capture adapter,
// never a production runner API or an intermediate-buffer materialization path.
[Serializable]
public sealed class NcnnGoldenTensorObservation
{
    public const string TensorSchema = "aiimage.inference.tensor/v1";

    public string schema_version = TensorSchema;
    public string case_id;
    public string node;
    public string blob;
    public int[] logical_shape;
    public int[] storage_shape;
    public string layout;
    public string dtype;
    public float[] values;
}

[Serializable]
public sealed class NcnnGoldenObservationBundle
{
    public const string BundleSchema = "aiimage.inference.golden-observations/v1";

    public string schema_version = BundleSchema;
    public string case_id;
    public NcnnGoldenTensorObservation[] tensors;
}

public static class NcnnGoldenDebugOracleReadback
{
    public const string Scope = "Debug/Oracle test only - texture-aware readback";

    // The caller must retain each blob with NcnnGraphSession.Infer(..., pinnedNames) before capture.
    public static NcnnGoldenTensorObservation CapturePinnedTexture(
        NcnnGraphSession.InferResult inference,
        string caseId,
        string node,
        string blob,
        int[] expectedStorageShape,
        string layout,
        string dtype)
    {
        if (inference == null)
            throw new ArgumentNullException(nameof(inference));
        if (string.IsNullOrWhiteSpace(caseId))
            throw new ArgumentException("Golden case id is required.", nameof(caseId));
        if (string.IsNullOrWhiteSpace(node))
            throw new ArgumentException("Producer node is required.", nameof(node));
        if (string.IsNullOrWhiteSpace(blob))
            throw new ArgumentException("Pinned blob name is required.", nameof(blob));
        if (expectedStorageShape == null || expectedStorageShape.Length != 5)
            throw new ArgumentException("Storage shape must use [dims,w,h,d,c].", nameof(expectedStorageShape));
        if (!inference.TryGetExistingTexture(blob, out var texture) || texture == null)
            throw new InvalidOperationException("Golden Debug/Oracle capture requires a pinned texture blob: " + blob);
        if (!inference.TryGetLogicalShape(blob, out var dims, out var w, out var h, out var d, out var c))
            throw new InvalidOperationException("Golden Debug/Oracle capture could not resolve logical shape: " + blob);

        ValidateStorage(texture, expectedStorageShape, blob);
        var values = inference.GetExistingTextureData(blob);
        var expectedCount = checked(Mathf.Max(1, w) * Mathf.Max(1, h) * Mathf.Max(1, d) * Mathf.Max(1, c));
        if (values == null || values.Length != expectedCount)
        {
            throw new InvalidOperationException(
                "Golden Debug/Oracle texture readback count mismatch for " + blob
                + " | expected=" + expectedCount
                + " | actual=" + (values == null ? 0 : values.Length));
        }

        return new NcnnGoldenTensorObservation
        {
            case_id = caseId,
            node = node,
            blob = blob,
            logical_shape = new[] { dims, w, h, d, c },
            storage_shape = (int[])expectedStorageShape.Clone(),
            layout = layout ?? string.Empty,
            dtype = dtype ?? string.Empty,
            values = values
        };
    }

    public static void WriteObservationBundle(string outputPath, string caseId, IEnumerable<NcnnGoldenTensorObservation> observations)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("Output path is required.", nameof(outputPath));
        if (string.IsNullOrWhiteSpace(caseId))
            throw new ArgumentException("Golden case id is required.", nameof(caseId));

        var bundle = new NcnnGoldenObservationBundle
        {
            case_id = caseId,
            tensors = observations == null ? Array.Empty<NcnnGoldenTensorObservation>() : new List<NcnnGoldenTensorObservation>(observations).ToArray()
        };
        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllText(outputPath, JsonUtility.ToJson(bundle, true));
    }

    private static void ValidateStorage(RenderTexture texture, int[] storageShape, string blob)
    {
        if (storageShape[0] < 1 || storageShape[0] > 4 || storageShape[1] <= 0 || storageShape[2] <= 0 || storageShape[3] <= 0 || storageShape[4] <= 0)
            throw new ArgumentException("Storage shape contains an invalid dimension.", nameof(storageShape));
        if (texture.width != storageShape[1] || texture.height != storageShape[2])
        {
            throw new InvalidOperationException(
                "Golden Debug/Oracle storage shape mismatch for " + blob
                + " | expected=" + storageShape[1] + "x" + storageShape[2]
                + " | actual=" + texture.width + "x" + texture.height);
        }

        var requiredSlices = storageShape[0] == 4
            ? storageShape[3] * Mathf.CeilToInt(storageShape[4] / 4f)
            : Mathf.CeilToInt(storageShape[4] / 4f);
        if (texture.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray
            && texture.volumeDepth < Mathf.Max(1, requiredSlices))
        {
            throw new InvalidOperationException(
                "Golden Debug/Oracle storage slices mismatch for " + blob
                + " | expected=" + requiredSlices
                + " | actual=" + texture.volumeDepth);
        }
    }
}
#endif
