using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;

public struct FaceRegionResult
{
    public Texture2D mask;
    public RectInt faceRect;
    public Vector2[] landmarks;
    public float score;
    public string error;
}

public sealed class NcnnFaceRegionGenerator : MonoBehaviour
{
    private struct FaceProposal
    {
        public Rect rect;
        public float score;
        public Vector2[] landmarks;
    }

    private readonly struct LetterboxInfo
    {
        public readonly Texture2D texture;
        public readonly float scale;
        public readonly int padLeft;
        public readonly int padTop;

        public LetterboxInfo(Texture2D texture, float scale, int padLeft, int padTop)
        {
            this.texture = texture;
            this.scale = scale;
            this.padLeft = padLeft;
            this.padTop = padTop;
        }
    }

    private readonly struct SimilarityTransform
    {
        public readonly float a;
        public readonly float b;
        public readonly float tx;
        public readonly float ty;
        public readonly bool valid;

        public SimilarityTransform(float a, float b, float tx, float ty, bool valid)
        {
            this.a = a;
            this.b = b;
            this.tx = tx;
            this.ty = ty;
            this.valid = valid;
        }

        public Vector2 Transform(Vector2 p)
        {
            return new Vector2(
                a * p.x - b * p.y + tx,
                b * p.x + a * p.y + ty);
        }
    }

    private static readonly Vector2[] CanonicalFivePointTemplate =
    {
        new Vector2(192.98138f, 239.94708f),
        new Vector2(318.90277f, 240.1936f),
        new Vector2(256.63416f, 314.01935f),
        new Vector2(201.26117f, 371.41043f),
        new Vector2(313.08905f, 371.15118f)
    };

    private static readonly int[] YoloV7StrideValues = { 8, 16, 32 };
    private static readonly float[][] YoloV7Anchors =
    {
        new[] { 4f, 5f, 6f, 8f, 10f, 12f },
        new[] { 15f, 19f, 23f, 30f, 39f, 52f },
        new[] { 72f, 97f, 123f, 164f, 209f, 297f }
    };

    public bool enableNcnnFaceRegion = true;
    public string paramRelativePath = "CodeFormer/models/yolov7-lite-e.param";
    public string binRelativePath = "CodeFormer/models/yolov7-lite-e.bin";
    public int inputSize = 640;
    public float probThreshold = 0.5f;
    public float nmsThreshold = 0.65f;
    public float maskRectExpand = 0.18f;
    public float maskSoftness = 0.10f;
    public float faceRectThreshold = 0.18f;

    private NcnnOps _ops;
    private NcnnRepro2 _repro;
    private bool _loaded;

    private void Awake()
    {
        _ops = new NcnnOps();
        _repro = new NcnnRepro2(_ops);
    }

    private void OnDestroy()
    {
        try { _repro?.Dispose(); } catch { }
        _repro = null;
        _ops = null;
        _loaded = false;
    }

    public async UniTask<FaceRegionResult> GenerateAsync(Texture2D src, bool dumpDebug, CancellationToken ct)
    {
        if (!enableNcnnFaceRegion)
            return new FaceRegionResult { error = "NcnnFaceRegion disabled" };
        if (src == null)
            return default;

        await EnsureLoaded(ct);
        if (_repro == null || _repro.Model == null)
            return new FaceRegionResult { error = "Face detector model unavailable" };

        Texture2D letterbox = null;
        RenderTexture inputPack4 = null;
        try
        {
            ct.ThrowIfCancellationRequested();

            var prep = BuildLetterbox(src, Mathf.Max(64, inputSize));
            letterbox = prep.texture;
            if (letterbox == null)
                return new FaceRegionResult { error = "Face detector letterbox build failed" };

            inputPack4 = _repro.RentTempArray(letterbox.width, letterbox.height, 1, RenderTextureFormat.ARGBHalf);
            _ops.PackRgbToPack4(letterbox, 0, 0, 1f, 1f, inputPack4);

            var pinned = new HashSet<string>(StringComparer.Ordinal)
            {
                "stride_8",
                "stride_16",
                "stride_32"
            };

            var proposals = new List<FaceProposal>();
            using (var infer = _repro.Infer(inputPack4, 1, "images", pinned))
            {
                for (var i = 0; i < YoloV7StrideValues.Length; i++)
                {
                    var blobName = "stride_" + YoloV7StrideValues[i].ToString(CultureInfo.InvariantCulture);
                    if (!infer.TryGetLogicalShape(blobName, out var dims, out var outW, out var outH, out _, out var outC))
                        continue;
                    if (dims != 3)
                        continue;

                    var data = infer.GetBufferData(blobName);
                    DecodeYoloV7LiteE(
                        proposals,
                        data,
                        outW,
                        outH,
                        outC,
                        YoloV7StrideValues[i],
                        YoloV7Anchors[i],
                        prep.scale,
                        prep.padLeft,
                        prep.padTop,
                        src.width,
                        src.height,
                        Mathf.Max(0.01f, probThreshold));
                }
            }

            if (proposals.Count == 0)
                return new FaceRegionResult { error = "No face detected" };

            SortProposals(proposals);
            var picked = ApplyNms(proposals, Mathf.Clamp01(nmsThreshold));
            if (picked.Count == 0)
                return new FaceRegionResult { error = "No face left after NMS" };

            FaceProposal best = PickBestFace(proposals, picked);
            var mask = BuildMask(src.width, src.height, best, out var faceRect);
            if (mask == null)
                return new FaceRegionResult { error = "Face mask build failed" };

            return new FaceRegionResult
            {
                mask = mask,
                faceRect = faceRect.width > 0 && faceRect.height > 0
                    ? faceRect
                    : ExpandRect(RoundRect(best.rect), src.width, src.height, 0.12f),
                landmarks = best.landmarks,
                score = best.score
            };
        }
        catch (OperationCanceledException)
        {
            return default;
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            return new FaceRegionResult { error = e.Message };
        }
        finally
        {
            if (letterbox != null)
                Destroy(letterbox);
            if (inputPack4 != null)
                _repro?.ReturnTempArray(inputPack4);
            _repro?.ClearTempPool();
        }
    }

    private async UniTask EnsureLoaded(CancellationToken ct)
    {
        if (_loaded)
            return;

        var paramPath = Path.Combine(Application.streamingAssetsPath, paramRelativePath);
        var binPath = Path.Combine(Application.streamingAssetsPath, binRelativePath);
        if (!File.Exists(paramPath))
            throw new InvalidOperationException("Missing face detector param: " + paramPath);
        if (!File.Exists(binPath))
            throw new InvalidOperationException("Missing face detector bin: " + binPath);

        var paramText = await File.ReadAllTextAsync(paramPath, ct);
        var bytes = await File.ReadAllBytesAsync(binPath, ct);
        using (var ms = new MemoryStream(bytes))
        using (var br = new NcnnBinReader(ms))
        {
            _repro.LoadModel(paramText, br);
        }
        _loaded = true;
    }

    private static LetterboxInfo BuildLetterbox(Texture2D src, int targetSize)
    {
        var srcW = Mathf.Max(1, src.width);
        var srcH = Mathf.Max(1, src.height);
        var scale = srcW > srcH ? (float)targetSize / srcW : (float)targetSize / srcH;
        var resizedW = Mathf.Clamp(Mathf.RoundToInt(srcW * scale), 1, targetSize);
        var resizedH = Mathf.Clamp(Mathf.RoundToInt(srcH * scale), 1, targetSize);
        var padLeft = (targetSize - resizedW) / 2;
        var padTop = (targetSize - resizedH) / 2;

        var srcPixels = src.GetPixels32();
        var dstPixels = new Color32[targetSize * targetSize];
        for (var i = 0; i < dstPixels.Length; i++)
            dstPixels[i] = new Color32(0, 0, 0, 255);

        for (var y = 0; y < resizedH; y++)
        {
            var sy = resizedH > 1 ? (float)y * (srcH - 1) / (resizedH - 1) : 0f;
            var y0 = Mathf.Clamp((int)sy, 0, srcH - 1);
            var y1 = Mathf.Clamp(y0 + 1, 0, srcH - 1);
            var ty = sy - y0;
            for (var x = 0; x < resizedW; x++)
            {
                var sx = resizedW > 1 ? (float)x * (srcW - 1) / (resizedW - 1) : 0f;
                var x0 = Mathf.Clamp((int)sx, 0, srcW - 1);
                var x1 = Mathf.Clamp(x0 + 1, 0, srcW - 1);
                var tx = sx - x0;
                dstPixels[(padTop + y) * targetSize + padLeft + x] = BilinearSample(srcPixels, srcW, srcH, x0, y0, x1, y1, tx, ty);
            }
        }

        var tex = new Texture2D(targetSize, targetSize, TextureFormat.RGBA32, false, true);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        tex.SetPixels32(dstPixels);
        tex.Apply(false, false);
        return new LetterboxInfo(tex, scale, padLeft, padTop);
    }

    private static Color32 BilinearSample(Color32[] src, int srcW, int srcH, int x0, int y0, int x1, int y1, float tx, float ty)
    {
        var c00 = src[y0 * srcW + x0];
        var c10 = src[y0 * srcW + x1];
        var c01 = src[y1 * srcW + x0];
        var c11 = src[y1 * srcW + x1];

        var r0 = Mathf.Lerp(c00.r, c10.r, tx);
        var g0 = Mathf.Lerp(c00.g, c10.g, tx);
        var b0 = Mathf.Lerp(c00.b, c10.b, tx);
        var a0 = Mathf.Lerp(c00.a, c10.a, tx);

        var r1 = Mathf.Lerp(c01.r, c11.r, tx);
        var g1 = Mathf.Lerp(c01.g, c11.g, tx);
        var b1 = Mathf.Lerp(c01.b, c11.b, tx);
        var a1 = Mathf.Lerp(c01.a, c11.a, tx);

        return new Color32(
            (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(r0, r1, ty)), 0, 255),
            (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(g0, g1, ty)), 0, 255),
            (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(b0, b1, ty)), 0, 255),
            (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(a0, a1, ty)), 0, 255));
    }

    private static void DecodeYoloV7LiteE(
        List<FaceProposal> proposals,
        float[] data,
        int featW,
        int featH,
        int featC,
        int stride,
        float[] anchors,
        float scale,
        int padLeft,
        int padTop,
        int imgW,
        int imgH,
        float probThreshold)
    {
        if (data == null || proposals == null || featW <= 0 || featH <= 0 || featC <= 0 || anchors == null || anchors.Length < 6)
            return;

        var numClass = featW - 20;
        if (numClass <= 0)
            return;

        var numAnchors = anchors.Length / 2;
        var numGrid = featH;
        var inputSize = stride * Mathf.RoundToInt(Mathf.Sqrt(numGrid));
        var numGridX = Mathf.Max(1, inputSize / stride);
        var numGridY = Mathf.Max(1, numGrid / numGridX);
        if (numGridX * numGridY != numGrid)
        {
            numGridX = Mathf.Max(1, numGridX);
            numGridY = Mathf.Max(1, numGrid / numGridX);
        }

        for (var q = 0; q < numAnchors; q++)
        {
            var anchorW = anchors[q * 2 + 0];
            var anchorH = anchors[q * 2 + 1];
            var channelBase = q * featH * featW;
            for (var gy = 0; gy < numGridY; gy++)
            {
                for (var gx = 0; gx < numGridX; gx++)
                {
                    var rowIndex = gy * numGridX + gx;
                    var offset = channelBase + rowIndex * featW;

                    var boxConfidence = Sigmoid(data[offset + 4]);
                    if (boxConfidence < probThreshold)
                        continue;

                    var classScore = float.NegativeInfinity;
                    for (var k = 0; k < numClass; k++)
                    {
                        var score = data[offset + 5 + k];
                        if (score > classScore)
                            classScore = score;
                    }

                    var confidence = boxConfidence * Sigmoid(classScore);
                    if (confidence < probThreshold)
                        continue;

                    var dx = Sigmoid(data[offset + 0]);
                    var dy = Sigmoid(data[offset + 1]);
                    var dw = Sigmoid(data[offset + 2]);
                    var dh = Sigmoid(data[offset + 3]);

                    var pbCx = (dx * 2f - 0.5f + gx) * stride;
                    var pbCy = (dy * 2f - 0.5f + gy) * stride;
                    var pbW = Mathf.Pow(dw * 2f, 2f) * anchorW;
                    var pbH = Mathf.Pow(dh * 2f, 2f) * anchorH;

                    var x0 = (pbCx - pbW * 0.5f - padLeft) / scale;
                    var y0 = (pbCy - pbH * 0.5f - padTop) / scale;
                    var x1 = (pbCx + pbW * 0.5f - padLeft) / scale;
                    var y1 = (pbCy + pbH * 0.5f - padTop) / scale;

                    x0 = Mathf.Clamp(x0, 0f, Mathf.Max(0f, imgW - 1f));
                    y0 = Mathf.Clamp(y0, 0f, Mathf.Max(0f, imgH - 1f));
                    x1 = Mathf.Clamp(x1, 0f, Mathf.Max(0f, imgW - 1f));
                    y1 = Mathf.Clamp(y1, 0f, Mathf.Max(0f, imgH - 1f));
                    if (x1 <= x0 || y1 <= y0)
                        continue;

                    var lm = new Vector2[5];
                    for (var l = 0; l < 5; l++)
                    {
                        var lx = ((data[offset + 6 + l * 3] * 2f - 0.5f + gx) * stride - padLeft) / scale;
                        var ly = ((data[offset + 7 + l * 3] * 2f - 0.5f + gy) * stride - padTop) / scale;
                        lm[l] = new Vector2(
                            Mathf.Clamp(lx, 0f, Mathf.Max(0f, imgW - 1f)),
                            Mathf.Clamp(ly, 0f, Mathf.Max(0f, imgH - 1f)));
                    }

                    proposals.Add(new FaceProposal
                    {
                        rect = Rect.MinMaxRect(x0, y0, x1, y1),
                        score = confidence,
                        landmarks = lm
                    });
                }
            }
        }
    }

    private static void SortProposals(List<FaceProposal> proposals)
    {
        proposals.Sort((a, b) => b.score.CompareTo(a.score));
    }

    private static List<int> ApplyNms(List<FaceProposal> proposals, float threshold)
    {
        var picked = new List<int>();
        var areas = new float[proposals.Count];
        for (var i = 0; i < proposals.Count; i++)
            areas[i] = proposals[i].rect.width * proposals[i].rect.height;

        for (var i = 0; i < proposals.Count; i++)
        {
            var keep = true;
            for (var j = 0; j < picked.Count; j++)
            {
                var inter = IntersectionArea(proposals[i].rect, proposals[picked[j]].rect);
                var union = areas[i] + areas[picked[j]] - inter;
                if (union > 0f && inter / union > threshold)
                {
                    keep = false;
                    break;
                }
            }

            if (keep)
                picked.Add(i);
        }

        return picked;
    }

    private static FaceProposal PickBestFace(List<FaceProposal> proposals, List<int> picked)
    {
        var best = proposals[picked[0]];
        var bestValue = ScoreFace(best);
        for (var i = 1; i < picked.Count; i++)
        {
            var candidate = proposals[picked[i]];
            var value = ScoreFace(candidate);
            if (value > bestValue)
            {
                best = candidate;
                bestValue = value;
            }
        }
        return best;
    }

    private Texture2D BuildMask(int width, int height, FaceProposal best, out RectInt faceRect)
    {
        var bytes = new byte[width * height * 2];
        var searchRect = ExpandRect(RoundRect(best.rect), width, height, Mathf.Clamp(maskRectExpand, 0f, 0.6f));
        var transform = SolveSimilarityTransform(best.landmarks, CanonicalFivePointTemplate);
        var threshold = Mathf.Clamp01(faceRectThreshold);
        var softness = Mathf.Clamp(maskSoftness, 0.01f, 0.35f);
        var minX = width;
        var minY = height;
        var maxX = -1;
        var maxY = -1;

        var canonicalCenter = new Vector2(256f, 278f);
        var radiusX = 160f;
        var radiusY = 205f;

        for (var y = searchRect.yMin; y < searchRect.yMax; y++)
        {
            for (var x = searchRect.xMin; x < searchRect.xMax; x++)
            {
                float value;
                if (transform.valid)
                {
                    var q = transform.Transform(new Vector2(x + 0.5f, y + 0.5f));
                    var dx = (q.x - canonicalCenter.x) / radiusX;
                    var dy = (q.y - canonicalCenter.y) / radiusY;
                    var d = Mathf.Sqrt(dx * dx + dy * dy);
                    value = Mathf.Clamp01((1f + softness - d) / (softness * 2f));
                }
                else
                {
                    value = EvaluateFallbackEllipse(best.rect, x + 0.5f, y + 0.5f, softness);
                }

                if (value <= 0f)
                    continue;

                var index = (y * width + x) * 2;
                var half = FloatToHalfBits(value);
                bytes[index + 0] = (byte)(half & 0xFF);
                bytes[index + 1] = (byte)((half >> 8) & 0xFF);

                if (value >= threshold)
                {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }

        faceRect = maxX >= minX && maxY >= minY
            ? new RectInt(minX, minY, maxX - minX + 1, maxY - minY + 1)
            : ExpandRect(RoundRect(best.rect), width, height, 0.12f);

        var tex = new Texture2D(width, height, TextureFormat.RHalf, false, true);
        tex.LoadRawTextureData(bytes);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        tex.name = "NcnnFaceMask";
        return tex;
    }

    private static float EvaluateFallbackEllipse(Rect rect, float x, float y, float softness)
    {
        var cx = rect.center.x;
        var cy = rect.center.y + rect.height * 0.06f;
        var rx = Mathf.Max(1f, rect.width * 0.48f);
        var ry = Mathf.Max(1f, rect.height * 0.60f);
        var dx = (x - cx) / rx;
        var dy = (y - cy) / ry;
        var d = Mathf.Sqrt(dx * dx + dy * dy);
        return Mathf.Clamp01((1f + softness - d) / (softness * 2f));
    }

    private static SimilarityTransform SolveSimilarityTransform(IReadOnlyList<Vector2> src, IReadOnlyList<Vector2> dst)
    {
        if (src == null || dst == null || src.Count < 2 || dst.Count < 2 || src.Count != dst.Count)
            return default;

        var ata = new float[4, 4];
        var atb = new float[4];

        for (var i = 0; i < src.Count; i++)
        {
            var x = src[i].x;
            var y = src[i].y;
            var u = dst[i].x;
            var v = dst[i].y;

            var r0 = new[] { x, -y, 1f, 0f };
            var r1 = new[] { y, x, 0f, 1f };
            AccumulateNormalEquation(ata, atb, r0, u);
            AccumulateNormalEquation(ata, atb, r1, v);
        }

        if (!SolveLinear4x4(ata, atb, out var solution))
            return default;

        return new SimilarityTransform(solution[0], solution[1], solution[2], solution[3], true);
    }

    private static void AccumulateNormalEquation(float[,] ata, float[] atb, float[] row, float value)
    {
        for (var i = 0; i < 4; i++)
        {
            atb[i] += row[i] * value;
            for (var j = 0; j < 4; j++)
                ata[i, j] += row[i] * row[j];
        }
    }

    private static bool SolveLinear4x4(float[,] a, float[] b, out float[] x)
    {
        x = new float[4];
        var m = new float[4, 5];
        for (var r = 0; r < 4; r++)
        {
            for (var c = 0; c < 4; c++)
                m[r, c] = a[r, c];
            m[r, 4] = b[r];
        }

        for (var col = 0; col < 4; col++)
        {
            var pivot = col;
            var pivotAbs = Mathf.Abs(m[pivot, col]);
            for (var row = col + 1; row < 4; row++)
            {
                var v = Mathf.Abs(m[row, col]);
                if (v > pivotAbs)
                {
                    pivot = row;
                    pivotAbs = v;
                }
            }

            if (pivotAbs < 1e-6f)
                return false;

            if (pivot != col)
            {
                for (var c = col; c < 5; c++)
                {
                    var tmp = m[col, c];
                    m[col, c] = m[pivot, c];
                    m[pivot, c] = tmp;
                }
            }

            var inv = 1f / m[col, col];
            for (var c = col; c < 5; c++)
                m[col, c] *= inv;

            for (var row = 0; row < 4; row++)
            {
                if (row == col)
                    continue;
                var factor = m[row, col];
                if (Mathf.Abs(factor) < 1e-6f)
                    continue;
                for (var c = col; c < 5; c++)
                    m[row, c] -= factor * m[col, c];
            }
        }

        for (var i = 0; i < 4; i++)
            x[i] = m[i, 4];
        return true;
    }

    private static float ScoreFace(FaceProposal proposal)
    {
        var area = Mathf.Max(1f, proposal.rect.width * proposal.rect.height);
        return proposal.score * Mathf.Sqrt(area);
    }

    private static RectInt RoundRect(Rect rect)
    {
        var x0 = Mathf.FloorToInt(rect.xMin);
        var y0 = Mathf.FloorToInt(rect.yMin);
        var x1 = Mathf.CeilToInt(rect.xMax);
        var y1 = Mathf.CeilToInt(rect.yMax);
        return new RectInt(x0, y0, Mathf.Max(1, x1 - x0), Mathf.Max(1, y1 - y0));
    }

    private static RectInt ExpandRect(RectInt rect, int width, int height, float expand)
    {
        var dw = Mathf.RoundToInt(rect.width * expand);
        var dh = Mathf.RoundToInt(rect.height * expand);
        var x0 = Mathf.Clamp(rect.x - dw, 0, Mathf.Max(0, width - 1));
        var y0 = Mathf.Clamp(rect.y - dh, 0, Mathf.Max(0, height - 1));
        var x1 = Mathf.Clamp(rect.xMax + dw, 0, width);
        var y1 = Mathf.Clamp(rect.yMax + dh, 0, height);
        return new RectInt(x0, y0, Mathf.Max(1, x1 - x0), Mathf.Max(1, y1 - y0));
    }

    private static float IntersectionArea(Rect a, Rect b)
    {
        var x0 = Mathf.Max(a.xMin, b.xMin);
        var y0 = Mathf.Max(a.yMin, b.yMin);
        var x1 = Mathf.Min(a.xMax, b.xMax);
        var y1 = Mathf.Min(a.yMax, b.yMax);
        return Mathf.Max(0f, x1 - x0) * Mathf.Max(0f, y1 - y0);
    }

    private static float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
    }

    private static ushort FloatToHalfBits(float value)
    {
        var f = BitConverter.SingleToInt32Bits(value);
        var sign = (f >> 16) & 0x8000;
        var mantissa = f & 0x007FFFFF;
        var exp = ((f >> 23) & 0xFF) - 127 + 15;

        if (exp <= 0)
        {
            if (exp < -10)
                return (ushort)sign;
            mantissa = (mantissa | 0x00800000) >> (1 - exp);
            return (ushort)(sign | ((mantissa + 0x00001000) >> 13));
        }

        if (exp >= 31)
            return (ushort)(sign | 0x7C00);

        return (ushort)(sign | (exp << 10) | ((mantissa + 0x00001000) >> 13));
    }
}
