using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Diagnostics;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;

public struct FaceRegionResult
{
    public Texture2D mask;
    public RectInt faceRect;
    public Vector2[] landmarks;
    public float score;
    public FaceRegionFace[] faces;
    public string dumpDir;
    public string error;
}

public struct FaceRegionFace
{
    public Rect rect;
    public RectInt rectInt;
    public Vector2[] landmarks;
    public float score;
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

        public bool TryInverseTransform(Vector2 p, out Vector2 result)
        {
            var det = a * a + b * b;
            if (!valid || det < 1e-8f)
            {
                result = default;
                return false;
            }

            var dx = p.x - tx;
            var dy = p.y - ty;
            var inv = 1f / det;
            result = new Vector2(
                (a * dx + b * dy) * inv,
                (-b * dx + a * dy) * inv);
            return true;
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

    private static readonly Vector2 CanonicalFaceCenter = new Vector2(256f, 278f);
    private const float CanonicalFaceRadiusX = 160f;
    private const float CanonicalFaceRadiusY = 205f;

    private static readonly int[] YoloV7StrideValues = { 8, 16, 32 };
    private static readonly float[][] YoloV7Anchors =
    {
        new[] { 4f, 5f, 6f, 8f, 10f, 12f },
        new[] { 15f, 19f, 23f, 30f, 39f, 52f },
        new[] { 72f, 97f, 123f, 164f, 209f, 297f }
    };

    public bool enableNcnnFaceRegion = true;
    public bool preferTexturePathForFaceDetector = true;
    public string paramRelativePath = "CodeFormer/models/yolov7-lite-e.param";
    public string binRelativePath = "CodeFormer/models/yolov7-lite-e.bin";
    public int inputSize = 640;
    public float probThreshold = 0.5f;
    public float nmsThreshold = 0.65f;
    public float maskRectExpand = 0.18f;
    public float maskSoftness = 0.10f;
    public float faceRectThreshold = 0.18f;
    public bool autoOpenDumpDir = false;
    public bool enableDetailedProposalDump = true;
    public bool useArgbFloatForDetector = true;
    public int maxDetectedFaces = 5;
    [Range(0.05f, 0.95f)] public float maxFaceAreaRatio = 0.45f;
    [Range(0.2f, 3f)] public float maxFaceAspectRatio = 1.6f;

    private NcnnOps _ops;
    private NcnnRepro2 _repro;
    private bool _loaded;

    private void Awake()
    {
        EnsureRuntimeObjects();
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
        EnsureRuntimeObjects();

        await EnsureLoaded(ct);
        if (_repro == null || _repro.Model == null)
            return new FaceRegionResult { error = "Face detector model unavailable" };

        Texture2D letterbox = null;
        RenderTexture inputPack4 = null;
        string dumpDir = null;
        List<string> debugLines = null;
        try
        {
            ct.ThrowIfCancellationRequested();
            NcnnGpuResourceTracker.Enabled = dumpDebug;
            if (dumpDebug)
                NcnnGpuResourceTracker.Reset("NcnnFaceRegionGenerator");

            var prep = BuildLetterbox(src, Mathf.Max(64, inputSize));
            letterbox = prep.texture;
            if (letterbox == null)
                return new FaceRegionResult { error = "Face detector letterbox build failed" };

            if (dumpDebug && enableDetailedProposalDump)
            {
                debugLines = new List<string>(128)
                {
                    "src=" + src.width + "x" + src.height,
                    "letterbox=" + letterbox.width + "x" + letterbox.height,
                    "scale=" + prep.scale.ToString("F6", CultureInfo.InvariantCulture),
                    "padLeft=" + prep.padLeft,
                    "padTop=" + prep.padTop,
                    "probThreshold=" + probThreshold.ToString("F4", CultureInfo.InvariantCulture),
                    "nmsThreshold=" + nmsThreshold.ToString("F4", CultureInfo.InvariantCulture)
                };
                _repro.DebugCompareTextureLayers = new HashSet<string>(StringComparer.Ordinal)
                {
                    "Conv_0",
                    "Conv_3",
                    "Conv_8",
                    "Conv_92",
                    "Conv_281",
                    "Conv_338",
                    "Conv_347",
                    "Conv_356",
                    "Conv_338",
                    "Conv_361",
                    "Conv_365",
                    "Conv_385",
                    "Conv_389",
                    "Conv_409",
                    "Conv_413",
                    "Conv_326",
                    "Conv_336",
                    "Conv_340",
                    "Conv_345",
                    "Conv_349",
                    "Conv_354"
                };
                _repro.DebugLog = line =>
                {
                    if (debugLines.Count < 512)
                        debugLines.Add(line);
                };
            }
            else
            {
                _repro.DebugCompareTextureLayers = null;
                _repro.DebugLog = null;
            }

            inputPack4 = _repro.RentTempArray(letterbox.width, letterbox.height, 1, useArgbFloatForDetector ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf);
            // Texture2D<float4> sampling from RGBA32 already yields normalized 0..1 values.
            // _ScaleX/_ScaleY here are spatial sampling scale, not color normalization.
            _ops.PackRgbToPack4(letterbox, 0, 0, 1f, 1f, inputPack4);

            var pinned = new HashSet<string>(StringComparer.Ordinal)
            {
                "stride_8",
                "stride_16",
                "stride_32",
                "842",
                "870",
                "898",
                "817",
                "818",
                "819",
                "821",
                "822",
                "823",
                "824",
                "393",
                "394",
                "464",
                "452",
                "463",
                "930",
                "469",
                "492",
                "497",
                "771",
                "774",
                "775",
                "777",
                "782",
                "783",
                "786",
                "789",
                "799",
                "802",
                "812",
                "815",
                "845",
                "846",
                "847",
                "849",
                "850",
                "851",
                "852",
                "873",
                "874",
                "875",
                "877",
                "878",
                "879",
                "880"
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
                    var strideProposalStart = proposals.Count;
                    DecodeYoloV7LiteE(
                        proposals,
                        data,
                        outW,
                        outH,
                        outC,
                        YoloV7StrideValues[i],
                        YoloV7Anchors[i],
                        letterbox.width,
                        letterbox.height,
                        prep.scale,
                        prep.padLeft,
                        prep.padTop,
                        src.width,
                        src.height,
                        Mathf.Max(0.01f, probThreshold));

                    if (debugLines != null)
                    {
                        var added = proposals.Count - strideProposalStart;
                        debugLines.Add(blobName
                            + " | logical_w=" + outW
                            + " | logical_h=" + outH
                            + " | logical_c=" + outC
                            + " | proposals=" + added);
                    }
                }

                if (debugLines != null)
                {
                    AppendBlobPreview(infer, debugLines, "817", 16);
                    AppendBlobPreview(infer, debugLines, "818", 16);
                    AppendBlobPreview(infer, debugLines, "819", 16);
                    AppendBlobPreview(infer, debugLines, "821", 16);
                    AppendBlobPreview(infer, debugLines, "822", 16);
                    AppendBlobPreview(infer, debugLines, "823", 16);
                    AppendBlobPreview(infer, debugLines, "824", 16);
                    AppendBlobPreview(infer, debugLines, "393", 16);
                    AppendBlobPreview(infer, debugLines, "394", 16);
                    AppendBlobPreview(infer, debugLines, "464", 16);
                    AppendBlobPreview(infer, debugLines, "452", 16);
                    AppendBlobPreview(infer, debugLines, "463", 16);
                    AppendBlobPreview(infer, debugLines, "930", 16);
                    AppendBlobPreview(infer, debugLines, "469", 16);
                    AppendBlobPreview(infer, debugLines, "492", 16);
                    AppendBlobPreview(infer, debugLines, "497", 16);
                    AppendBlobPreview(infer, debugLines, "771", 16);
                    AppendBlobPreview(infer, debugLines, "774", 16);
                    AppendBlobPreview(infer, debugLines, "775", 16);
                    AppendBlobPreview(infer, debugLines, "777", 16);
                    AppendBlobPreview(infer, debugLines, "782", 16);
                    AppendBlobPreview(infer, debugLines, "783", 16);
                    AppendBlobPreview(infer, debugLines, "786", 16);
                    AppendBlobPreview(infer, debugLines, "789", 16);
                    AppendBlobPreview(infer, debugLines, "799", 16);
                    AppendBlobPreview(infer, debugLines, "802", 16);
                    AppendBlobPreview(infer, debugLines, "812", 16);
                    AppendBlobPreview(infer, debugLines, "815", 16);
                    AppendBlobPreview(infer, debugLines, "842", 16);
                    AppendBlobPreview(infer, debugLines, "845", 16);
                    AppendBlobPreview(infer, debugLines, "846", 16);
                    AppendBlobPreview(infer, debugLines, "847", 16);
                    AppendBlobPreview(infer, debugLines, "849", 16);
                    AppendBlobPreview(infer, debugLines, "850", 16);
                    AppendBlobPreview(infer, debugLines, "851", 16);
                    AppendBlobPreview(infer, debugLines, "852", 16);
                    AppendBlobPreview(infer, debugLines, "870", 16);
                    AppendBlobPreview(infer, debugLines, "873", 16);
                    AppendBlobPreview(infer, debugLines, "874", 16);
                    AppendBlobPreview(infer, debugLines, "875", 16);
                    AppendBlobPreview(infer, debugLines, "877", 16);
                    AppendBlobPreview(infer, debugLines, "878", 16);
                    AppendBlobPreview(infer, debugLines, "879", 16);
                    AppendBlobPreview(infer, debugLines, "880", 16);
                    AppendBlobPreview(infer, debugLines, "898", 16);
                    AppendBlobPreview(infer, debugLines, "stride_8", 16);
                    AppendBlobPreview(infer, debugLines, "stride_16", 16);
                    AppendBlobPreview(infer, debugLines, "stride_32", 16);
                }
            }

            if (proposals.Count == 0)
                return new FaceRegionResult { error = "No face detected" };

            SortProposals(proposals);
            var picked = ApplyNms(proposals, Mathf.Clamp01(nmsThreshold));
            if (picked.Count == 0)
                return new FaceRegionResult { error = "No face left after NMS" };

            var faces = BuildFaces(proposals, picked, src.width, src.height, Mathf.Max(1, maxDetectedFaces));
            if (faces == null || faces.Length == 0)
                return new FaceRegionResult { error = "No face left after postprocess" };

            var primary = faces[0];
            var primaryProposal = proposals[picked[0]];
            var mask = BuildMask(src.width, src.height, primaryProposal, out var maskRect);
            if (mask == null)
                return new FaceRegionResult { error = "Face mask build failed" };

            if (dumpDebug)
            {
                dumpDir = CreateDumpDir();
                if (debugLines != null)
                {
                    AppendProposalSummary(debugLines, proposals, picked, primaryProposal, src.width, src.height);
                    debugLines.Add(NcnnGpuResourceTracker.BuildSummary());
                }
                DumpMaskPng(mask, dumpDir, "ncnn_face_mask.png");
                DumpLandmarkOverlay(src, primaryProposal, primary.rectInt, dumpDir, "ncnn_face_landmarks.png");
                if (debugLines != null)
                    File.WriteAllLines(Path.Combine(dumpDir, "ncnn_face_debug.txt"), debugLines);
                NcnnGpuResourceTracker.WriteReport(dumpDir, "ncnn_face_gpu_resources.txt");
                if (autoOpenDumpDir && !string.IsNullOrWhiteSpace(dumpDir))
                    TryOpenFolderInShell(dumpDir);
            }

            return new FaceRegionResult
            {
                mask = mask,
                faceRect = primary.rectInt.width > 0 && primary.rectInt.height > 0
                    ? primary.rectInt
                    : (maskRect.width > 0 && maskRect.height > 0 ? maskRect : ClampRectToImage(RoundRect(primary.rect), src.width, src.height)),
                landmarks = primary.landmarks,
                score = primary.score,
                faces = faces,
                dumpDir = dumpDir
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
        int inputW,
        int inputH,
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
        if (inputW <= 0 || inputH <= 0)
            return;

        int numGridX;
        int numGridY;
        if (inputW > inputH)
        {
            numGridX = Mathf.Max(1, inputW / stride);
            numGridY = Mathf.Max(1, numGrid / numGridX);
        }
        else
        {
            numGridY = Mathf.Max(1, inputH / stride);
            numGridX = Mathf.Max(1, numGrid / numGridY);
        }

        if (numGridX * numGridY != numGrid)
        {
            numGridX = Mathf.Max(1, inputW / stride);
            numGridY = Mathf.Max(1, inputH / stride);
            if (numGridX * numGridY != numGrid)
            {
                numGridX = Mathf.Max(1, numGridX);
                numGridY = Mathf.Max(1, numGrid / numGridX);
            }
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

    private static FaceRegionFace[] BuildFaces(List<FaceProposal> proposals, List<int> picked, int imgW, int imgH, int maxFaces)
    {
        if (proposals == null || picked == null || picked.Count == 0)
            return Array.Empty<FaceRegionFace>();

        var count = maxFaces > 0 ? Mathf.Min(maxFaces, picked.Count) : picked.Count;
        var faces = new FaceRegionFace[count];
        for (var i = 0; i < count; i++)
        {
            var proposal = proposals[picked[i]];
            faces[i] = new FaceRegionFace
            {
                rect = proposal.rect,
                rectInt = ClampRectToImage(RoundRect(proposal.rect), imgW, imgH),
                landmarks = proposal.landmarks,
                score = proposal.score
            };
        }
        return faces;
    }

    private static FaceProposal PickBestFace(List<FaceProposal> proposals, List<int> picked, int imgW, int imgH, float maxAreaRatio, float maxAspectRatio)
    {
        var filtered = new List<FaceProposal>(picked.Count);
        for (var i = 0; i < picked.Count; i++)
        {
            var candidate = proposals[picked[i]];
            if (IsFaceProposalReasonable(candidate, imgW, imgH, maxAreaRatio, maxAspectRatio))
                filtered.Add(candidate);
        }

        var source = filtered.Count > 0 ? filtered : null;
        var best = source != null ? source[0] : proposals[picked[0]];
        var count = source != null ? source.Count : picked.Count;
        var bestRefRect = GetProposalReferenceRect(best, imgW, imgH);
        var bestPriority = ComputeProposalPriority(best, bestRefRect, imgW, imgH);
        for (var i = 1; i < count; i++)
        {
            var candidate = source != null ? source[i] : proposals[picked[i]];
            var candidateRefRect = GetProposalReferenceRect(candidate, imgW, imgH);
            var candidatePriority = ComputeProposalPriority(candidate, candidateRefRect, imgW, imgH);
            if (candidatePriority > bestPriority + 1e-6f)
            {
                best = candidate;
                bestRefRect = candidateRefRect;
                bestPriority = candidatePriority;
            }
        }
        return best;
    }

    private static bool IsFaceProposalReasonable(FaceProposal proposal, int imgW, int imgH, float maxAreaRatio, float maxAspectRatio)
    {
        var referenceRect = GetProposalReferenceRect(proposal, imgW, imgH);
        var area = Mathf.Max(1f, referenceRect.width * referenceRect.height);
        var imageArea = Mathf.Max(1f, imgW * imgH);
        if (area / imageArea > maxAreaRatio)
            return false;

        var aspect = referenceRect.width / Mathf.Max(1f, referenceRect.height);
        if (aspect > maxAspectRatio || aspect < 1f / maxAspectRatio)
            return false;

        return true;
    }

    private void EnsureRuntimeObjects()
    {
        if (_ops == null)
            _ops = new NcnnOps();
        if (_repro == null)
            _repro = new NcnnRepro2(_ops);
        _repro.PreferTexturePathForFaceDetector = preferTexturePathForFaceDetector;
        _repro.TensorTextureFormat = useArgbFloatForDetector ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
    }

    private static void AppendProposalSummary(List<string> lines, List<FaceProposal> proposals, List<int> picked, FaceProposal best, int imgW, int imgH)
    {
        if (lines == null)
            return;

        lines.Add("proposal_count=" + proposals.Count);
        lines.Add("picked_count=" + (picked != null ? picked.Count : 0));

        var sorted = new List<FaceProposal>(proposals);
        SortProposals(sorted);
        var topN = Mathf.Min(12, sorted.Count);
        for (var i = 0; i < topN; i++)
        {
            var p = sorted[i];
            var areaRatio = (p.rect.width * p.rect.height) / Mathf.Max(1f, imgW * imgH);
            lines.Add("top_proposal[" + i + "]"
                + " | score=" + p.score.ToString("F6", CultureInfo.InvariantCulture)
                + " | rect=" + RectToString(p.rect)
                + " | area_ratio=" + areaRatio.ToString("F6", CultureInfo.InvariantCulture)
                + " | landmarks=" + LandmarksToString(p.landmarks));
        }

        var bestAreaRatio = (best.rect.width * best.rect.height) / Mathf.Max(1f, imgW * imgH);
        lines.Add("best"
            + " | score=" + best.score.ToString("F6", CultureInfo.InvariantCulture)
            + " | rect=" + RectToString(best.rect)
            + " | area_ratio=" + bestAreaRatio.ToString("F6", CultureInfo.InvariantCulture)
            + " | landmarks=" + LandmarksToString(best.landmarks));

        if (picked == null)
            return;

        for (var i = 0; i < picked.Count; i++)
        {
            var index = picked[i];
            if (index < 0 || index >= proposals.Count)
                continue;
            var p = proposals[index];
            lines.Add("picked[" + i + "]"
                + " | proposal_index=" + index
                + " | score=" + p.score.ToString("F6", CultureInfo.InvariantCulture)
                + " | rect=" + RectToString(p.rect)
                + " | landmarks=" + LandmarksToString(p.landmarks));
        }
    }

    private static string LandmarksToString(IReadOnlyList<Vector2> landmarks)
    {
        if (landmarks == null || landmarks.Count == 0)
            return "[]";

        var parts = new string[landmarks.Count];
        for (var i = 0; i < landmarks.Count; i++)
        {
            parts[i] = "("
                + landmarks[i].x.ToString("F2", CultureInfo.InvariantCulture)
                + ","
                + landmarks[i].y.ToString("F2", CultureInfo.InvariantCulture)
                + ")";
        }

        return "[" + string.Join(" ", parts) + "]";
    }

    private void AppendBlobPreview(NcnnRepro2.InferResult infer, List<string> lines, string blobName, int previewCount)
    {
        if (infer == null || lines == null || string.IsNullOrWhiteSpace(blobName))
            return;

        try
        {
            if (!infer.TryGetLogicalShape(blobName, out var dims, out var w, out var h, out var d, out var c))
            {
                    lines.Add(blobName + " | shape=missing");
                    return;
                }

            float[] data;
            string sourceKind;
            try
            {
                data = infer.GetBufferData(blobName);
                sourceKind = "buffer";
            }
            catch
            {
                var tex = infer.GetTexture(blobName);
                if (tex == null)
                {
                    lines.Add(blobName + " | preview_error=texture missing");
                    return;
                }

                var packs = tex.volumeDepth > 0 ? tex.volumeDepth : 1;
                var physicalChannels = packs * 4;
                using var tempBuffer = new ComputeBuffer(tex.width * tex.height * physicalChannels, sizeof(float), ComputeBufferType.Structured);
                _ops.Pack4ToBufferCHW(tex, tex.width, tex.height, physicalChannels, tempBuffer);
                data = new float[tempBuffer.count];
                tempBuffer.GetData(data);
                sourceKind = "texture";
            }

            var finiteCount = 0;
            var nonFiniteCount = 0;
            if (data != null)
            {
                for (var i = 0; i < data.Length; i++)
                {
                    if (float.IsNaN(data[i]) || float.IsInfinity(data[i]))
                        nonFiniteCount++;
                    else
                        finiteCount++;
                }
            }

            lines.Add(blobName + " | source=" + sourceKind + " | dims=" + dims + " | w=" + w + " | h=" + h + " | d=" + d + " | c=" + c + " | count=" + (data != null ? data.Length : 0) + " | finite=" + finiteCount + " | nonfinite=" + nonFiniteCount);
            if (data == null || data.Length == 0)
                return;

            var take = Mathf.Min(previewCount, data.Length);
            var parts = new string[take];
            for (var i = 0; i < take; i++)
                parts[i] = data[i].ToString("G9", CultureInfo.InvariantCulture);
            lines.Add(blobName + " | first=" + string.Join(",", parts));
        }
        catch (Exception e)
        {
            lines.Add(blobName + " | preview_error=" + e.Message);
        }
    }

    private static string RectToString(Rect rect)
    {
        return rect.xMin.ToString("F2", CultureInfo.InvariantCulture)
            + ","
            + rect.yMin.ToString("F2", CultureInfo.InvariantCulture)
            + ","
            + rect.width.ToString("F2", CultureInfo.InvariantCulture)
            + ","
            + rect.height.ToString("F2", CultureInfo.InvariantCulture);
    }

    private Texture2D BuildMask(int width, int height, FaceProposal best, out RectInt faceRect)
    {
        var bytes = new byte[width * height * 2];
        var transform = SolveSimilarityTransform(best.landmarks, CanonicalFivePointTemplate);
        var threshold = Mathf.Clamp01(faceRectThreshold);
        var softness = Mathf.Clamp(maskSoftness, 0.01f, 0.35f);
        var searchRect = ExpandRect(RoundRect(best.rect), width, height, Mathf.Clamp(maskRectExpand, 0f, 0.6f));
        var hasLandmarkRect = TryEstimateLandmarkFaceRect(best, width, height, out var landmarkRect);
        if (hasLandmarkRect)
            searchRect = ExpandRect(RoundRect(landmarkRect), width, height, Mathf.Clamp(maskRectExpand, 0f, 0.45f));
        else if (TryGetTransformedEllipseRect(transform, width, height, 1f + softness, out var landmarkSupportRect))
            searchRect = ExpandRect(RoundRect(landmarkSupportRect), width, height, Mathf.Clamp(maskRectExpand, 0f, 0.6f));
        var minX = width;
        var minY = height;
        var maxX = -1;
        var maxY = -1;

        for (var y = searchRect.yMin; y < searchRect.yMax; y++)
        {
            for (var x = searchRect.xMin; x < searchRect.xMax; x++)
            {
                float value;
                if (hasLandmarkRect)
                {
                    value = EvaluateFallbackEllipse(landmarkRect, x + 0.5f, y + 0.5f, softness);
                }
                else if (transform.valid)
                {
                    var q = transform.Transform(new Vector2(x + 0.5f, y + 0.5f));
                    var dx = (q.x - CanonicalFaceCenter.x) / CanonicalFaceRadiusX;
                    var dy = (q.y - CanonicalFaceCenter.y) / CanonicalFaceRadiusY;
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

        if (maxX >= minX && maxY >= minY)
        {
            faceRect = new RectInt(minX, minY, maxX - minX + 1, maxY - minY + 1);
        }
        else if (hasLandmarkRect)
        {
            faceRect = ClampRectToImage(RoundRect(landmarkRect), width, height);
        }
        else if (TryGetTransformedEllipseRect(transform, width, height, Mathf.Max(0.75f, 1f + softness - threshold * softness * 2f), out var landmarkFaceRect))
        {
            faceRect = ClampRectToImage(RoundRect(landmarkFaceRect), width, height);
        }
        else
        {
            faceRect = ExpandRect(RoundRect(best.rect), width, height, 0.12f);
        }

        var tex = new Texture2D(width, height, TextureFormat.RHalf, false, true);
        tex.LoadRawTextureData(bytes);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        tex.name = "NcnnFaceMask";
        return tex;
    }

    private static Rect GetProposalReferenceRect(FaceProposal proposal, int imgW, int imgH)
    {
        if (TryEstimateLandmarkFaceRect(proposal, imgW, imgH, out var landmarkRect))
            return landmarkRect;
        if (TryGetTransformedEllipseRect(SolveSimilarityTransform(proposal.landmarks, CanonicalFivePointTemplate), imgW, imgH, 1.02f, out landmarkRect))
            return landmarkRect;
        return proposal.rect;
    }

    private static bool TryEstimateLandmarkFaceRect(FaceProposal proposal, int imgW, int imgH, out Rect rect)
    {
        rect = default;
        var landmarks = proposal.landmarks;
        if (landmarks == null || landmarks.Length < 5)
            return false;

        var leftEye = landmarks[0];
        var rightEye = landmarks[1];
        var mouthLeft = landmarks[3];
        var mouthRight = landmarks[4];
        var eyeCenter = (leftEye + rightEye) * 0.5f;
        var mouthCenter = (mouthLeft + mouthRight) * 0.5f;

        var eyeDist = Vector2.Distance(leftEye, rightEye);
        var mouthDist = Vector2.Distance(mouthLeft, mouthRight);
        var eyeToMouth = Vector2.Distance(eyeCenter, mouthCenter);
        if (eyeDist < 4f || eyeToMouth < 4f)
            return false;

        var centroid = Vector2.zero;
        for (var i = 0; i < landmarks.Length; i++)
            centroid += landmarks[i];
        centroid /= landmarks.Length;

        var outlierIndex = -1;
        var maxDistSq = float.NegativeInfinity;
        for (var i = 0; i < landmarks.Length; i++)
        {
            var distSq = (landmarks[i] - centroid).sqrMagnitude;
            if (distSq > maxDistSq)
            {
                maxDistSq = distSq;
                outlierIndex = i;
            }
        }

        var minX = float.PositiveInfinity;
        var minY = float.PositiveInfinity;
        var maxX = float.NegativeInfinity;
        var maxY = float.NegativeInfinity;
        for (var i = 0; i < landmarks.Length; i++)
        {
            if (i == outlierIndex)
                continue;
            minX = Mathf.Min(minX, landmarks[i].x);
            minY = Mathf.Min(minY, landmarks[i].y);
            maxX = Mathf.Max(maxX, landmarks[i].x);
            maxY = Mathf.Max(maxY, landmarks[i].y);
        }

        var spanX = Mathf.Max(eyeDist, maxX - minX);
        var spanY = Mathf.Max(eyeToMouth, maxY - minY);
        var faceWidth = Mathf.Max(spanX * 2.05f, mouthDist * 2.35f, eyeDist * 2.45f);
        var faceHeight = Mathf.Max(spanY * 3.15f, eyeToMouth * 2.8f, faceWidth * 1.15f);
        var center = new Vector2((minX + maxX) * 0.5f, (minY + maxY) * 0.5f - spanY * 0.20f);

        var x0 = center.x - faceWidth * 0.5f;
        var y0 = center.y - faceHeight * 0.46f;
        rect = Rect.MinMaxRect(
            Mathf.Clamp(x0, 0f, Mathf.Max(0f, imgW - 1f)),
            Mathf.Clamp(y0, 0f, Mathf.Max(0f, imgH - 1f)),
            Mathf.Clamp(x0 + faceWidth, 0f, Mathf.Max(0f, imgW - 1f)),
            Mathf.Clamp(y0 + faceHeight, 0f, Mathf.Max(0f, imgH - 1f)));

        return rect.width > 8f && rect.height > 8f;
    }

    private static bool TryGetTransformedEllipseRect(SimilarityTransform transform, int width, int height, float ellipseScale, out Rect rect)
    {
        rect = default;
        if (!transform.valid || ellipseScale <= 0f)
            return false;

        var minX = float.PositiveInfinity;
        var minY = float.PositiveInfinity;
        var maxX = float.NegativeInfinity;
        var maxY = float.NegativeInfinity;

        for (var i = 0; i < 48; i++)
        {
            var t = i * Mathf.PI * 2f / 48f;
            var canonical = new Vector2(
                CanonicalFaceCenter.x + Mathf.Cos(t) * CanonicalFaceRadiusX * ellipseScale,
                CanonicalFaceCenter.y + Mathf.Sin(t) * CanonicalFaceRadiusY * ellipseScale);
            if (!transform.TryInverseTransform(canonical, out var imagePoint))
                return false;

            minX = Mathf.Min(minX, imagePoint.x);
            minY = Mathf.Min(minY, imagePoint.y);
            maxX = Mathf.Max(maxX, imagePoint.x);
            maxY = Mathf.Max(maxY, imagePoint.y);
        }

        if (float.IsNaN(minX) || float.IsInfinity(minX)
            || float.IsNaN(minY) || float.IsInfinity(minY)
            || float.IsNaN(maxX) || float.IsInfinity(maxX)
            || float.IsNaN(maxY) || float.IsInfinity(maxY))
            return false;

        minX = Mathf.Clamp(minX, 0f, Mathf.Max(0f, width - 1f));
        minY = Mathf.Clamp(minY, 0f, Mathf.Max(0f, height - 1f));
        maxX = Mathf.Clamp(maxX, 0f, Mathf.Max(0f, width - 1f));
        maxY = Mathf.Clamp(maxY, 0f, Mathf.Max(0f, height - 1f));
        if (maxX <= minX || maxY <= minY)
            return false;

        rect = Rect.MinMaxRect(minX, minY, maxX, maxY);
        return true;
    }

    private static float ComputeNormalizedCenterDistance(Rect rect, int imgW, int imgH)
    {
        var dx = (rect.center.x - imgW * 0.5f) / Mathf.Max(1f, imgW);
        var dy = (rect.center.y - imgH * 0.5f) / Mathf.Max(1f, imgH);
        return dx * dx + dy * dy;
    }

    private static float ComputeProposalPriority(FaceProposal proposal, Rect rect, int imgW, int imgH)
    {
        var centerDist = Mathf.Sqrt(ComputeNormalizedCenterDistance(rect, imgW, imgH));
        var centerBonus = 1f - Mathf.Clamp01(centerDist * 2f);
        var borderPenalty = TouchesImageBorder(rect, imgW, imgH) ? 0.06f : 0f;
        var sizePenalty = 0f;
        if (proposal.landmarks != null && proposal.landmarks.Length >= 5)
        {
            var eyeDist = Vector2.Distance(proposal.landmarks[0], proposal.landmarks[1]);
            var mouthCenter = (proposal.landmarks[3] + proposal.landmarks[4]) * 0.5f;
            var eyeCenter = (proposal.landmarks[0] + proposal.landmarks[1]) * 0.5f;
            var eyeToMouth = Vector2.Distance(eyeCenter, mouthCenter);
            if (proposal.rect.width < eyeDist * 0.85f)
                sizePenalty += 0.10f;
            if (proposal.rect.height < eyeToMouth * 1.15f)
                sizePenalty += 0.10f;
            if (proposal.rect.width > eyeDist * 5.0f)
                sizePenalty += 0.05f;
            if (proposal.rect.height > eyeToMouth * 5.2f)
                sizePenalty += 0.05f;
        }

        return proposal.score * 0.88f + centerBonus * 0.18f - borderPenalty - sizePenalty;
    }

    private static bool TouchesImageBorder(Rect rect, int imgW, int imgH)
    {
        return rect.xMin <= 1f
            || rect.yMin <= 1f
            || rect.xMax >= imgW - 1f
            || rect.yMax >= imgH - 1f;
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

    private static RectInt ClampRectToImage(RectInt rect, int width, int height)
    {
        var x0 = Mathf.Clamp(rect.xMin, 0, Mathf.Max(0, width - 1));
        var y0 = Mathf.Clamp(rect.yMin, 0, Mathf.Max(0, height - 1));
        var x1 = Mathf.Clamp(rect.xMax, 0, width);
        var y1 = Mathf.Clamp(rect.yMax, 0, height);
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

    private static void DumpMaskPng(Texture2D mask, string dir, string fileName)
    {
        if (mask == null || string.IsNullOrWhiteSpace(dir))
            return;
        try
        {
            var outTex = new Texture2D(mask.width, mask.height, TextureFormat.RGBA32, false, true);
            var raw = mask.GetRawTextureData<byte>();
            var colors = new Color32[mask.width * mask.height];
            for (var i = 0; i < colors.Length; i++)
            {
                var lo = raw[i * 2 + 0];
                var hi = raw[i * 2 + 1];
                var half = (ushort)(lo | (hi << 8));
                var v = HalfToFloat(half);
                var b = (byte)Mathf.Clamp(Mathf.RoundToInt(v * 255f), 0, 255);
                colors[i] = new Color32(b, b, b, 255);
            }
            outTex.SetPixels32(colors);
            outTex.Apply(false, false);
            File.WriteAllBytes(Path.Combine(dir, fileName), outTex.EncodeToPNG());
            Destroy(outTex);
        }
        catch
        {
        }
    }

    private static void DumpLandmarkOverlay(Texture2D src, FaceProposal proposal, RectInt faceRect, string dir, string fileName)
    {
        if (src == null || string.IsNullOrWhiteSpace(dir))
            return;
        try
        {
            var tex = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false, true);
            tex.SetPixels32(src.GetPixels32());
            DrawRect(tex, ClampRectToImage(faceRect, src.width, src.height), new Color32(0, 255, 0, 255));
            DrawRect(tex, ClampRectToImage(RoundRect(proposal.rect), src.width, src.height), new Color32(255, 220, 32, 255));
            if (proposal.landmarks != null)
            {
                for (var i = 0; i < proposal.landmarks.Length; i++)
                    DrawPoint(tex, Mathf.RoundToInt(proposal.landmarks[i].x), Mathf.RoundToInt(proposal.landmarks[i].y), new Color32(255, 64, 64, 255), 3);
            }
            tex.Apply(false, false);
            File.WriteAllBytes(Path.Combine(dir, fileName), tex.EncodeToPNG());
            Destroy(tex);
        }
        catch
        {
        }
    }

    private static void DrawRect(Texture2D tex, RectInt rect, Color32 color)
    {
        for (var x = rect.xMin; x < rect.xMax; x++)
        {
            SetPixelSafe(tex, x, rect.yMin, color);
            SetPixelSafe(tex, x, rect.yMax - 1, color);
        }
        for (var y = rect.yMin; y < rect.yMax; y++)
        {
            SetPixelSafe(tex, rect.xMin, y, color);
            SetPixelSafe(tex, rect.xMax - 1, y, color);
        }
    }

    private static void DrawPoint(Texture2D tex, int cx, int cy, Color32 color, int radius)
    {
        for (var y = -radius; y <= radius; y++)
        {
            for (var x = -radius; x <= radius; x++)
            {
                if (x * x + y * y > radius * radius)
                    continue;
                SetPixelSafe(tex, cx + x, cy + y, color);
            }
        }
    }

    private static void SetPixelSafe(Texture2D tex, int x, int y, Color32 color)
    {
        if (tex == null || x < 0 || y < 0 || x >= tex.width || y >= tex.height)
            return;
        tex.SetPixel(x, y, color);
    }

    private static float HalfToFloat(ushort h)
    {
        uint sign = (uint)(h >> 15) & 1u;
        uint exp = (uint)(h >> 10) & 0x1Fu;
        uint mant = (uint)h & 0x3FFu;
        if (exp == 0)
        {
            if (mant == 0) return sign == 0 ? 0f : -0f;
            var v = mant / 1024f;
            v *= Mathf.Pow(2f, -14f);
            return sign == 0 ? v : -v;
        }
        if (exp == 31)
        {
            if (mant == 0) return sign == 0 ? float.PositiveInfinity : float.NegativeInfinity;
            return float.NaN;
        }
        var value = 1f + mant / 1024f;
        value *= Mathf.Pow(2f, (int)exp - 15);
        return sign == 0 ? value : -value;
    }

    private static string CreateDumpDir()
    {
        var root = Application.temporaryCachePath;
        if (string.IsNullOrWhiteSpace(root))
            root = Path.GetTempPath();
        var dir = Path.Combine(root, "AIImage_NcnnFaceRegion_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        try { Directory.CreateDirectory(dir); } catch { }
        return dir;
    }

#if !UNITY_WEBGL
    private static void TryOpenFolderInShell(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            return;
        try
        {
            Process.Start(new ProcessStartInfo { FileName = directoryPath, UseShellExecute = true });
        }
        catch
        {
        }
    }
#endif
}
