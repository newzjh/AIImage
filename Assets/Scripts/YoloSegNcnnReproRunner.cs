using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;

public struct YoloSegDetection
{
    public Rect rect;
    public int label;
    public float probability;
    public int gridIndex;
    public int maskPixelCount;
}

public struct YoloSegResult
{
    public Texture2D texture;
    public Texture2D mask;
    public Texture2D overlay;
    public YoloSegDetection[] detections;
    public int personCount;
    public float maskCoverage01;
    public string error;
    public long elapsedMs;
}

public sealed class YoloSegNcnnReproRunner : MonoBehaviour
{
    public enum YoloSegModelVariant
    {
        YoloV8nSeg,
        Yolo11nSeg
    }

    private readonly struct LetterboxState
    {
        public readonly int originalWidth;
        public readonly int originalHeight;
        public readonly int resizedWidth;
        public readonly int resizedHeight;
        public readonly int inputWidth;
        public readonly int inputHeight;
        public readonly int padLeft;
        public readonly int padRight;
        public readonly int padTop;
        public readonly int padBottom;
        public readonly float scale;

        public LetterboxState(
            int originalWidth,
            int originalHeight,
            int resizedWidth,
            int resizedHeight,
            int inputWidth,
            int inputHeight,
            int padLeft,
            int padRight,
            int padTop,
            int padBottom,
            float scale)
        {
            this.originalWidth = originalWidth;
            this.originalHeight = originalHeight;
            this.resizedWidth = resizedWidth;
            this.resizedHeight = resizedHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.padLeft = padLeft;
            this.padRight = padRight;
            this.padTop = padTop;
            this.padBottom = padBottom;
            this.scale = scale;
        }
    }

    private readonly struct BlobData
    {
        public readonly string name;
        public readonly float[] values;
        public readonly int dims;
        public readonly int w;
        public readonly int h;
        public readonly int d;
        public readonly int c;

        public BlobData(string name, float[] values, int dims, int w, int h, int d, int c)
        {
            this.name = name;
            this.values = values;
            this.dims = dims;
            this.w = w;
            this.h = h;
            this.d = d;
            this.c = c;
        }

        public int ElementCount => Mathf.Max(1, w) * Mathf.Max(1, h) * Mathf.Max(1, d) * Mathf.Max(1, c);
    }

    private struct Proposal
    {
        public Rect rect;
        public int label;
        public float probability;
        public int gridIndex;
    }

    private const string InputBlobName = "in0";
    private const string OutputBoxesBlobName = "out0";
    private const string OutputMaskCoeffBlobName = "out1";
    private const string OutputMaskProtoBlobName = "out2";
    private const int PersonClassId = 0;
    private const int MaskCoeffChannels = 32;
    private const int RegMax1 = 16;
    private const float PadColor01 = 114f / 255f;

    private static readonly int[] DefaultStrides = { 8, 16, 32 };
    private static readonly string[] DebugBlobNames =
    {
        "139", "140", "141",
        "160", "161", "162",
        "180", "181", "182",
        "194", "195", "196",
        "201", "202", "203",
        "208", "209", "210",
        "222", "245", "246",
        "233", "247", "248",
        "244", "249", "250",
        OutputBoxesBlobName, OutputMaskCoeffBlobName, OutputMaskProtoBlobName
    };

    public YoloSegModelVariant modelVariant = YoloSegModelVariant.YoloV8nSeg;
    public string yolo8ParamRelativePath = "Yolo/yolov8n_seg.ncnn.param";
    public string yolo8BinRelativePath = "Yolo/yolov8n_seg.ncnn.bin";
    public string yolo11ParamRelativePath = "Yolo/yolo11n_seg.ncnn.param";
    public string yolo11BinRelativePath = "Yolo/yolo11n_seg.ncnn.bin";
    public int targetSize = 640;
    public int maxStride = 32;
    public float probThreshold = 0.25f;
    public float nmsThreshold = 0.45f;
    public float maskThreshold = 0.5f;
    public bool agnosticNms = false;
    public bool targetPersonOnly = true;
    public bool flipYInput = true;
    public bool enableTempPool = true;
    public int maxPooledPerShape = 4;
    public bool forceBufferConvolution = false;
    public bool forceBufferBinaryOp = false;
    public bool useArgbFloatTensor = false;
    public bool enableDepthWiseTextureConvolution = true;
    public bool enableConv1x1TextureConvolution = true;
    public bool enableMaskClose = true;
    [Range(0, 6)] public int maskCloseRadius = 1;
    public bool enableMaskDilate = true;
    [Range(0, 6)] public int maskDilateRadius = 1;
    public bool enableDebugDump = false;
    public Color32 overlayColor = new Color32(255, 90, 90, 255);
    [Range(0f, 1f)] public float overlayOpacity = 0.45f;

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnRepro _repro;
    private string _loadedModelKey;
    private string _lastDumpDir;
    private string _lastSummaryText;

    public string LastDumpDir => _lastDumpDir;
    public string LastSummaryText => _lastSummaryText;

    private void Awake()
    {
        EnsureRuntimeObjects();
    }

    private void OnDestroy()
    {
        Release();
    }

    public async UniTask<YoloSegResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (src == null)
            return default;

        var totalSw = Stopwatch.StartNew();
        _lastDumpDir = null;
        _lastSummaryText = null;

        YoloSegResult Finish(YoloSegResult result)
        {
            result.elapsedMs = totalSw.ElapsedMilliseconds;
            return result;
        }

        RenderTexture resizedRt = null;
        RenderTexture corePack4 = null;
        RenderTexture inputPack4 = null;
        Texture2D readableSrc = null;

        try
        {
            EnsureRuntimeObjects();
            ApplyReproOptions();
            UnityEngine.Debug.Log("[YoloSegRunner] ProcessAsync start");
            if (enableDebugDump)
                _lastDumpDir = CreateDumpDir();

            UnityEngine.Debug.Log("[YoloSegRunner] EnsureLoaded begin");
            await EnsureLoaded(ct);
            UnityEngine.Debug.Log("[YoloSegRunner] EnsureLoaded end");
            ct.ThrowIfCancellationRequested();

            ReportProgress(0.08f, "Prepare input");
            await YieldIfNeeded();

            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: EnsureReadable begin");
            readableSrc = EnsureReadable(src);
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: EnsureReadable end");
            if (readableSrc == null)
                return Finish(new YoloSegResult { error = "Prepare source pixels failed" });

            var letterbox = ComputeLetterbox(readableSrc.width, readableSrc.height, Mathf.Max(32, targetSize), Mathf.Max(32, maxStride));
            _currentLetterbox = letterbox;
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: ResizeTextureBilinear begin " + letterbox.resizedWidth + "x" + letterbox.resizedHeight);
            resizedRt = ResizeTextureBilinear(readableSrc, letterbox.resizedWidth, letterbox.resizedHeight);
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: ResizeTextureBilinear end");
            if (resizedRt == null)
                return Finish(new YoloSegResult { error = "Resize input failed" });

            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: RentTempArray core begin");
            corePack4 = _repro.RentTempArray(letterbox.resizedWidth, letterbox.resizedHeight, 1, RenderTextureFormat.ARGBHalf);
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: RentTempArray core end");
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: RentTempArray input begin");
            inputPack4 = _repro.RentTempArray(letterbox.inputWidth, letterbox.inputHeight, 1, RenderTextureFormat.ARGBHalf);
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: RentTempArray input end");
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: PackRgbToPack4 begin");
            _ops.PackRgbToPack4(resizedRt, 0, 0, 1f, 1f, corePack4, flipYInput);
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: PackRgbToPack4 end");
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: PaddingPack4 begin");
            _ops.PaddingPack4(
                corePack4,
                1,
                letterbox.padLeft,
                letterbox.padRight,
                letterbox.padTop,
                letterbox.padBottom,
                0,
                new Vector4(PadColor01, PadColor01, PadColor01, 0f),
                inputPack4);
            UnityEngine.Debug.Log("[YoloSegRunner] Prepare input: PaddingPack4 end");

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                var inputLogical = ReadPack4TextureLogical(inputPack4, letterbox.inputWidth, letterbox.inputHeight, 3);
                var debugInputTexture = BuildRgbTextureFromChw(inputLogical, letterbox.inputWidth, letterbox.inputHeight, 3);
                TryWriteTexturePng(debugInputTexture, _lastDumpDir, "00_letterbox_input.png");
                TryDumpPack4TextureLogicalSummary(inputLogical, letterbox.inputWidth, letterbox.inputHeight, 3, Path.Combine(_lastDumpDir, "blob_in0_summary.txt"));
                TryWriteFloatArray(inputLogical, _lastDumpDir, "blob_in0_f32.bin");
                if (debugInputTexture != null)
                    Destroy(debugInputTexture);
            }

            ReportProgress(0.35f, "Run YOLO seg");
            await YieldIfNeeded();
            ct.ThrowIfCancellationRequested();

            BlobData predBlob;
            BlobData coeffBlob;
            BlobData protoBlob;
            HashSet<string> pinned = null;
            if (enableDebugDump)
                pinned = new HashSet<string>(DebugBlobNames, StringComparer.Ordinal);

            UnityEngine.Debug.Log("[YoloSegRunner] Infer begin");
            using (var infer = _repro.Infer(inputPack4, 1, InputBlobName, pinned))
            {
                UnityEngine.Debug.Log("[YoloSegRunner] Infer read blobs begin");
                predBlob = ReadBlobData(infer, OutputBoxesBlobName);
                coeffBlob = ReadBlobData(infer, OutputMaskCoeffBlobName);
                protoBlob = ReadBlobData(infer, OutputMaskProtoBlobName);
                UnityEngine.Debug.Log("[YoloSegRunner] Infer read blobs end");

                if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                {
                    TryWriteFloatArray(predBlob.values, _lastDumpDir, "blob_out0_f32.bin");
                    TryWriteFloatArray(coeffBlob.values, _lastDumpDir, "blob_out1_f32.bin");
                    TryWriteFloatArray(protoBlob.values, _lastDumpDir, "blob_out2_f32.bin");
                    for (var i = 0; i < DebugBlobNames.Length; i++)
                    {
                        TryDumpBlobSummary(infer, DebugBlobNames[i], Path.Combine(_lastDumpDir, "blob_" + DebugBlobNames[i] + "_summary.txt"));
                        TryDumpBlobFloatArray(infer, DebugBlobNames[i], _lastDumpDir, "blob_" + DebugBlobNames[i] + "_f32.bin");
                    }
                }
            }
            UnityEngine.Debug.Log("[YoloSegRunner] Infer end");

            if (predBlob.values == null || predBlob.values.Length == 0)
                return Finish(new YoloSegResult { error = "YOLO pred blob missing" });
            if (coeffBlob.values == null || coeffBlob.values.Length == 0)
                return Finish(new YoloSegResult { error = "YOLO mask coeff blob missing" });
            if (protoBlob.values == null || protoBlob.values.Length == 0)
                return Finish(new YoloSegResult { error = "YOLO mask proto blob missing" });

            var predCols = Resolve2DCols(predBlob, OutputBoxesBlobName);
            var predRows = Resolve2DRows(predBlob, OutputBoxesBlobName);
            var coeffCols = Resolve2DCols(coeffBlob, OutputMaskCoeffBlobName);
            var coeffRows = Resolve2DRows(coeffBlob, OutputMaskCoeffBlobName);
            if (predRows <= 0 || predCols <= 0)
                return Finish(new YoloSegResult { error = "YOLO pred blob shape invalid" });
            if (coeffRows != predRows)
                return Finish(new YoloSegResult { error = "YOLO pred/mask coeff rows mismatch" });
            if (coeffCols != MaskCoeffChannels)
                return Finish(new YoloSegResult { error = "YOLO mask coeff channels mismatch: " + coeffCols });
            if (predCols <= RegMax1 * 4)
                return Finish(new YoloSegResult { error = "YOLO pred class columns invalid: " + predCols });

            var proposals = new List<Proposal>(Mathf.Max(32, predRows / 8));
            GenerateProposals(predBlob.values, predRows, predCols, letterbox, proposals);
            proposals.Sort(static (a, b) => b.probability.CompareTo(a.probability));

            var picked = new List<int>(proposals.Count);
            NmsSortedBBoxes(proposals, picked, Mathf.Clamp01(nmsThreshold), agnosticNms);
            UnityEngine.Debug.Log("[YoloSegRunner] Proposals generated=" + proposals.Count + " picked=" + picked.Count);

            ReportProgress(0.62f, "Build person mask");
            await YieldIfNeeded();
            ct.ThrowIfCancellationRequested();

            var detections = new List<YoloSegDetection>(picked.Count);
            var unionMask = new bool[readableSrc.width * readableSrc.height];
            var resizedMaskWidth = Mathf.Max(1, (int)(letterbox.inputWidth / Mathf.Max(letterbox.scale, 1e-6f)));
            var resizedMaskHeight = Mathf.Max(1, (int)(letterbox.inputHeight / Mathf.Max(letterbox.scale, 1e-6f)));
            var protoWidth = protoBlob.w;
            var protoHeight = protoBlob.h;
            var protoChannels = protoBlob.c;
            if (protoBlob.dims != 3 || protoWidth <= 0 || protoHeight <= 0 || protoChannels != MaskCoeffChannels)
                return Finish(new YoloSegResult { error = "YOLO proto blob shape invalid" });

            for (var i = 0; i < picked.Count; i++)
            {
                ct.ThrowIfCancellationRequested();

                var proposal = proposals[picked[i]];
                if (proposal.gridIndex < 0 || proposal.gridIndex >= coeffRows)
                    continue;

                var coeffOffset = proposal.gridIndex * coeffCols;
                var protoMask = BuildObjectProtoMask(coeffBlob.values, coeffOffset, coeffCols, protoBlob.values, protoWidth, protoHeight, protoChannels);
                var clippedRect = ClipRectToImage(proposal.rect, readableSrc.width, readableSrc.height);
                var startX = Mathf.Clamp((int)clippedRect.x, 0, Mathf.Max(0, readableSrc.width - 1));
                var startY = Mathf.Clamp((int)clippedRect.y, 0, Mathf.Max(0, readableSrc.height - 1));
                var maskWidth = Mathf.Min(readableSrc.width - startX, Mathf.Max(0, (int)clippedRect.width));
                var maskHeight = Mathf.Min(readableSrc.height - startY, Mathf.Max(0, (int)clippedRect.height));
                if (maskWidth <= 0 || maskHeight <= 0)
                    continue;

                var paddedStartX = (int)(letterbox.padLeft / Mathf.Max(letterbox.scale, 1e-6f) + clippedRect.x);
                var paddedStartY = (int)(letterbox.padTop / Mathf.Max(letterbox.scale, 1e-6f) + clippedRect.y);
                var maskPixels = 0;
                for (var y = 0; y < maskHeight; y++)
                {
                    var sampleY = paddedStartY + y;
                    if ((uint)sampleY >= (uint)resizedMaskHeight)
                        continue;

                    var dstRow = (startY + y) * readableSrc.width;
                    for (var x = 0; x < maskWidth; x++)
                    {
                        var sampleX = paddedStartX + x;
                        if ((uint)sampleX >= (uint)resizedMaskWidth)
                            continue;

                        var maskValue = SampleResizedMask(protoMask, protoWidth, protoHeight, resizedMaskWidth, resizedMaskHeight, sampleX, sampleY);
                        if (maskValue <= maskThreshold)
                            continue;

                        var index = dstRow + startX + x;
                        if (!unionMask[index])
                        {
                            unionMask[index] = true;
                            maskPixels++;
                        }
                    }
                }

                detections.Add(new YoloSegDetection
                {
                    rect = clippedRect,
                    label = proposal.label,
                    probability = proposal.probability,
                    gridIndex = proposal.gridIndex,
                    maskPixelCount = maskPixels
                });
            }
            UnityEngine.Debug.Log("[YoloSegRunner] Mask build end detections=" + detections.Count);

            if (enableMaskClose && maskCloseRadius > 0)
                unionMask = MorphClose(unionMask, readableSrc.width, readableSrc.height, maskCloseRadius);
            if (enableMaskDilate && maskDilateRadius > 0)
                unionMask = MorphDilate(unionMask, readableSrc.width, readableSrc.height, maskDilateRadius);

            ReportProgress(0.82f, "Build outputs");
            await YieldIfNeeded();
            ct.ThrowIfCancellationRequested();

            var outputMask = BuildMaskTexture(unionMask, readableSrc.width, readableSrc.height);
            var transparent = BuildTransparentTexture(readableSrc, unionMask);
            var overlay = BuildOverlayTexture(readableSrc, unionMask, overlayColor, Mathf.Clamp01(overlayOpacity));
            var coverage = ComputeMaskCoverage(unionMask);
            _lastSummaryText = BuildSummary(
                readableSrc.width,
                readableSrc.height,
                letterbox,
                predBlob,
                coeffBlob,
                protoBlob,
                detections,
                coverage);

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                TryWriteTexturePng(outputMask, _lastDumpDir, "01_person_mask.png");
                TryWriteTexturePng(transparent, _lastDumpDir, "02_transparent_cutout.png");
                TryWriteTexturePng(overlay, _lastDumpDir, "03_overlay.png");
                TryWriteTextFile(_lastSummaryText, _lastDumpDir, "summary.txt");
            }

            UnityEngine.Debug.Log("[YoloSegRunner] ProcessAsync success");

            return Finish(new YoloSegResult
            {
                texture = transparent,
                mask = outputMask,
                overlay = overlay,
                detections = detections.ToArray(),
                personCount = detections.Count,
                maskCoverage01 = coverage
            });
        }
        catch (OperationCanceledException)
        {
            return Finish(new YoloSegResult { error = "Cancelled" });
        }
        catch (Exception e)
        {
            return Finish(new YoloSegResult { error = e.Message });
        }
        finally
        {
            if (resizedRt != null)
                ReleaseTemporaryRt(resizedRt);
            if (corePack4 != null)
                _repro?.ReturnTempArray(corePack4);
            if (inputPack4 != null)
                _repro?.ReturnTempArray(inputPack4);
            _repro?.ClearTempPool();
            if (readableSrc != null && readableSrc != src)
                Destroy(readableSrc);
            ReportProgress(1f, string.Empty);
        }
    }

    private void EnsureRuntimeObjects()
    {
        _ops ??= new NcnnOps();
        _repro ??= new NcnnRepro(_ops);
    }

    private void ApplyReproOptions()
    {
        if (_repro == null)
            return;

        _repro.EnableTempPool = enableTempPool;
        _repro.MaxPooledPerShape = maxPooledPerShape;
        _repro.ForceBufferConvolutionAll = forceBufferConvolution;
        _repro.ForceBufferBinaryOpAll = forceBufferBinaryOp;
        _repro.ForceBufferGeluAll = false;
        _repro.EnableDepthWiseTextureConvolution = modelVariant == YoloSegModelVariant.Yolo11nSeg
            ? false
            : enableDepthWiseTextureConvolution;
        _repro.EnableConv1x1TextureConvolution = enableConv1x1TextureConvolution;
        _repro.TensorTextureFormat = useArgbFloatTensor ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
        _repro.DebugCompareTextureLayers = null;
        _repro.DebugLog = null;
    }

    private async UniTask EnsureLoaded(CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var (paramRelativePath, binRelativePath, modelKey) = ResolveModelPaths();
        if (string.Equals(_loadedModelKey, modelKey, StringComparison.Ordinal) && _repro?.Model != null)
            return;

        var paramPath = Path.Combine(Application.streamingAssetsPath, paramRelativePath);
        var binPath = Path.Combine(Application.streamingAssetsPath, binRelativePath);
        if (!File.Exists(paramPath))
            throw new FileNotFoundException("YOLO seg param not found", paramPath);
        if (!File.Exists(binPath))
            throw new FileNotFoundException("YOLO seg bin not found", binPath);

        var paramText = await File.ReadAllTextAsync(paramPath, ct);
        var binBytes = await File.ReadAllBytesAsync(binPath, ct);
        using var ms = new MemoryStream(binBytes, false);
        using var br = new NcnnBinReader(ms);
        _repro.LoadModel(paramText, br);
        _loadedModelKey = modelKey;
    }

    private (string paramRelativePath, string binRelativePath, string modelKey) ResolveModelPaths()
    {
        return modelVariant == YoloSegModelVariant.Yolo11nSeg
            ? (yolo11ParamRelativePath, yolo11BinRelativePath, "yolo11n_seg")
            : (yolo8ParamRelativePath, yolo8BinRelativePath, "yolov8n_seg");
    }

    private void Release()
    {
        try { _repro?.Dispose(); } catch { }
        _repro = null;
        _ops = null;
        _loadedModelKey = null;
        _lastDumpDir = null;
        _lastSummaryText = null;
    }

    private BlobData ReadBlobData(NcnnRepro.InferResult infer, string blobName)
    {
        if (infer == null)
            throw new ArgumentNullException(nameof(infer));
        if (string.IsNullOrWhiteSpace(blobName))
            throw new ArgumentException("blobName is empty", nameof(blobName));
        if (!infer.TryGetLogicalShape(blobName, out var dims, out var w, out var h, out var d, out var c))
            throw new InvalidOperationException("Blob shape not found: " + blobName);

        float[] values;
        try
        {
            values = infer.GetBufferData(blobName);
        }
        catch
        {
            var texture = infer.GetTexture(blobName);
            if (texture == null)
                throw new InvalidOperationException("Blob data not found: " + blobName);

            var packs = texture.volumeDepth > 0 ? texture.volumeDepth : 1;
            var physicalChannels = packs * 4;
            var physicalCount = texture.width * texture.height * physicalChannels;
            using var physicalBuffer = new ComputeBuffer(physicalCount, sizeof(float), ComputeBufferType.Structured);
            _ops.Pack4ToBufferCHW(texture, texture.width, texture.height, physicalChannels, physicalBuffer);
            var physical = new float[physicalCount];
            physicalBuffer.GetData(physical);
            var logicalCount = Mathf.Max(1, w) * Mathf.Max(1, h) * Mathf.Max(1, d) * Mathf.Max(1, c);
            values = new float[Mathf.Min(logicalCount, physical.Length)];
            Array.Copy(physical, values, values.Length);
        }

        return new BlobData(blobName, values, dims, w, h, d, c);
    }

    private void GenerateProposals(float[] pred, int rowCount, int rowWidth, LetterboxState letterbox, List<Proposal> proposals)
    {
        if (pred == null || proposals == null)
            return;

        var numClasses = rowWidth - RegMax1 * 4;
        var rowOffset = 0;
        for (var strideIndex = 0; strideIndex < DefaultStrides.Length; strideIndex++)
        {
            var stride = DefaultStrides[strideIndex];
            var gridX = letterbox.inputWidth / stride;
            var gridY = letterbox.inputHeight / stride;
            var gridCount = gridX * gridY;
            if (rowOffset + gridCount > rowCount)
                break;

            for (var y = 0; y < gridY; y++)
            {
                for (var x = 0; x < gridX; x++)
                {
                    var gridIndex = rowOffset + y * gridX + x;
                    var rowBase = gridIndex * rowWidth;
                    var bestLabel = -1;
                    var bestLogit = float.NegativeInfinity;
                    for (var k = 0; k < numClasses; k++)
                    {
                        var score = pred[rowBase + RegMax1 * 4 + k];
                        if (score > bestLogit)
                        {
                            bestLogit = score;
                            bestLabel = k;
                        }
                    }

                    var probability = Sigmoid(bestLogit);
                    if (probability < probThreshold)
                        continue;
                    if (targetPersonOnly && bestLabel != PersonClassId)
                        continue;

                    var l = DecodeDistance(pred, rowBase + 0 * RegMax1, stride);
                    var t = DecodeDistance(pred, rowBase + 1 * RegMax1, stride);
                    var r = DecodeDistance(pred, rowBase + 2 * RegMax1, stride);
                    var b = DecodeDistance(pred, rowBase + 3 * RegMax1, stride);

                    var centerX = (x + 0.5f) * stride;
                    var centerY = (y + 0.5f) * stride;
                    var x0 = centerX - l;
                    var y0 = centerY - t;
                    var x1 = centerX + r;
                    var y1 = centerY + b;

                    proposals.Add(new Proposal
                    {
                        rect = new Rect(x0, y0, x1 - x0, y1 - y0),
                        label = bestLabel,
                        probability = probability,
                        gridIndex = gridIndex
                    });
                }
            }

            rowOffset += gridCount;
        }
    }

    private static float DecodeDistance(float[] rowValues, int rowBase, int stride)
    {
        var max = rowValues[rowBase];
        for (var i = 1; i < RegMax1; i++)
        {
            var v = rowValues[rowBase + i];
            if (v > max)
                max = v;
        }

        var sum = 0f;
        var weighted = 0f;
        for (var i = 0; i < RegMax1; i++)
        {
            var e = Mathf.Exp(rowValues[rowBase + i] - max);
            sum += e;
            weighted += i * e;
        }

        if (sum <= 1e-12f)
            return 0f;
        return weighted / sum * stride;
    }

    private static void NmsSortedBBoxes(List<Proposal> proposals, List<int> picked, float threshold, bool agnostic)
    {
        picked.Clear();
        if (proposals == null || proposals.Count == 0)
            return;

        var areas = new float[proposals.Count];
        for (var i = 0; i < proposals.Count; i++)
            areas[i] = Mathf.Max(0f, proposals[i].rect.width) * Mathf.Max(0f, proposals[i].rect.height);

        for (var i = 0; i < proposals.Count; i++)
        {
            var keep = true;
            var a = proposals[i];
            for (var j = 0; j < picked.Count; j++)
            {
                var b = proposals[picked[j]];
                if (!agnostic && a.label != b.label)
                    continue;

                var inter = IntersectionArea(a.rect, b.rect);
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
    }

    private static float IntersectionArea(Rect a, Rect b)
    {
        var x0 = Mathf.Max(a.xMin, b.xMin);
        var y0 = Mathf.Max(a.yMin, b.yMin);
        var x1 = Mathf.Min(a.xMax, b.xMax);
        var y1 = Mathf.Min(a.yMax, b.yMax);
        if (x1 <= x0 || y1 <= y0)
            return 0f;
        return (x1 - x0) * (y1 - y0);
    }

    private Rect ClipRectToImage(Rect rect, int width, int height)
    {
        var x0 = (rect.x - _currentLetterbox.padLeft) / Mathf.Max(_currentLetterbox.scale, 1e-6f);
        var y0 = (rect.y - _currentLetterbox.padTop) / Mathf.Max(_currentLetterbox.scale, 1e-6f);
        var x1 = (rect.x + rect.width - _currentLetterbox.padLeft) / Mathf.Max(_currentLetterbox.scale, 1e-6f);
        var y1 = (rect.y + rect.height - _currentLetterbox.padTop) / Mathf.Max(_currentLetterbox.scale, 1e-6f);

        x0 = Mathf.Clamp(x0, 0f, Mathf.Max(0f, width - 1f));
        y0 = Mathf.Clamp(y0, 0f, Mathf.Max(0f, height - 1f));
        x1 = Mathf.Clamp(x1, 0f, Mathf.Max(0f, width - 1f));
        y1 = Mathf.Clamp(y1, 0f, Mathf.Max(0f, height - 1f));

        return new Rect(x0, y0, Mathf.Max(0f, x1 - x0), Mathf.Max(0f, y1 - y0));
    }

    private LetterboxState _currentLetterbox;

    private static LetterboxState ComputeLetterbox(int srcWidth, int srcHeight, int targetSize, int maxStride)
    {
        var w = srcWidth;
        var h = srcHeight;
        var scale = 1f;
        if (w > h)
        {
            scale = (float)targetSize / Mathf.Max(1, w);
            w = targetSize;
            h = Mathf.Max(1, (int)(h * scale));
        }
        else
        {
            scale = (float)targetSize / Mathf.Max(1, h);
            h = targetSize;
            w = Mathf.Max(1, (int)(w * scale));
        }

        var wpad = ((w + maxStride - 1) / maxStride) * maxStride - w;
        var hpad = ((h + maxStride - 1) / maxStride) * maxStride - h;
        var padLeft = wpad / 2;
        var padRight = wpad - padLeft;
        var padTop = hpad / 2;
        var padBottom = hpad - padTop;

        return new LetterboxState(
            srcWidth,
            srcHeight,
            w,
            h,
            w + wpad,
            h + hpad,
            padLeft,
            padRight,
            padTop,
            padBottom,
            scale);
    }

    private static float[] BuildObjectProtoMask(float[] coeffValues, int coeffOffset, int coeffCount, float[] protoValues, int protoWidth, int protoHeight, int protoChannels)
    {
        if (coeffCount < protoChannels)
            throw new InvalidOperationException("mask coeff width smaller than proto channels");

        var plane = protoWidth * protoHeight;
        var output = new float[plane];
        for (var i = 0; i < plane; i++)
        {
            var sum = 0f;
            for (var c = 0; c < protoChannels; c++)
                sum += coeffValues[coeffOffset + c] * protoValues[c * plane + i];
            output[i] = Sigmoid(sum);
        }
        return output;
    }

    private static float SampleResizedMask(float[] protoMask, int srcWidth, int srcHeight, int dstWidth, int dstHeight, int dstX, int dstY)
    {
        if (protoMask == null || protoMask.Length != srcWidth * srcHeight)
            return 0f;

        var fx = ((dstX + 0.5f) * srcWidth / Mathf.Max(1, dstWidth)) - 0.5f;
        var fy = ((dstY + 0.5f) * srcHeight / Mathf.Max(1, dstHeight)) - 0.5f;

        var x0 = Mathf.Clamp((int)Mathf.Floor(fx), 0, Mathf.Max(0, srcWidth - 1));
        var y0 = Mathf.Clamp((int)Mathf.Floor(fy), 0, Mathf.Max(0, srcHeight - 1));
        var x1 = Mathf.Clamp(x0 + 1, 0, Mathf.Max(0, srcWidth - 1));
        var y1 = Mathf.Clamp(y0 + 1, 0, Mathf.Max(0, srcHeight - 1));
        var tx = Mathf.Clamp01(fx - x0);
        var ty = Mathf.Clamp01(fy - y0);

        var v00 = protoMask[y0 * srcWidth + x0];
        var v10 = protoMask[y0 * srcWidth + x1];
        var v01 = protoMask[y1 * srcWidth + x0];
        var v11 = protoMask[y1 * srcWidth + x1];
        var v0 = Mathf.Lerp(v00, v10, tx);
        var v1 = Mathf.Lerp(v01, v11, tx);
        return Mathf.Lerp(v0, v1, ty);
    }

    private static Texture2D BuildMaskTexture(bool[] mask, int width, int height)
    {
        var pixels = new Color32[width * height];
        for (var i = 0; i < pixels.Length; i++)
        {
            var v = mask != null && i < mask.Length && mask[i] ? (byte)255 : (byte)0;
            pixels[i] = new Color32(v, v, v, 255);
        }

        var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
        texture.SetPixels32(pixels);
        texture.Apply(false, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Bilinear;
        return texture;
    }

    private static Texture2D BuildTransparentTexture(Texture2D source, bool[] mask)
    {
        var srcPixels = source.GetPixels32();
        var pixels = new Color32[srcPixels.Length];
        for (var i = 0; i < srcPixels.Length; i++)
        {
            var p = srcPixels[i];
            if (mask != null && i < mask.Length && mask[i])
                p.a = 0;
            pixels[i] = p;
        }

        var texture = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, false);
        texture.SetPixels32(pixels);
        texture.Apply(false, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Bilinear;
        return texture;
    }

    private static Texture2D BuildOverlayTexture(Texture2D source, bool[] mask, Color32 tint, float opacity)
    {
        var srcPixels = source.GetPixels32();
        var pixels = new Color32[srcPixels.Length];
        for (var i = 0; i < srcPixels.Length; i++)
        {
            var p = srcPixels[i];
            if (mask != null && i < mask.Length && mask[i])
            {
                p = new Color32(
                    (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(p.r, tint.r, opacity)), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(p.g, tint.g, opacity)), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(p.b, tint.b, opacity)), 0, 255),
                    255);
            }
            pixels[i] = p;
        }

        var texture = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, false);
        texture.SetPixels32(pixels);
        texture.Apply(false, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Bilinear;
        return texture;
    }

    private static float ComputeMaskCoverage(bool[] mask)
    {
        if (mask == null || mask.Length == 0)
            return 0f;
        var count = 0;
        for (var i = 0; i < mask.Length; i++)
        {
            if (mask[i])
                count++;
        }
        return (float)count / mask.Length;
    }

    private static bool[] MorphClose(bool[] mask, int width, int height, int radius)
    {
        return MorphErode(MorphDilate(mask, width, height, radius), width, height, radius);
    }

    private static bool[] MorphDilate(bool[] mask, int width, int height, int radius)
    {
        if (mask == null || mask.Length != width * height || radius <= 0)
            return mask ?? Array.Empty<bool>();

        var result = new bool[mask.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var keep = false;
                for (var ky = -radius; ky <= radius && !keep; ky++)
                {
                    var ny = y + ky;
                    if ((uint)ny >= (uint)height)
                        continue;
                    var row = ny * width;
                    for (var kx = -radius; kx <= radius; kx++)
                    {
                        var nx = x + kx;
                        if ((uint)nx >= (uint)width)
                            continue;
                        if (!mask[row + nx])
                            continue;
                        keep = true;
                        break;
                    }
                }
                result[y * width + x] = keep;
            }
        }
        return result;
    }

    private static bool[] MorphErode(bool[] mask, int width, int height, int radius)
    {
        if (mask == null || mask.Length != width * height || radius <= 0)
            return mask ?? Array.Empty<bool>();

        var result = new bool[mask.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var keep = true;
                for (var ky = -radius; ky <= radius && keep; ky++)
                {
                    var ny = y + ky;
                    if ((uint)ny >= (uint)height)
                    {
                        keep = false;
                        break;
                    }

                    var row = ny * width;
                    for (var kx = -radius; kx <= radius; kx++)
                    {
                        var nx = x + kx;
                        if ((uint)nx >= (uint)width || !mask[row + nx])
                        {
                            keep = false;
                            break;
                        }
                    }
                }
                result[y * width + x] = keep;
            }
        }
        return result;
    }

    private static int Resolve2DCols(BlobData blob, string blobName)
    {
        if (blob.dims != 2)
            throw new InvalidOperationException(blobName + " dims expected 2 but got " + blob.dims);
        return blob.w;
    }

    private static int Resolve2DRows(BlobData blob, string blobName)
    {
        if (blob.dims != 2)
            throw new InvalidOperationException(blobName + " dims expected 2 but got " + blob.dims);
        return blob.h;
    }

    private static float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
    }

    private static RenderTexture ResizeTextureBilinear(Texture src, int width, int height)
    {
        if (src == null)
            return null;
        var rt = GetTemporaryRt(width, height, RenderTextureFormat.ARGB32, false);
        Graphics.Blit(src, rt);
        return rt;
    }

    private static Texture2D EnsureReadable(Texture2D src)
    {
        if (src == null)
            return null;
        try
        {
            var _ = src.GetPixel(0, 0);
            return src;
        }
        catch
        {
            var rt = ResizeTextureBilinear(src, src.width, src.height);
            if (rt == null)
                return null;
            try
            {
                return RenderTextureToTexture2D(rt, src.width, src.height);
            }
            finally
            {
                ReleaseTemporaryRt(rt);
            }
        }
    }

    private static Texture2D RenderTextureToTexture2D(RenderTexture rt, int width, int height)
    {
        var prev = RenderTexture.active;
        try
        {
            RenderTexture.active = rt;
            var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
            texture.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            texture.Apply(false, false);
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.filterMode = FilterMode.Bilinear;
            return texture;
        }
        finally
        {
            RenderTexture.active = prev;
        }
    }

    private static RenderTexture GetTemporaryRt(int width, int height, RenderTextureFormat format, bool enableRandomWrite)
    {
        var rt = RenderTexture.GetTemporary(width, height, 0, format, RenderTextureReadWrite.Default);
        rt.enableRandomWrite = enableRandomWrite;
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.filterMode = FilterMode.Bilinear;
        rt.Create();
        return rt;
    }

    private static void ReleaseTemporaryRt(RenderTexture rt)
    {
        if (rt == null)
            return;
        RenderTexture.ReleaseTemporary(rt);
    }

    private static void TryWriteTexturePng(Texture2D texture, string dir, string fileName)
    {
        if (texture == null || string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(fileName))
            return;
        try
        {
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, fileName), texture.EncodeToPNG());
        }
        catch
        {
        }
    }

    private static void TryWriteTextFile(string text, string dir, string fileName)
    {
        if (string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(fileName))
            return;
        try
        {
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, fileName), text ?? string.Empty, Encoding.UTF8);
        }
        catch
        {
        }
    }

    private static string CreateDumpDir()
    {
        var root = Path.Combine(Application.dataPath, "..", "Logs", "YoloSegNcnnRepro");
        Directory.CreateDirectory(root);
        var dir = Path.Combine(root, DateTime.Now.ToString("yyyyMMdd-HHmmss-fff", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static string BuildSummary(
        int sourceWidth,
        int sourceHeight,
        LetterboxState letterbox,
        BlobData predBlob,
        BlobData coeffBlob,
        BlobData protoBlob,
        List<YoloSegDetection> detections,
        float coverage)
    {
        var sb = new StringBuilder(1024);
        sb.AppendLine("source=" + sourceWidth + "x" + sourceHeight);
        sb.AppendLine("letterbox_resized=" + letterbox.resizedWidth + "x" + letterbox.resizedHeight);
        sb.AppendLine("letterbox_input=" + letterbox.inputWidth + "x" + letterbox.inputHeight);
        sb.AppendLine("pad=" + letterbox.padLeft + "," + letterbox.padTop + "," + letterbox.padRight + "," + letterbox.padBottom);
        sb.AppendLine("scale=" + letterbox.scale.ToString("0.000000", CultureInfo.InvariantCulture));
        sb.AppendLine("out0=" + predBlob.dims + "d " + predBlob.w + "x" + predBlob.h + "x" + predBlob.d + "x" + predBlob.c);
        sb.AppendLine("out1=" + coeffBlob.dims + "d " + coeffBlob.w + "x" + coeffBlob.h + "x" + coeffBlob.d + "x" + coeffBlob.c);
        sb.AppendLine("out2=" + protoBlob.dims + "d " + protoBlob.w + "x" + protoBlob.h + "x" + protoBlob.d + "x" + protoBlob.c);
        sb.AppendLine("detections=" + (detections?.Count ?? 0));
        sb.AppendLine("mask_coverage=" + coverage.ToString("0.000000", CultureInfo.InvariantCulture));
        if (detections != null)
        {
            for (var i = 0; i < detections.Count; i++)
            {
                var d = detections[i];
                sb.AppendLine(
                    i.ToString(CultureInfo.InvariantCulture) + "\tlabel=" + d.label
                    + "\tprob=" + d.probability.ToString("0.000000", CultureInfo.InvariantCulture)
                    + "\tgrid=" + d.gridIndex
                    + "\trect=" + d.rect.x.ToString("0.00", CultureInfo.InvariantCulture)
                    + "," + d.rect.y.ToString("0.00", CultureInfo.InvariantCulture)
                    + "," + d.rect.width.ToString("0.00", CultureInfo.InvariantCulture)
                    + "," + d.rect.height.ToString("0.00", CultureInfo.InvariantCulture)
                    + "\tmask_pixels=" + d.maskPixelCount.ToString(CultureInfo.InvariantCulture));
            }
        }
        return sb.ToString();
    }

    private void TryDumpBlobSummary(NcnnRepro.InferResult infer, string blobName, string path)
    {
        if (infer == null || string.IsNullOrWhiteSpace(blobName) || string.IsNullOrWhiteSpace(path))
            return;

        try
        {
            var blob = ReadBlobData(infer, blobName);
            var text = BuildBlobSummary(blob);
            if (string.IsNullOrWhiteSpace(text))
                return;
            TryWriteTextFile(text, Path.GetDirectoryName(path), Path.GetFileName(path));
        }
        catch
        {
        }
    }

    private static void TryWriteFloatArray(float[] values, string dir, string fileName)
    {
        if (values == null || string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(fileName))
            return;
        try
        {
            Directory.CreateDirectory(dir);
            var bytes = new byte[values.Length * sizeof(float)];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(dir, fileName), bytes);
        }
        catch
        {
        }
    }

    private void TryDumpBlobFloatArray(NcnnRepro.InferResult infer, string blobName, string dir, string fileName)
    {
        if (infer == null || string.IsNullOrWhiteSpace(blobName))
            return;

        try
        {
            var blob = ReadBlobData(infer, blobName);
            if (blob.values == null || blob.values.Length == 0)
                return;
            TryWriteFloatArray(blob.values, dir, fileName);
        }
        catch
        {
        }
    }

    private void TryDumpPack4TextureLogicalSummary(float[] logical, int width, int height, int channels, string path)
    {
        if (logical == null || logical.Length == 0 || width <= 0 || height <= 0 || channels <= 0 || string.IsNullOrWhiteSpace(path))
            return;

        try
        {
            var blob = new BlobData(InputBlobName, logical, 3, width, height, 1, channels);
            var text = BuildBlobSummary(blob);
            if (string.IsNullOrWhiteSpace(text))
                return;
            TryWriteTextFile(text, Path.GetDirectoryName(path), Path.GetFileName(path));
        }
        catch
        {
        }
    }

    private float[] ReadPack4TextureLogical(RenderTexture texture, int width, int height, int channels)
    {
        var physicalChannels = Mathf.Max(4, ((channels + 3) / 4) * 4);
        var physicalCount = width * height * physicalChannels;
        using var physicalBuffer = new ComputeBuffer(physicalCount, sizeof(float), ComputeBufferType.Structured);
        _ops.Pack4ToBufferCHW(texture, width, height, physicalChannels, physicalBuffer);
        var physical = new float[physicalCount];
        physicalBuffer.GetData(physical);
        var logical = new float[width * height * channels];
        Array.Copy(physical, logical, logical.Length);
        return logical;
    }

    private static Texture2D BuildRgbTextureFromChw(float[] chw, int width, int height, int channels)
    {
        if (chw == null || width <= 0 || height <= 0 || channels <= 0)
            return null;

        var plane = width * height;
        var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
        var pixels = new Color32[plane];
        for (var i = 0; i < plane; i++)
        {
            byte Sample(int channel)
            {
                if (channel >= channels)
                    return 0;
                var offset = channel * plane + i;
                if ((uint)offset >= (uint)chw.Length)
                    return 0;
                return (byte)Mathf.Clamp(Mathf.RoundToInt(chw[offset] * 255f), 0, 255);
            }

            pixels[i] = new Color32(Sample(0), Sample(1), Sample(2), 255);
        }

        texture.SetPixels32(pixels);
        texture.Apply(false, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Bilinear;
        return texture;
    }

    private static string BuildBlobSummary(BlobData blob)
    {
        if (blob.values == null)
            return null;

        var sb = new StringBuilder(2048);
        sb.AppendLine("name=" + blob.name);
        sb.AppendLine("shape=" + blob.dims + "d " + blob.w + "x" + blob.h + "x" + blob.d + "x" + blob.c);
        var finiteCount = 0;
        var nanCount = 0;
        var infCount = 0;
        double sum = 0d;
        float min = float.PositiveInfinity;
        float max = float.NegativeInfinity;
        for (var i = 0; i < blob.values.Length; i++)
        {
            var v = blob.values[i];
            if (float.IsNaN(v))
            {
                nanCount++;
                continue;
            }
            if (float.IsInfinity(v))
            {
                infCount++;
                continue;
            }
            finiteCount++;
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }

        sb.AppendLine("count=" + blob.values.Length);
        sb.AppendLine("finite=" + finiteCount);
        sb.AppendLine("nan=" + nanCount);
        sb.AppendLine("inf=" + infCount);
        sb.AppendLine("min=" + (finiteCount > 0 ? min.ToString("R", CultureInfo.InvariantCulture) : "NaN"));
        sb.AppendLine("max=" + (finiteCount > 0 ? max.ToString("R", CultureInfo.InvariantCulture) : "NaN"));
        sb.AppendLine("mean=" + (finiteCount > 0 ? (sum / finiteCount).ToString("R", CultureInfo.InvariantCulture) : "NaN"));

        if (blob.dims == 2 && blob.w > 0 && blob.h > 0)
        {
            var rowIndices = new List<int> { 0 };
            if (blob.h > 1) rowIndices.Add(Mathf.Min(1, blob.h - 1));
            if (blob.h > 2) rowIndices.Add(blob.h / 2);
            if (blob.h > 3) rowIndices.Add(blob.h - 1);
            var seen = new HashSet<int>();
            foreach (var row in rowIndices)
            {
                if (!seen.Add(row))
                    continue;
                sb.Append("row[").Append(row).Append("]=");
                var cols = Mathf.Min(blob.w, 24);
                for (var x = 0; x < cols; x++)
                {
                    if (x > 0) sb.Append(", ");
                    sb.Append(blob.values[row * blob.w + x].ToString("0.######", CultureInfo.InvariantCulture));
                }
                sb.AppendLine();
            }
        }
        else
        {
            sb.Append("head=");
            var head = Mathf.Min(blob.values.Length, 32);
            for (var i = 0; i < head; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(blob.values[i].ToString("0.######", CultureInfo.InvariantCulture));
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    private void ReportProgress(float progress01, string text)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(progress01), text ?? string.Empty); } catch { }
    }

    private static async UniTask YieldIfNeeded()
    {
        if (!Application.isBatchMode)
            await UniTask.Yield();
    }
}
