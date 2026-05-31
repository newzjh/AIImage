using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;

public struct CodeFormerResult
{
    public Texture2D texture;
    public string error;
    public long elapsedMs;
}
public sealed class CodeFormerNcnnReproRunner2 : MonoBehaviour
{
    private readonly struct Affine2D
    {
        public readonly float m00;
        public readonly float m01;
        public readonly float m02;
        public readonly float m10;
        public readonly float m11;
        public readonly float m12;
        public readonly bool valid;

        public Affine2D(float m00, float m01, float m02, float m10, float m11, float m12, bool valid)
        {
            this.m00 = m00;
            this.m01 = m01;
            this.m02 = m02;
            this.m10 = m10;
            this.m11 = m11;
            this.m12 = m12;
            this.valid = valid;
        }

        public Vector2 Transform(Vector2 p)
        {
            return new Vector2(
                m00 * p.x + m01 * p.y + m02,
                m10 * p.x + m11 * p.y + m12);
        }

        public bool TryInverse(out Affine2D inverse)
        {
            var det = m00 * m11 - m01 * m10;
            if (!valid || Mathf.Abs(det) < 1e-8f)
            {
                inverse = default;
                return false;
            }

            var inv = 1f / det;
            var i00 = m11 * inv;
            var i01 = -m01 * inv;
            var i10 = -m10 * inv;
            var i11 = m00 * inv;
            var i02 = -(i00 * m02 + i01 * m12);
            var i12 = -(i10 * m02 + i11 * m12);
            inverse = new Affine2D(i00, i01, i02, i10, i11, i12, true);
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

    private static readonly Vector2[] CanonicalFivePointTemplateBottomOrigin =
    {
        new Vector2(192.98138f, 271.05292f),
        new Vector2(318.90277f, 270.8064f),
        new Vector2(256.63416f, 196.98065f),
        new Vector2(201.26117f, 139.58957f),
        new Vector2(313.08905f, 139.84882f)
    };

    private struct CodeFormer512RunResult
    {
        public Texture texture;
        public string error;
        public string dumpDir;
    }

    public string encoderParamRelativePath = "CodeFormer/models/encoder.param";
    public string encoderBinRelativePath = "CodeFormer/models/encoder.bin";
    public string generatorParamRelativePath = "CodeFormer/models/generator.param";
    public string generatorBinRelativePath = "CodeFormer/models/generator.bin";
    public int maxInputLongSide = 2048;
    public float faceMaskThreshold = 0.2f;
    public float faceBoxExpand = 0.35f;
    public bool enableTempPool = false;
    public int maxPooledPerShape = 2;
    public bool enableWinograd23 = false;
    public bool enableDebugDump = false;
    public bool enableFaceRegionDebugDump = false;
    [Range(0f, 1f)] public float codeFormerSftMulScale = 1f;
    [Range(0f, 1f)] public float codeFormerSftAddScale = 1f;
    public bool codeFormerBypassSftMul = false;
    public bool codeFormerOnlyTargetLastSftBlock = true;

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnRepro2 _encoderRepro;
    private NcnnRepro2 _generatorRepro;
    private bool _encoderLoaded;
    private bool _generatorLoaded;
    private bool _loaded;
    private string _lastDumpDir;
    public string LastDumpDir => _lastDumpDir;

    private void Awake()
    {
        EnsureRuntimeObjects();
    }

    private void OnDestroy()
    {
        Release();
    }

    public async UniTask<CodeFormerResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (src == null)
            return default;

        var totalSw = Stopwatch.StartNew();
        var originalW0 = src.width;
        var originalH0 = src.height;

        CodeFormerResult Finish(CodeFormerResult r)
        {
            r.elapsedMs = totalSw.ElapsedMilliseconds;
            try
            {
                UnityEngine.Debug.Log("[TIMING] CodeFormer(repro2) " + r.elapsedMs + " ms | in=" + originalW0 + "x" + originalH0 + " | err=" + (r.error ?? "") + " | dump=" + (_lastDumpDir ?? ""));
            }
            catch
            {
            }
            return r;
        }

        try
        {
            EnsureRuntimeObjects();
            ApplyReproOptions();
            if (enableDebugDump)
            {
                NcnnGpuResourceTracker.Enabled = true;
                NcnnGpuResourceTracker.Reset("CodeFormerNcnnReproRunner2");
            }
            await EnsureLoaded();
            if (_encoderRepro.Model == null || _generatorRepro.Model == null)
                return Finish(new CodeFormerResult { error = "CodeFormer(repro2) model unavailable" });

            var originalW = src.width;
            var originalH = src.height;
            var maxSide = Mathf.Max(originalW, originalH);
            var limit = Mathf.Max(256, maxInputLongSide);

            ReportProgress(0f, "Prepare input");
            await UniTask.Yield();

            NcnnFaceRegionGenerator faceRegion = null;
            Texture2D scaled = null;
            Texture2D workingTex = null;
            try
            {
                var inputTex = src;
                float scaleDown = 1f;
                if (maxSide > limit)
                {
                    ReportProgress(0.02f, "Scale down");
                    await UniTask.Yield();
                    scaleDown = (float)limit / maxSide;
                    var sw = Mathf.Max(1, Mathf.RoundToInt(originalW * scaleDown));
                    var sh = Mathf.Max(1, Mathf.RoundToInt(originalH * scaleDown));
                    scaled = ResizeTextureBilinear(src, sw, sh);
                    if (scaled == null)
                        return Finish(new CodeFormerResult { error = "Scale input failed" });
                    inputTex = scaled;
                }

                ReportProgress(0.06f, "Detect face area");
                await UniTask.Yield();
                faceRegion = GetComponent<NcnnFaceRegionGenerator>();
                if (faceRegion == null)
                    faceRegion = gameObject.AddComponent<NcnnFaceRegionGenerator>();
                FaceRegionFace[] faces = null;
                if (faceRegion != null && faceRegion.enabled)
                {
                    var rr = await faceRegion.GenerateAsync(inputTex, enableFaceRegionDebugDump, ct);
                    if (string.IsNullOrWhiteSpace(rr.error) && rr.faces != null && rr.faces.Length > 0)
                    {
                        faces = rr.faces;
                        if (enableFaceRegionDebugDump && !string.IsNullOrWhiteSpace(rr.dumpDir))
                            UnityEngine.Debug.Log("[CodeFormer Repro] face region dump: " + rr.dumpDir);
                    }
                }

                if (faces == null || faces.Length == 0)
                    return Finish(new CodeFormerResult { error = "No face detected" });

                workingTex = CopyTexture(inputTex);
                if (workingTex == null)
                    return Finish(new CodeFormerResult { error = "Working texture copy failed" });

                var restoredFaceCount = 0;
                for (var i = 0; i < faces.Length; i++)
                {
                    ct.ThrowIfCancellationRequested();
                    var face = faces[i];
                    ReportProgress(0.12f + 0.70f * ((float)i / Mathf.Max(1, faces.Length)), "Restore face " + (i + 1) + "/" + faces.Length);
                    await UniTask.Yield();

                    Texture2D alignedFaceTex = null;
                    RenderTexture alignedFaceRt = null;
                    Texture restoredFaceGpu = null;
                    Texture2D restoredFaceTex = null;
                    try
                    {
                        alignedFaceTex = AlignFaceToTemplate(inputTex, face);
                        if (alignedFaceTex == null)
                            continue;

                        alignedFaceRt = ResizeTextureBilinear((Texture)alignedFaceTex, 512, 512);
                        if (alignedFaceRt == null)
                            continue;

                        var runResult = await RunCodeFormer512Async(alignedFaceRt, ct);
                        if (!string.IsNullOrWhiteSpace(runResult.dumpDir))
                            _lastDumpDir = runResult.dumpDir;
                        restoredFaceGpu = runResult.texture;
                        if (restoredFaceGpu == null)
                        {
                            if (!string.IsNullOrWhiteSpace(runResult.error))
                                UnityEngine.Debug.LogWarning("[CodeFormer Repro] face " + i + " failed: " + runResult.error);
                            continue;
                        }

                        restoredFaceTex = TextureToTexture2D(restoredFaceGpu, 512, 512);
                        if (restoredFaceTex == null)
                            continue;

                        PasteAlignedFaceInPlace(workingTex, restoredFaceTex, face);
                        restoredFaceCount++;
                    }
                    finally
                    {
                        DestroyObjectSafe(alignedFaceTex);
                        if (alignedFaceRt != null) ReleaseTemporaryRt(alignedFaceRt);
                        if (restoredFaceGpu != null) ReleaseTextureIfTemporary(restoredFaceGpu);
                        DestroyObjectSafe(restoredFaceTex);
                    }
                }

                if (restoredFaceCount == 0)
                    return Finish(new CodeFormerResult { error = "CodeFormer(repro2) inference failed for all faces" });

                Texture2D finalTex = workingTex;
                workingTex = null;
                if (Mathf.Abs(scaleDown - 1f) > 1e-6f)
                {
                    ReportProgress(0.95f, "Restore original size");
                    await UniTask.Yield();
                    var resized = ResizeTextureBilinear(finalTex, originalW, originalH);
                    DestroyObjectSafe(finalTex);
                    finalTex = resized;
                    if (finalTex == null)
                        return Finish(new CodeFormerResult { error = "Upscale to original size failed" });
                }

                finalTex.wrapMode = TextureWrapMode.Clamp;
                finalTex.filterMode = FilterMode.Bilinear;
                finalTex.name = "CodeFormer_Repro2";
                ReportProgress(1f, "Done");
                await UniTask.Yield();
                return Finish(new CodeFormerResult { texture = finalTex });
            }
            finally
            {
                DestroyObjectSafe(scaled);
                DestroyObjectSafe(workingTex);
                if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                    NcnnGpuResourceTracker.WriteReport(_lastDumpDir, "codeformer_gpu_resources.txt");
                _encoderRepro?.ClearTempPool();
                _generatorRepro?.ClearTempPool();
            }
        }
        catch (Exception e)
        {
            return Finish(new CodeFormerResult { error = e.Message });
        }
    }

    private async UniTask<CodeFormer512RunResult> RunCodeFormer512Async(RenderTexture face512, CancellationToken ct)
    {
        if (face512 == null || face512.width != 512 || face512.height != 512)
            return new CodeFormer512RunResult { error = "Invalid face input: expected 512x512 render texture" };

        RenderTexture encoderInput = null;
        RenderTexture restored = null;
        RenderTexture encFeat32 = null;
        RenderTexture encFeat64 = null;
        RenderTexture encFeat128 = null;
        RenderTexture encFeat256 = null;
        RenderTexture lqFeat = null;
        NcnnTensorBuffer minEncodingTensor = null;
        var stage = "init";
        string dumpDir = null;

        try
        {
            ct.ThrowIfCancellationRequested();

            stage = "prepare encoder input";
            ReportProgress(0.18f, "Run encoder");
            await UniTask.Yield();

            encoderInput = _encoderRepro.RentTempArray(512, 512, 1, RenderTextureFormat.ARGBHalf);
            _ops.PackRgbToPack4Gfpgan(face512, 0, 0, 1, 1, encoderInput);

            var pinned = new HashSet<string>(StringComparer.Ordinal)
            {
                "enc_feat_32",
                "enc_feat_64",
                "enc_feat_128",
                "enc_feat_256",
                "lq_feat",
                "soft_one_hot"
            };
            if (enableDebugDump)
            {
                pinned.Add("1293");
                pinned.Add("1302");
                pinned.Add("1305");
                pinned.Add("1316");
                pinned.Add("1414");
                pinned.Add("1451");
                pinned.Add("1452");
                pinned.Add("1548");
                pinned.Add("1549");
                pinned.Add("1684");
                pinned.Add("1819");
                pinned.Add("1954");
                pinned.Add("2089");
                pinned.Add("2224");
                pinned.Add("2359");
                pinned.Add("2494");
                pinned.Add("2520");
                pinned.Add("2531");
                pinned.Add("2533");
            }

            stage = "encoder inference";
            using (var encoderResult = _encoderRepro.Infer(encoderInput, 1, "input", pinned))
            {
                stage = "extract encoder blob enc_feat_32";
                encFeat32 = encoderResult.ExtractTexture("enc_feat_32");
                stage = "extract encoder blob enc_feat_64";
                encFeat64 = encoderResult.ExtractTexture("enc_feat_64");
                stage = "extract encoder blob enc_feat_128";
                encFeat128 = encoderResult.ExtractTexture("enc_feat_128");
                stage = "extract encoder blob enc_feat_256";
                encFeat256 = encoderResult.ExtractTexture("enc_feat_256");
                stage = "extract encoder blob lq_feat";
                lqFeat = encoderResult.ExtractTexture("lq_feat");

                stage = "read encoder blob soft_one_hot";
                var softOneHot = encoderResult.GetBufferData("soft_one_hot");
                stage = "convert soft_one_hot to min encoding";
                minEncodingTensor = ConvertSoftOneHotToMinEncodingTensor(softOneHot);

                if (enableDebugDump)
                {
                    dumpDir = CreateDumpDir();
                    stage = "dump encoder tail";
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "29_enc_1293.png", "1293", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "30_enc_1302.png", "1302", 256, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "31_enc_1305.png", "1305", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "32_enc_1316.png", "1316", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "33_enc_1414.png", "1414", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "33b_enc_1451.png", "1451", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "33c_enc_1452.png", "1452", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "33d_enc_1548.png", "1548", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "34_enc_1549.png", "1549", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "35_enc_1684.png", "1684", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "36_enc_1819.png", "1819", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "37_enc_1954.png", "1954", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "38_enc_2089.png", "2089", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "39_enc_2224.png", "2224", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "40_enc_2359.png", "2359", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "41_enc_2494.png", "2494", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "42_enc_2520.png", "2520", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "43_enc_2531.png", "2531", 512, 256, ct);
                    await DumpBufferBlobAsNormalizedImageAsync(encoderResult, dumpDir, "44_enc_2533.png", "2533", 1024, 256, ct);
                    await DumpFloatArrayAsNormalizedImageAsync(dumpDir, "45_soft_one_hot.png", softOneHot, 1024, 256, ct);
                    AppendMatrixStatsLine(dumpDir, "soft_one_hot", softOneHot, 1024, 256, true);
                    AppendBinaryPatternStatsLine(dumpDir, "min_encoding", minEncodingTensor, 1024, 256);
                }
            }

            if (enableDebugDump)
            {
                stage = "dump encoder stages";
                await DumpRgbTextureAsync(dumpDir, "00_face512.png", face512, ct);
                await DumpPack4TextureAsync(dumpDir, "01_enc_feat_32.png", encFeat32, ct);
                await DumpPack4TextureAsync(dumpDir, "02_enc_feat_64.png", encFeat64, ct);
                await DumpPack4TextureAsync(dumpDir, "03_enc_feat_128.png", encFeat128, ct);
                await DumpPack4TextureAsync(dumpDir, "04_enc_feat_256.png", encFeat256, ct);
                await DumpPack4TextureAsync(dumpDir, "05_lq_feat.png", lqFeat, ct);
                await DumpBinaryTensorAsImageAsync(dumpDir, "06_min_encoding.png", minEncodingTensor, 1024, 256, ct);
            }

            if (encFeat32 == null || encFeat64 == null || encFeat128 == null || encFeat256 == null || lqFeat == null || minEncodingTensor == null)
            {
                return new CodeFormer512RunResult
                {
                    error = "CodeFormer(repro2) encoder outputs incomplete"
                        + " | enc_feat_32=" + BoolText(encFeat32 != null)
                        + " enc_feat_64=" + BoolText(encFeat64 != null)
                        + " enc_feat_128=" + BoolText(encFeat128 != null)
                        + " enc_feat_256=" + BoolText(encFeat256 != null)
                        + " lq_feat=" + BoolText(lqFeat != null)
                        + " min_encoding=" + BoolText(minEncodingTensor != null),
                    dumpDir = dumpDir
                };
            }

            stage = "prepare generator inputs";
            ReportProgress(0.4f, "Run generator");
            await UniTask.Yield();

            var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
            {
                { "enc_feat_32", encFeat32 },
                { "enc_feat_64", encFeat64 },
                { "enc_feat_128", encFeat128 },
                { "enc_feat_256", encFeat256 },
                { "style_feat", lqFeat }
            };

            var bufferInputs = new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal)
            {
                { "input", minEncodingTensor }
            };

            stage = "generator inference";
            HashSet<string> generatorPinned = null;
            if (enableDebugDump)
            {
                generatorPinned = new HashSet<string>(StringComparer.Ordinal)
                {
                    "548",
                    "549",
                    "554",
                    "556",
                    "564",
                    "579",
                    "683",
                    "698",
                    "1383",
                    "1383_splitncnn_1",
                    "1425",
                    "1425_splitncnn_0",
                    "1454",
                    "1417",
                    "1420",
                    "1421",
                    "1422",
                    "1453",
                    "1028",
                    "1033",
                    "1064",
                    "1246",
                    "1459",
                    "out"
                };
            }

            using (var generatorResult = _generatorRepro.InferWithMultiInputs(textureInputs, bufferInputs, generatorPinned))
            {
                if (enableDebugDump)
                {
                    stage = "dump generator stages";
                    try { await DumpBinaryTensorAsImageAsync(dumpDir, "07_input_min_encoding.png", minEncodingTensor, 1024, 256, ct); } catch (Exception e) { UnityEngine.Debug.LogWarning("[CodeFormer(repro2)] dump skip 07_input_min_encoding | " + e.Message); }
                    try { await DumpPack4TextureAsync(dumpDir, "08_style_feat_lq.png", lqFeat, ct); } catch (Exception e) { UnityEngine.Debug.LogWarning("[CodeFormer(repro2)] dump skip 08_style_feat_lq | " + e.Message); }
                    try { await DumpPack4TextureAsync(dumpDir, "09_input_enc_feat_256.png", encFeat256, ct); } catch (Exception e) { UnityEngine.Debug.LogWarning("[CodeFormer(repro2)] dump skip 09_input_enc_feat_256 | " + e.Message); }
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "548");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "549");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "554");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "556");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "564");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "579");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "683");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "698");
                    await DumpInferBlobAsync(generatorResult, dumpDir, "09b_blob_1383.png", "1383", ct);
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1383");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1383_splitncnn_1");
                    await DumpInferBlobAsync(generatorResult, dumpDir, "09c_blob_1425.png", "1425", ct);
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1425");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1425_splitncnn_0");
                    await DumpInferBlobAsync(generatorResult, dumpDir, "09d_blob_1454.png", "1454", ct);
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1454");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1417");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1420");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1421");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1422");
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1453");
                    await DumpInferBlobAsync(generatorResult, dumpDir, "10_blob_1028.png", "1028", ct);
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1028");
                    await DumpInferBlobAsync(generatorResult, dumpDir, "11_blob_1033.png", "1033", ct);
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1033");
                    await DumpInferBlobAsync(generatorResult, dumpDir, "12_blob_1064.png", "1064", ct);
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1064");
                    await DumpInferBlobAsync(generatorResult, dumpDir, "13_blob_1246.png", "1246", ct);
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1246");
                    await DumpInferBlobAsync(generatorResult, dumpDir, "14_blob_1459.png", "1459", ct);
                    await DumpInferBlobStatsAsync(generatorResult, dumpDir, "1459");
                }

                stage = "extract generator blob out";
                var outputTex = generatorResult.ExtractTexture("out");
                if (outputTex == null)
                {
                    return new CodeFormer512RunResult { error = "CodeFormer(repro2) generator output blob 'out' is null", dumpDir = dumpDir };
                }

                if (enableDebugDump)
                {
                    stage = "dump generator output pack4";
                    await DumpPack4TextureAsync(dumpDir, "15_out_pack4.png", outputTex, ct);
                    await AppendPack4TextureStatsAsync(dumpDir, "out_pack4", outputTex);
                }

                stage = "clip generator output";
                var clipTex = _generatorRepro.RentTempArray(outputTex.width, outputTex.height, 1, RenderTextureFormat.ARGBHalf);
                _ops.ClipPack4(outputTex, -1f, 1f, 1, clipTex);

                stage = "convert output to RGB";
                restored = GetTemporaryRt(outputTex.width, outputTex.height, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear, true);
                _ops.Pack4ToRgb01(clipTex, restored);

                if (enableDebugDump)
                {
                    stage = "dump generator output rgb";
                    await DumpRgbTextureAsync(dumpDir, "16_out_rgb.png", restored, ct);
                    TryOpenFolderInShell(dumpDir);
                }

                _generatorRepro.ReturnTempArray(clipTex);
            }

            return new CodeFormer512RunResult { texture = restored, dumpDir = dumpDir };
        }
        catch (OperationCanceledException)
        {
            if (restored != null) ReleaseTemporaryRt(restored);
            return default;
        }
        catch (Exception e)
        {
            if (restored != null) ReleaseTemporaryRt(restored);
            try
            {
                UnityEngine.Debug.LogError("[CodeFormer(repro2)] stage failed: " + stage);
                UnityEngine.Debug.LogException(e);
            }
            catch
            {
            }
            return new CodeFormer512RunResult
            {
                error = "CodeFormer(repro2) failed at " + stage + " | " + e.GetType().Name + ": " + e.Message,
                dumpDir = dumpDir
            };
        }
        finally
        {
            if (encoderInput != null) _encoderRepro.ReturnTempArray(encoderInput);
            if (encFeat32 != null) _encoderRepro.ReturnTempArray(encFeat32);
            if (encFeat64 != null) _encoderRepro.ReturnTempArray(encFeat64);
            if (encFeat128 != null) _encoderRepro.ReturnTempArray(encFeat128);
            if (encFeat256 != null) _encoderRepro.ReturnTempArray(encFeat256);
            if (lqFeat != null) _encoderRepro.ReturnTempArray(lqFeat);
            minEncodingTensor?.Dispose();
        }
    }

    private static string BoolText(bool value)
    {
        return value ? "ok" : "missing";
    }

    private async UniTask DumpInferBlobAsync(NcnnRepro2.InferResult inferResult, string dir, string fileName, string blobName, CancellationToken ct)
    {
        if (!enableDebugDump || inferResult == null || string.IsNullOrWhiteSpace(dir))
            return;

        try
        {
            var tex = inferResult.GetTexture(blobName);
            await DumpPack4TextureAsync(dir, fileName, tex, ct);
        }
        catch (Exception e)
        {
            try { UnityEngine.Debug.LogWarning("[CodeFormer(repro2)] dump skip blob " + blobName + " | " + e.Message); } catch { }
        }
    }

    private async UniTask DumpInferBlobStatsAsync(NcnnRepro2.InferResult inferResult, string dir, string blobName)
    {
        if (!enableDebugDump || inferResult == null || string.IsNullOrWhiteSpace(dir))
            return;

        try
        {
            try
            {
                if (inferResult.TryGetLogicalShape(blobName, out var dims, out var w, out var h, out var d, out var c))
                {
                    var path = Path.Combine(dir, "generator_stats.txt");
                    File.AppendAllText(path, blobName + " | view=dims" + dims + " w=" + w + " h=" + h + " d=" + d + " c=" + c + Environment.NewLine);

                    if (dims == 1 || inferResult.GetBuffer(blobName) != null)
                    {
                        var data = inferResult.GetBufferData(blobName);
                        AppendStatsLineTo(path, blobName, data);
                        if (dims == 1)
                            AppendMatrixStatsLineTo(path, blobName, data, w, 1, false);
                        else if (dims == 2)
                            AppendMatrixStatsLineTo(path, blobName, data, w, h, false);
                        else if (dims == 3)
                            AppendMatrixStatsLineTo(path, blobName, data, w * c, h, false);
                        else
                            AppendMatrixStatsLineTo(path, blobName, data, w * c, h * d, false);
                    }
                    else
                    {
                        var tex = inferResult.GetTexture(blobName);
                        await AppendPack4TextureStatsAsync(dir, blobName, tex, dims, w, h, d, c);
                    }
                }
                else
                {
                    var tex = inferResult.GetTexture(blobName);
                    await AppendPack4TextureStatsAsync(dir, blobName, tex);
                }
            }
            catch
            {
                var tex = inferResult.GetTexture(blobName);
                await AppendPack4TextureStatsAsync(dir, blobName, tex);
            }
        }
        catch (Exception e)
        {
            try { UnityEngine.Debug.LogWarning("[CodeFormer(repro2)] dump skip blob stats " + blobName + " | " + e.Message); } catch { }
        }
    }

    private async UniTask DumpPack4TextureAsync(string dir, string fileName, RenderTexture pack4Tex, CancellationToken ct)
    {
        if (!enableDebugDump || pack4Tex == null || string.IsNullOrWhiteSpace(dir))
            return;

        var vis = GetTemporaryRt(pack4Tex.width, pack4Tex.height, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear, true);
        try
        {
            _ops.Pack4ToRgb01(pack4Tex, vis);
            await DumpRgbTextureAsync(dir, fileName, vis, ct);
        }
        finally
        {
            ReleaseTemporaryRt(vis);
        }
    }

    private async UniTask AppendPack4TextureStatsAsync(string dir, string blobName, RenderTexture pack4Tex)
    {
        await AppendPack4TextureStatsAsync(dir, blobName, pack4Tex, 3, pack4Tex.width, pack4Tex.height, 1, (pack4Tex.volumeDepth > 0 ? pack4Tex.volumeDepth : 1) * 4);
    }

    private async UniTask AppendPack4TextureStatsAsync(string dir, string blobName, RenderTexture pack4Tex, int dims, int logicalW, int logicalH, int logicalD, int logicalC)
    {
        if (!enableDebugDump || pack4Tex == null || string.IsNullOrWhiteSpace(dir))
            return;

        var channels = (pack4Tex.volumeDepth > 0 ? pack4Tex.volumeDepth : 1) * 4;
        var total = pack4Tex.width * pack4Tex.height * channels;
        var buffer = new ComputeBuffer(total, sizeof(float), ComputeBufferType.Structured);
        try
        {
            _ops.Pack4ToBufferCHW(pack4Tex, pack4Tex.width, pack4Tex.height, channels, buffer);
            var data = new float[total];
            buffer.GetData(data);
            var path = Path.Combine(dir, "generator_stats.txt");
            AppendStatsLineTo(path, blobName, data);
            if (dims == 1)
                AppendMatrixStatsLineTo(path, blobName, data, logicalW, 1, false);
            else if (dims == 2)
                AppendMatrixStatsLineTo(path, blobName, data, logicalW, logicalH, false);
            else if (dims == 3)
                AppendMatrixStatsLineTo(path, blobName, data, logicalW * logicalC, logicalH, false);
            else
                AppendMatrixStatsLineTo(path, blobName, data, logicalW * logicalC, logicalH * logicalD, false);
        }
        finally
        {
            buffer.Dispose();
        }
    }

    private async UniTask DumpRgbTextureAsync(string dir, string fileName, Texture texture, CancellationToken ct)
    {
        if (!enableDebugDump || texture == null || string.IsNullOrWhiteSpace(dir))
            return;

        RenderTexture rt = texture as RenderTexture;
        RenderTexture tempRt = null;
        try
        {
            if (rt == null)
            {
                tempRt = GetTemporaryRt(texture.width, texture.height, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear, false);
                Graphics.Blit(texture, tempRt);
                rt = tempRt;
            }

            var tex2D = await ReadbackTextureAsync(rt, rt.width, rt.height, ct);
            if (tex2D == null)
                return;
            try
            {
                var bytes = tex2D.EncodeToPNG();
                await File.WriteAllBytesAsync(Path.Combine(dir, fileName), bytes, ct);
            }
            finally
            {
                DestroyObjectSafe(tex2D);
            }
        }
        finally
        {
            if (tempRt != null)
                ReleaseTemporaryRt(tempRt);
        }
    }

    private async UniTask DumpBinaryTensorAsImageAsync(string dir, string fileName, NcnnTensorBuffer tensor, int width, int height, CancellationToken ct)
    {
        if (!enableDebugDump || tensor == null || tensor.buffer == null || string.IsNullOrWhiteSpace(dir))
            return;

        var data = new float[tensor.buffer.count];
        tensor.buffer.GetData(data);
        var tex = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
        try
        {
            var pixels = new Color32[width * height];
            for (var i = 0; i < pixels.Length && i < data.Length; i++)
            {
                var v = data[i] > 0.5f ? (byte)255 : (byte)0;
                pixels[i] = new Color32(v, v, v, 255);
            }
            tex.SetPixels32(pixels);
            tex.Apply(false, false);
            var bytes = tex.EncodeToPNG();
            await File.WriteAllBytesAsync(Path.Combine(dir, fileName), bytes, ct);
        }
        finally
        {
            DestroyObjectSafe(tex);
        }
    }

    private async UniTask DumpBufferBlobAsNormalizedImageAsync(NcnnRepro2.InferResult inferResult, string dir, string fileName, string blobName, int width, int height, CancellationToken ct)
    {
        if (!enableDebugDump || inferResult == null || string.IsNullOrWhiteSpace(dir))
            return;

        try
        {
            var data = inferResult.GetBufferData(blobName);
            await DumpFloatArrayAsNormalizedImageAsync(dir, fileName, data, width, height, ct);
            AppendStatsLine(dir, blobName, data);
            AppendMatrixStatsLine(dir, blobName, data, width, height, false);
        }
        catch (Exception e)
        {
            try { UnityEngine.Debug.LogWarning("[CodeFormer(repro2)] dump skip buffer " + blobName + " | " + e.Message); } catch { }
        }
    }

    private async UniTask DumpFloatArrayAsNormalizedImageAsync(string dir, string fileName, float[] data, int width, int height, CancellationToken ct)
    {
        if (!enableDebugDump || string.IsNullOrWhiteSpace(dir) || data == null || data.Length < width * height)
            return;

        float min = float.PositiveInfinity;
        float max = float.NegativeInfinity;
        var finiteCount = 0;
        for (var i = 0; i < width * height; i++)
        {
            var v = data[i];
            if (float.IsNaN(v) || float.IsInfinity(v))
                continue;
            finiteCount++;
            if (v < min) min = v;
            if (v > max) max = v;
        }

        if (finiteCount == 0)
            return;

        var scale = Mathf.Abs(max - min) > 1e-12f ? 1f / (max - min) : 0f;
        var tex = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
        try
        {
            var pixels = new Color32[width * height];
            for (var i = 0; i < width * height; i++)
            {
                var v = data[i];
                byte c;
                if (float.IsNaN(v))
                    c = 255;
                else if (float.IsInfinity(v))
                    c = 255;
                else
                    c = (byte)Mathf.Clamp(Mathf.RoundToInt((v - min) * scale * 255f), 0, 255);
                pixels[i] = new Color32(c, c, c, 255);
            }
            tex.SetPixels32(pixels);
            tex.Apply(false, false);
            var bytes = tex.EncodeToPNG();
            await File.WriteAllBytesAsync(Path.Combine(dir, fileName), bytes, ct);
        }
        finally
        {
            DestroyObjectSafe(tex);
        }
    }

    private static void AppendStatsLine(string dir, string blobName, float[] data)
    {
        if (string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(blobName) || data == null || data.Length == 0)
            return;

        try
        {
            double sum = 0d;
            double sq = 0d;
            var finite = 0;
            var nan = 0;
            var inf = 0;
            float min = float.PositiveInfinity;
            float max = float.NegativeInfinity;
            for (var i = 0; i < data.Length; i++)
            {
                var v = data[i];
                if (float.IsNaN(v))
                {
                    nan++;
                    continue;
                }
                if (float.IsInfinity(v))
                {
                    inf++;
                    continue;
                }
                finite++;
                sum += v;
                sq += v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }

            var mean = finite > 0 ? sum / finite : 0d;
            var var = finite > 0 ? Math.Max(0d, sq / finite - mean * mean) : 0d;
            var std = Math.Sqrt(var);
            var line = blobName
                + " | count=" + data.Length
                + " finite=" + finite
                + " nan=" + nan
                + " inf=" + inf
                + " min=" + min.ToString("G9")
                + " max=" + max.ToString("G9")
                + " mean=" + mean.ToString("G9")
                + " std=" + std.ToString("G9")
                + Environment.NewLine;
            File.AppendAllText(Path.Combine(dir, "encoder_stats.txt"), line);
        }
        catch
        {
        }
    }

    private static void AppendStatsLineTo(string path, string blobName, float[] data)
    {
        if (string.IsNullOrWhiteSpace(path) || string.IsNullOrWhiteSpace(blobName) || data == null || data.Length == 0)
            return;

        try
        {
            double sum = 0d;
            double sq = 0d;
            var finite = 0;
            var nan = 0;
            var inf = 0;
            float min = float.PositiveInfinity;
            float max = float.NegativeInfinity;
            for (var i = 0; i < data.Length; i++)
            {
                var v = data[i];
                if (float.IsNaN(v))
                {
                    nan++;
                    continue;
                }
                if (float.IsInfinity(v))
                {
                    inf++;
                    continue;
                }
                finite++;
                sum += v;
                sq += v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }

            var mean = finite > 0 ? sum / finite : 0d;
            var var = finite > 0 ? Math.Max(0d, sq / finite - mean * mean) : 0d;
            var std = Math.Sqrt(var);
            var line = blobName
                + " | count=" + data.Length
                + " finite=" + finite
                + " nan=" + nan
                + " inf=" + inf
                + " min=" + min.ToString("G9")
                + " max=" + max.ToString("G9")
                + " mean=" + mean.ToString("G9")
                + " std=" + std.ToString("G9")
                + Environment.NewLine;
            File.AppendAllText(path, line);
        }
        catch
        {
        }
    }

    private static void AppendMatrixStatsLine(string dir, string blobName, float[] data, int width, int height, bool treatAsProbability)
    {
        if (string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(blobName) || data == null || width <= 0 || height <= 0 || data.Length < width * height)
            return;

        try
        {
            var argmaxCounts = new Dictionary<int, int>();
            double rowMaxSum = 0d;
            double rowSecondSum = 0d;
            double rowGapSum = 0d;
            double rowEntropySum = 0d;
            for (var y = 0; y < height; y++)
            {
                var rowBase = y * width;
                var maxIndex = 0;
                var maxValue = float.NegativeInfinity;
                var secondValue = float.NegativeInfinity;
                double entropy = 0d;
                double rowSum = 0d;

                for (var x = 0; x < width; x++)
                {
                    var v = data[rowBase + x];
                    if (v > maxValue)
                    {
                        secondValue = maxValue;
                        maxValue = v;
                        maxIndex = x;
                    }
                    else if (v > secondValue)
                    {
                        secondValue = v;
                    }

                    if (treatAsProbability && v > 0f && !float.IsNaN(v) && !float.IsInfinity(v))
                    {
                        rowSum += v;
                        entropy -= v * Math.Log(Math.Max(v, 1e-30f));
                    }
                }

                if (!argmaxCounts.TryGetValue(maxIndex, out var count))
                    count = 0;
                argmaxCounts[maxIndex] = count + 1;

                rowMaxSum += maxValue;
                rowSecondSum += secondValue;
                rowGapSum += maxValue - secondValue;
                if (treatAsProbability)
                {
                    if (rowSum > 0d)
                        rowEntropySum += entropy;
                }
            }

            var topBins = new List<KeyValuePair<int, int>>(argmaxCounts);
            topBins.Sort((a, b) => b.Value != a.Value ? b.Value.CompareTo(a.Value) : a.Key.CompareTo(b.Key));
            var topSummary = "";
            var take = Math.Min(8, topBins.Count);
            for (var i = 0; i < take; i++)
            {
                if (i > 0) topSummary += ",";
                topSummary += topBins[i].Key + ":" + topBins[i].Value;
            }

            var line = blobName
                + " | matrix=" + width + "x" + height
                + " unique_argmax=" + argmaxCounts.Count
                + " avg_row_max=" + (rowMaxSum / height).ToString("G9")
                + " avg_row_second=" + (rowSecondSum / height).ToString("G9")
                + " avg_row_gap=" + (rowGapSum / height).ToString("G9")
                + (treatAsProbability ? " avg_row_entropy=" + (rowEntropySum / height).ToString("G9") : "")
                + " top_argmax_bins=" + topSummary
                + Environment.NewLine;
            File.AppendAllText(Path.Combine(dir, "encoder_stats.txt"), line);
        }
        catch
        {
        }
    }

    private static void AppendMatrixStatsLineTo(string path, string blobName, float[] data, int width, int height, bool treatAsProbability)
    {
        if (string.IsNullOrWhiteSpace(path) || string.IsNullOrWhiteSpace(blobName) || data == null || width <= 0 || height <= 0 || data.Length < width * height)
            return;

        try
        {
            var argmaxCounts = new Dictionary<int, int>();
            double rowMaxSum = 0d;
            double rowSecondSum = 0d;
            double rowGapSum = 0d;
            double rowEntropySum = 0d;
            for (var y = 0; y < height; y++)
            {
                var rowBase = y * width;
                var maxIndex = 0;
                var maxValue = float.NegativeInfinity;
                var secondValue = float.NegativeInfinity;
                double entropy = 0d;
                double rowSum = 0d;

                for (var x = 0; x < width; x++)
                {
                    var v = data[rowBase + x];
                    if (v > maxValue)
                    {
                        secondValue = maxValue;
                        maxValue = v;
                        maxIndex = x;
                    }
                    else if (v > secondValue)
                    {
                        secondValue = v;
                    }

                    if (treatAsProbability && v > 0f && !float.IsNaN(v) && !float.IsInfinity(v))
                    {
                        rowSum += v;
                        entropy -= v * Math.Log(Math.Max(v, 1e-30f));
                    }
                }

                if (!argmaxCounts.TryGetValue(maxIndex, out var count))
                    count = 0;
                argmaxCounts[maxIndex] = count + 1;

                rowMaxSum += maxValue;
                rowSecondSum += secondValue;
                rowGapSum += maxValue - secondValue;
                if (treatAsProbability && rowSum > 0d)
                    rowEntropySum += entropy;
            }

            var topBins = new List<KeyValuePair<int, int>>(argmaxCounts);
            topBins.Sort((a, b) => b.Value != a.Value ? b.Value.CompareTo(a.Value) : a.Key.CompareTo(b.Key));
            var topSummary = "";
            var take = Math.Min(8, topBins.Count);
            for (var i = 0; i < take; i++)
            {
                if (i > 0) topSummary += ",";
                topSummary += topBins[i].Key + ":" + topBins[i].Value;
            }

            var line = blobName
                + " | matrix=" + width + "x" + height
                + " unique_argmax=" + argmaxCounts.Count
                + " avg_row_max=" + (rowMaxSum / height).ToString("G9")
                + " avg_row_second=" + (rowSecondSum / height).ToString("G9")
                + " avg_row_gap=" + (rowGapSum / height).ToString("G9")
                + (treatAsProbability ? " avg_row_entropy=" + (rowEntropySum / height).ToString("G9") : "")
                + " top_argmax_bins=" + topSummary
                + Environment.NewLine;
            File.AppendAllText(path, line);
        }
        catch
        {
        }
    }

    private static void AppendBinaryPatternStatsLine(string dir, string blobName, NcnnTensorBuffer tensor, int width, int height)
    {
        if (string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(blobName) || tensor == null || tensor.buffer == null)
            return;

        try
        {
            var data = new float[tensor.buffer.count];
            tensor.buffer.GetData(data);
            var activeCount = 0;
            var activeRows = 0;
            var activeCols = new HashSet<int>();
            var topCols = new Dictionary<int, int>();
            for (var y = 0; y < height; y++)
            {
                var rowActive = 0;
                var rowBase = y * width;
                for (var x = 0; x < width; x++)
                {
                    if (data[rowBase + x] > 0.5f)
                    {
                        activeCount++;
                        rowActive++;
                        activeCols.Add(x);
                        if (!topCols.TryGetValue(x, out var count))
                            count = 0;
                        topCols[x] = count + 1;
                    }
                }
                if (rowActive > 0)
                    activeRows++;
            }

            var bins = new List<KeyValuePair<int, int>>(topCols);
            bins.Sort((a, b) => b.Value != a.Value ? b.Value.CompareTo(a.Value) : a.Key.CompareTo(b.Key));
            var topSummary = "";
            var take = Math.Min(8, bins.Count);
            for (var i = 0; i < take; i++)
            {
                if (i > 0) topSummary += ",";
                topSummary += bins[i].Key + ":" + bins[i].Value;
            }

            var line = blobName
                + " | binary=" + width + "x" + height
                + " active_count=" + activeCount
                + " active_rows=" + activeRows
                + " unique_active_cols=" + activeCols.Count
                + " top_active_cols=" + topSummary
                + Environment.NewLine;
            File.AppendAllText(Path.Combine(dir, "encoder_stats.txt"), line);
        }
        catch
        {
        }
    }

    private static async UniTask<Texture2D> ReadbackTextureAsync(RenderTexture rt, int w, int h, CancellationToken ct)
    {
        var tcs = new UniTaskCompletionSource<AsyncGPUReadbackRequest>();
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, req => tcs.TrySetResult(req));
        var request = await tcs.Task.AttachExternalCancellation(ct);
        if (request.hasError)
            return null;

        var data = request.GetData<byte>();
        var tex = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
        tex.LoadRawTextureData(data);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
    }

    private static string CreateDumpDir()
    {
        var root = Application.temporaryCachePath;
        if (string.IsNullOrWhiteSpace(root))
            root = Path.GetTempPath();
        var dir = Path.Combine(root, "AIImage_CodeFormerRepro2_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
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
            try { Directory.CreateDirectory(directoryPath); } catch { }
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            Process.Start(new ProcessStartInfo { FileName = directoryPath, UseShellExecute = true });
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
            Process.Start(new ProcessStartInfo("open", directoryPath) { UseShellExecute = false });
#elif UNITY_STANDALONE_LINUX
            Process.Start(new ProcessStartInfo("xdg-open", directoryPath) { UseShellExecute = false });
#else
            var url = "file://" + directoryPath.Replace('\\', '/');
            Application.OpenURL(url);
#endif
        }
        catch
        {
        }
    }
#endif

    private NcnnTensorBuffer ConvertSoftOneHotToMinEncodingTensor(float[] softOneHot)
    {
        const int codebookSize = 1024;
        const int tokenCount = 256;
        if (softOneHot == null || softOneHot.Length < codebookSize * tokenCount)
            throw new InvalidOperationException("soft_one_hot buffer size mismatch");

        var minEncodings = new float[codebookSize * tokenCount];
        for (var token = 0; token < tokenCount; token++)
        {
            var rowStart = token * codebookSize;
            var maxIndex = 0;
            var maxValue = softOneHot[rowStart];
            for (var i = 1; i < codebookSize; i++)
            {
                var value = softOneHot[rowStart + i];
                if (value > maxValue)
                {
                    maxValue = value;
                    maxIndex = i;
                }
            }
            minEncodings[rowStart + maxIndex] = 1f;
        }

        var tensor = new NcnnTensorBuffer(codebookSize, tokenCount);
        tensor.buffer.SetData(minEncodings);
        return tensor;
    }

    private void ApplyReproOptions()
    {
        if (_encoderRepro == null || _generatorRepro == null)
            return;

        _encoderRepro.EnableTempPool = enableTempPool;
        _generatorRepro.EnableTempPool = enableTempPool;
        _encoderRepro.MaxPooledPerShape = maxPooledPerShape;
        _generatorRepro.MaxPooledPerShape = maxPooledPerShape;
        _encoderRepro.EnableWinograd23 = enableWinograd23;
        _generatorRepro.EnableWinograd23 = enableWinograd23;
        _encoderRepro.CodeFormerSftMulScale = 1f;
        _generatorRepro.CodeFormerSftMulScale = Mathf.Clamp01(codeFormerSftMulScale);
        _encoderRepro.CodeFormerSftAddScale = 1f;
        _generatorRepro.CodeFormerSftAddScale = Mathf.Clamp01(codeFormerSftAddScale);
        _encoderRepro.CodeFormerBypassSftMul = false;
        _generatorRepro.CodeFormerBypassSftMul = codeFormerBypassSftMul;
        _encoderRepro.CodeFormerTargetSftMulLayer = null;
        _encoderRepro.CodeFormerTargetSftAddLayer = null;
        _encoderRepro.CodeFormerTargetSftResidualLayer = null;
        _generatorRepro.CodeFormerTargetSftMulLayer = codeFormerOnlyTargetLastSftBlock ? "Mul_900" : null;
        _generatorRepro.CodeFormerTargetSftAddLayer = codeFormerOnlyTargetLastSftBlock ? "Add_901" : null;
        _generatorRepro.CodeFormerTargetSftResidualLayer = codeFormerOnlyTargetLastSftBlock ? "Add_904" : null;
    }

    private async UniTask EnsureLoaded()
    {
        if (_loaded)
            return;

        await LoadEncoderAsync();
        await LoadGeneratorAsync();
        _loaded = true;
    }

    private async UniTask LoadEncoderAsync()
    {
        if (_encoderLoaded)
            return;
        EnsureRuntimeObjects();

        var paramPath = Path.Combine(Application.streamingAssetsPath, encoderParamRelativePath);
        var binPath = Path.Combine(Application.streamingAssetsPath, encoderBinRelativePath);
        if (!File.Exists(paramPath))
            throw new InvalidOperationException("Missing encoder param: " + paramPath);
        if (!File.Exists(binPath))
            throw new InvalidOperationException("Missing encoder bin: " + binPath);

        ReportProgress(0.02f, "Load encoder");
        await UniTask.Yield();

        var paramText = await File.ReadAllTextAsync(paramPath);
        var bytes = await File.ReadAllBytesAsync(binPath);
        using (var ms = new MemoryStream(bytes))
        using (var br = new NcnnBinReader(ms))
        {
            _encoderRepro.LoadModel(paramText, br);
        }
        _encoderLoaded = true;
    }

    private async UniTask LoadGeneratorAsync()
    {
        if (_generatorLoaded)
            return;
        EnsureRuntimeObjects();

        var paramPath = Path.Combine(Application.streamingAssetsPath, generatorParamRelativePath);
        var binPath = Path.Combine(Application.streamingAssetsPath, generatorBinRelativePath);
        if (!File.Exists(paramPath))
            throw new InvalidOperationException("Missing generator param: " + paramPath);
        if (!File.Exists(binPath))
            throw new InvalidOperationException("Missing generator bin: " + binPath);

        ReportProgress(0.08f, "Load generator");
        await UniTask.Yield();

        var paramText = await File.ReadAllTextAsync(paramPath);
        var bytes = await File.ReadAllBytesAsync(binPath);
        using (var ms = new MemoryStream(bytes))
        using (var br = new NcnnBinReader(ms))
        {
            _generatorRepro.LoadModel(paramText, br);
        }
        _generatorLoaded = true;
    }

    private void Release()
    {
        try { _encoderRepro?.Dispose(); } catch { }
        try { _generatorRepro?.Dispose(); } catch { }
        _encoderRepro = null;
        _generatorRepro = null;
        _loaded = false;
        _encoderLoaded = false;
        _generatorLoaded = false;
    }

    private void EnsureRuntimeObjects()
    {
        if (_ops == null)
            _ops = new NcnnOps();
        if (_encoderRepro == null)
            _encoderRepro = new NcnnRepro2(_ops);
        if (_generatorRepro == null)
            _generatorRepro = new NcnnRepro2(_ops);
        ApplyReproOptions();
    }

    private void ReportProgress(float progress01, string text)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(progress01), text ?? ""); } catch { }
    }

    private static Texture2D CopyTexture(Texture2D src)
    {
        if (src == null)
            return null;
        var tex = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false, true);
        tex.SetPixels32(src.GetPixels32());
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
    }

    private static void DestroyObjectSafe(UnityEngine.Object obj)
    {
        if (obj == null)
            return;
        if (Application.isPlaying)
            Destroy(obj);
        else
            DestroyImmediate(obj);
    }

    private static Texture2D AlignFaceToTemplate(Texture2D src, FaceRegionFace face)
    {
        if (src == null)
            return null;
        var srcPixels = src.GetPixels32();
        var sourceLandmarks = ResolveAlignmentLandmarks(face, src.width, src.height);
        if (sourceLandmarks == null || sourceLandmarks.Length < 5)
            return null;
        if (!SolveRobustSimilarityTransform(sourceLandmarks, CanonicalFivePointTemplateBottomOrigin, out var sourceToAligned))
            return null;
        if (!sourceToAligned.TryInverse(out var alignedToSource))
            return null;

        var dst = new Texture2D(512, 512, TextureFormat.RGBA32, false, true);
        var dstPixels = new Color32[512 * 512];
        var border = new Color32(135, 133, 132, 255);
        for (var y = 0; y < 512; y++)
        {
            for (var x = 0; x < 512; x++)
            {
                var sourcePos = alignedToSource.Transform(new Vector2(x + 0.5f, y + 0.5f));
                dstPixels[y * 512 + x] = BilinearSampleClamp(srcPixels, src.width, src.height, sourcePos.x - 0.5f, sourcePos.y - 0.5f, border);
            }
        }

        dst.SetPixels32(dstPixels);
        dst.Apply(false, false);
        dst.wrapMode = TextureWrapMode.Clamp;
        dst.filterMode = FilterMode.Bilinear;
        return dst;
    }

    private static void PasteAlignedFaceInPlace(Texture2D baseTex, Texture2D restoredFace, FaceRegionFace face)
    {
        if (baseTex == null || restoredFace == null)
            return;
        var sourceLandmarks = ResolveAlignmentLandmarks(face, baseTex.width, baseTex.height);
        if (sourceLandmarks == null || sourceLandmarks.Length < 5)
            return;
        if (!SolveRobustSimilarityTransform(sourceLandmarks, CanonicalFivePointTemplateBottomOrigin, out var sourceToAligned))
            return;
        if (!sourceToAligned.TryInverse(out var alignedToSource))
            return;

        // Match the official paste path's one-pixel translation tweak before warping back.
        var alignedToSourceShifted = new Affine2D(
            alignedToSource.m00,
            alignedToSource.m01,
            alignedToSource.m02 + 1f,
            alignedToSource.m10,
            alignedToSource.m11,
            alignedToSource.m12 + 1f,
            true);
        if (!alignedToSourceShifted.TryInverse(out var pasteSampleTransform))
            return;

        var basePixels = baseTex.GetPixels32();
        var facePixels = restoredFace.GetPixels32();
        var imageW = baseTex.width;
        var imageH = baseTex.height;
        var warpedBounds = ComputeWarpedAlignedBounds(alignedToSourceShifted, imageW, imageH);
        if (warpedBounds.width <= 0 || warpedBounds.height <= 0)
            return;

        var invRestored = new Color32[imageW * imageH];
        var invMask = new byte[imageW * imageH];
        FillWarpedFaceAndMask(facePixels, restoredFace.width, restoredFace.height, pasteSampleTransform, warpedBounds, imageW, imageH, invRestored, invMask);

        var invMaskErosion = new byte[imageW * imageH];
        var firstErodeBounds = ExpandRectPixels(warpedBounds, imageW, imageH, 4);
        ErodeMask(invMask, invMaskErosion, imageW, imageH, BuildEllipseKernelOffsets(4, 4), firstErodeBounds);

        var totalFaceArea = CountNonZero(invMaskErosion, firstErodeBounds, imageW);
        if (totalFaceArea <= 0)
            return;

        var pastedFace = new Color32[imageW * imageH];
        MaskFace(invRestored, invMaskErosion, pastedFace, firstErodeBounds, imageW);

        var wEdge = Mathf.Max(0, Mathf.FloorToInt(Mathf.Sqrt(totalFaceArea) / 20f));
        var erosionRadius = wEdge * 2;
        var invMaskCenter = new byte[imageW * imageH];
        if (erosionRadius >= 2)
        {
            var centerBounds = ExpandRectPixels(warpedBounds, imageW, imageH, erosionRadius + 2);
            ErodeMask(invMaskErosion, invMaskCenter, imageW, imageH, BuildEllipseKernelOffsets(erosionRadius, erosionRadius), centerBounds);
        }
        else
        {
            Array.Copy(invMaskErosion, invMaskCenter, invMaskErosion.Length);
        }

        var blurSize = wEdge * 2;
        var blurKernelSize = Mathf.Max(1, blurSize + 1);
        if ((blurKernelSize & 1) == 0)
            blurKernelSize += 1;
        var blurRadius = blurKernelSize / 2;
        var blendBounds = ExpandRectPixels(warpedBounds, imageW, imageH, blurRadius * 2 + 4);
        var invSoftMask = new float[imageW * imageH];
        GaussianBlurMaskToFloat(invMaskCenter, imageW, imageH, blurKernelSize, blendBounds, invSoftMask);

        BlendPastedFace(basePixels, pastedFace, invSoftMask, blendBounds, imageW);

        baseTex.SetPixels32(basePixels);
        baseTex.Apply(false, false);
    }

    private static RectInt ComputeWarpedAlignedBounds(Affine2D alignedToSource, int imageW, int imageH)
    {
        var corners = new[]
        {
            alignedToSource.Transform(new Vector2(0f, 0f)),
            alignedToSource.Transform(new Vector2(511f, 0f)),
            alignedToSource.Transform(new Vector2(0f, 511f)),
            alignedToSource.Transform(new Vector2(511f, 511f))
        };

        var minX = float.PositiveInfinity;
        var minY = float.PositiveInfinity;
        var maxX = float.NegativeInfinity;
        var maxY = float.NegativeInfinity;
        for (var i = 0; i < corners.Length; i++)
        {
            minX = Mathf.Min(minX, corners[i].x);
            minY = Mathf.Min(minY, corners[i].y);
            maxX = Mathf.Max(maxX, corners[i].x);
            maxY = Mathf.Max(maxY, corners[i].y);
        }

        var rect = new RectInt(
            Mathf.FloorToInt(minX) - 2,
            Mathf.FloorToInt(minY) - 2,
            Mathf.CeilToInt(maxX) - Mathf.FloorToInt(minX) + 4,
            Mathf.CeilToInt(maxY) - Mathf.FloorToInt(minY) + 4);
        return ClampRect(rect, imageW, imageH);
    }

    private static void FillWarpedFaceAndMask(
        Color32[] facePixels,
        int faceW,
        int faceH,
        Affine2D pasteSampleTransform,
        RectInt bounds,
        int imageW,
        int imageH,
        Color32[] invRestored,
        byte[] invMask)
    {
        if (facePixels == null || invRestored == null || invMask == null)
            return;

        for (var y = bounds.yMin; y < bounds.yMax; y++)
        {
            for (var x = bounds.xMin; x < bounds.xMax; x++)
            {
                var alignedPos = pasteSampleTransform.Transform(new Vector2(x + 0.5f, y + 0.5f));
                var sampleX = alignedPos.x - 0.5f;
                var sampleY = alignedPos.y - 0.5f;
                if (sampleX < 0f || sampleY < 0f || sampleX > faceW - 1f || sampleY > faceH - 1f)
                    continue;

                var idx = y * imageW + x;
                invMask[idx] = 255;
                invRestored[idx] = BilinearSampleClamp(facePixels, faceW, faceH, sampleX, sampleY, new Color32(0, 0, 0, 255));
            }
        }
    }

    private static void MaskFace(Color32[] invRestored, byte[] invMask, Color32[] pastedFace, RectInt bounds, int imageW)
    {
        if (invRestored == null || invMask == null || pastedFace == null)
            return;

        for (var y = bounds.yMin; y < bounds.yMax; y++)
        {
            for (var x = bounds.xMin; x < bounds.xMax; x++)
            {
                var idx = y * imageW + x;
                if (invMask[idx] > 0)
                    pastedFace[idx] = invRestored[idx];
            }
        }
    }

    private static RectInt ExpandRectPixels(RectInt rect, int imageW, int imageH, int padding)
    {
        if (rect.width <= 0 || rect.height <= 0)
            return rect;
        return ClampRect(
            new RectInt(rect.x - padding, rect.y - padding, rect.width + padding * 2, rect.height + padding * 2),
            imageW,
            imageH);
    }

    private static Vector2Int[] BuildEllipseKernelOffsets(int kernelW, int kernelH)
    {
        if (kernelW <= 0 || kernelH <= 0)
            return Array.Empty<Vector2Int>();

        var offsets = new List<Vector2Int>(kernelW * kernelH);
        var originX = kernelW / 2;
        var originY = kernelH / 2;
        var halfW = Mathf.Max(0.5f, kernelW * 0.5f);
        var halfH = Mathf.Max(0.5f, kernelH * 0.5f);
        for (var y = 0; y < kernelH; y++)
        {
            for (var x = 0; x < kernelW; x++)
            {
                var dx = (x + 0.5f - halfW) / halfW;
                var dy = (y + 0.5f - halfH) / halfH;
                if (dx * dx + dy * dy <= 1f)
                    offsets.Add(new Vector2Int(x - originX, y - originY));
            }
        }

        return offsets.Count > 0 ? offsets.ToArray() : new[] { Vector2Int.zero };
    }

    private static void ErodeMask(byte[] src, byte[] dst, int imageW, int imageH, Vector2Int[] kernelOffsets, RectInt bounds)
    {
        if (src == null || dst == null || kernelOffsets == null || bounds.width <= 0 || bounds.height <= 0)
            return;

        for (var y = bounds.yMin; y < bounds.yMax; y++)
        {
            for (var x = bounds.xMin; x < bounds.xMax; x++)
            {
                var keep = true;
                for (var i = 0; i < kernelOffsets.Length; i++)
                {
                    var sx = x + kernelOffsets[i].x;
                    var sy = y + kernelOffsets[i].y;
                    if (sx < 0 || sy < 0 || sx >= imageW || sy >= imageH || src[sy * imageW + sx] == 0)
                    {
                        keep = false;
                        break;
                    }
                }

                dst[y * imageW + x] = keep ? (byte)255 : (byte)0;
            }
        }
    }

    private static int CountNonZero(byte[] src, RectInt bounds, int imageW)
    {
        if (src == null || bounds.width <= 0 || bounds.height <= 0)
            return 0;

        var count = 0;
        for (var y = bounds.yMin; y < bounds.yMax; y++)
        {
            for (var x = bounds.xMin; x < bounds.xMax; x++)
            {
                if (src[y * imageW + x] != 0)
                    count++;
            }
        }

        return count;
    }

    private static void GaussianBlurMaskToFloat(byte[] src, int imageW, int imageH, int kernelSize, RectInt bounds, float[] dst)
    {
        if (src == null || dst == null || bounds.width <= 0 || bounds.height <= 0)
            return;

        if (kernelSize <= 1)
        {
            for (var y = bounds.yMin; y < bounds.yMax; y++)
            {
                for (var x = bounds.xMin; x < bounds.xMax; x++)
                    dst[y * imageW + x] = src[y * imageW + x] / 255f;
            }
            return;
        }

        var radius = kernelSize / 2;
        var kernel = BuildGaussianKernel1D(kernelSize);
        var temp = new float[imageW * imageH];
        var tempBounds = ExpandRectPixels(bounds, imageW, imageH, radius);

        for (var y = tempBounds.yMin; y < tempBounds.yMax; y++)
        {
            for (var x = bounds.xMin; x < bounds.xMax; x++)
            {
                float sum = 0f;
                for (var k = -radius; k <= radius; k++)
                {
                    var sx = Reflect101Index(x + k, imageW);
                    sum += src[y * imageW + sx] * kernel[k + radius];
                }

                temp[y * imageW + x] = sum;
            }
        }

        for (var y = bounds.yMin; y < bounds.yMax; y++)
        {
            for (var x = bounds.xMin; x < bounds.xMax; x++)
            {
                float sum = 0f;
                for (var k = -radius; k <= radius; k++)
                {
                    var sy = Reflect101Index(y + k, imageH);
                    sum += temp[sy * imageW + x] * kernel[k + radius];
                }

                dst[y * imageW + x] = sum / 255f;
            }
        }
    }

    private static float[] BuildGaussianKernel1D(int kernelSize)
    {
        var kernel = new float[kernelSize];
        if (kernelSize <= 1)
        {
            kernel[0] = 1f;
            return kernel;
        }

        var radius = kernelSize / 2;
        var sigma = 0.3f * ((kernelSize - 1) * 0.5f - 1f) + 0.8f;
        sigma = Mathf.Max(1e-3f, sigma);
        var invTwoSigmaSq = 1f / (2f * sigma * sigma);
        var sum = 0f;
        for (var i = 0; i < kernelSize; i++)
        {
            var x = i - radius;
            var w = Mathf.Exp(-(x * x) * invTwoSigmaSq);
            kernel[i] = w;
            sum += w;
        }

        if (sum > 1e-8f)
        {
            var inv = 1f / sum;
            for (var i = 0; i < kernel.Length; i++)
                kernel[i] *= inv;
        }

        return kernel;
    }

    private static int Reflect101Index(int index, int size)
    {
        if (size <= 1)
            return 0;

        while (index < 0 || index >= size)
        {
            if (index < 0)
                index = -index;
            else
                index = size * 2 - index - 2;
        }

        return index;
    }

    private static void BlendPastedFace(Color32[] basePixels, Color32[] pastedFace, float[] softMask, RectInt bounds, int imageW)
    {
        if (basePixels == null || pastedFace == null || softMask == null)
            return;

        for (var y = bounds.yMin; y < bounds.yMax; y++)
        {
            for (var x = bounds.xMin; x < bounds.xMax; x++)
            {
                var idx = y * imageW + x;
                var alpha = Mathf.Clamp01(softMask[idx]);
                if (alpha <= 1e-5f)
                    continue;

                var face = pastedFace[idx];
                var src = basePixels[idx];
                basePixels[idx] = new Color32(
                    (byte)Mathf.Clamp(Mathf.RoundToInt(src.r * (1f - alpha) + face.r * alpha), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(src.g * (1f - alpha) + face.g * alpha), 0, 255),
                    (byte)Mathf.Clamp(Mathf.RoundToInt(src.b * (1f - alpha) + face.b * alpha), 0, 255),
                    255);
            }
        }
    }

    private static Vector2[] ResolveAlignmentLandmarks(FaceRegionFace face, int imgW, int imgH)
    {
        if (face.landmarks == null || face.landmarks.Length < 5)
            return BuildFallbackLandmarks(face.rectInt, imgW, imgH);

        if (!SolveRobustSimilarityTransform(face.landmarks, CanonicalFivePointTemplateBottomOrigin, out var sourceToAligned))
            return BuildFallbackLandmarks(face.rectInt, imgW, imgH);

        if (IsAlignmentTransformReasonable(sourceToAligned, face.rect))
            return face.landmarks;

        return BuildFallbackLandmarks(face.rectInt, imgW, imgH);
    }

    private static bool IsAlignmentTransformReasonable(Affine2D sourceToAligned, Rect faceRect)
    {
        if (!sourceToAligned.valid || faceRect.width <= 1f || faceRect.height <= 1f)
            return false;

        var scale = Mathf.Sqrt(sourceToAligned.m00 * sourceToAligned.m00 + sourceToAligned.m10 * sourceToAligned.m10);
        var referenceFaceSize = Mathf.Max(8f, Mathf.Max(faceRect.width, faceRect.height));
        var expectedScale = 512f / referenceFaceSize;
        return scale <= expectedScale * 1.35f;
    }

    private static Vector2[] BuildFallbackLandmarks(RectInt rect, int imgW, int imgH)
    {
        if (rect.width <= 1 || rect.height <= 1)
            return null;

        var expanded = ExpandRect(rect, imgW, imgH, 0.08f);
        var x = expanded.xMin;
        var y = expanded.yMin;
        var w = Mathf.Max(1, expanded.width);
        var h = Mathf.Max(1, expanded.height);

        return new[]
        {
            new Vector2(x + w * 0.35f, y + h * 0.62f),
            new Vector2(x + w * 0.65f, y + h * 0.62f),
            new Vector2(x + w * 0.50f, y + h * 0.46f),
            new Vector2(x + w * 0.39f, y + h * 0.27f),
            new Vector2(x + w * 0.61f, y + h * 0.27f)
        };
    }

    private static bool SolveRobustSimilarityTransform(IReadOnlyList<Vector2> src, IReadOnlyList<Vector2> dst, out Affine2D transform)
    {
        transform = default;
        if (src == null || dst == null || src.Count < 2 || src.Count != dst.Count)
            return false;

        var bestScore = float.PositiveInfinity;
        var bestMean = float.PositiveInfinity;
        var found = false;
        for (var skipIndex = -1; skipIndex < src.Count; skipIndex++)
        {
            if (!SolveSimilarityTransformLeastSquares(src, dst, skipIndex, out var candidate))
                continue;

            EvaluateTransformFit(candidate, src, dst, out var medianSqError, out var meanSqError);
            if (!found
                || medianSqError < bestScore - 1e-4f
                || (Mathf.Abs(medianSqError - bestScore) <= 1e-4f && meanSqError < bestMean - 1e-4f)
                || (Mathf.Abs(medianSqError - bestScore) <= 1e-4f && Mathf.Abs(meanSqError - bestMean) <= 1e-4f && skipIndex < 0))
            {
                bestScore = medianSqError;
                bestMean = meanSqError;
                transform = candidate;
                found = true;
            }
        }

        return found;
    }

    private static bool SolveSimilarityTransformLeastSquares(IReadOnlyList<Vector2> src, IReadOnlyList<Vector2> dst, int skipIndex, out Affine2D transform)
    {
        transform = default;
        if (src == null || dst == null || src.Count < 2 || src.Count != dst.Count)
            return false;

        var ata = new float[4, 4];
        var atb = new float[4];
        var usedCount = 0;
        for (var i = 0; i < src.Count; i++)
        {
            if (i == skipIndex)
                continue;

            var x = src[i].x;
            var y = src[i].y;
            var u = dst[i].x;
            var v = dst[i].y;

            var r0 = new[] { x, -y, 1f, 0f };
            var r1 = new[] { y, x, 0f, 1f };
            AccumulateNormalEquation(ata, atb, r0, u);
            AccumulateNormalEquation(ata, atb, r1, v);
            usedCount++;
        }

        if (usedCount < 2)
            return false;
        if (!SolveLinear4x4(ata, atb, out var s))
            return false;

        transform = new Affine2D(s[0], -s[1], s[2], s[1], s[0], s[3], true);
        return true;
    }

    private static void EvaluateTransformFit(Affine2D transform, IReadOnlyList<Vector2> src, IReadOnlyList<Vector2> dst, out float medianSqError, out float meanSqError)
    {
        medianSqError = float.PositiveInfinity;
        meanSqError = float.PositiveInfinity;
        if (!transform.valid || src == null || dst == null || src.Count == 0 || src.Count != dst.Count)
            return;

        var errors = new float[src.Count];
        var sum = 0f;
        for (var i = 0; i < src.Count; i++)
        {
            var delta = transform.Transform(src[i]) - dst[i];
            var errSq = delta.sqrMagnitude;
            errors[i] = errSq;
            sum += errSq;
        }

        Array.Sort(errors);
        medianSqError = errors[errors.Length / 2];
        meanSqError = sum / errors.Length;
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

    private static Color32 BilinearSampleClamp(Color32[] src, int srcW, int srcH, float x, float y, Color32 border)
    {
        if (src == null || src.Length == 0 || srcW <= 0 || srcH <= 0)
            return border;
        if (x < 0f || y < 0f || x > srcW - 1f || y > srcH - 1f)
            return border;

        var x0 = Mathf.Clamp(Mathf.FloorToInt(x), 0, srcW - 1);
        var y0 = Mathf.Clamp(Mathf.FloorToInt(y), 0, srcH - 1);
        var x1 = Mathf.Clamp(x0 + 1, 0, srcW - 1);
        var y1 = Mathf.Clamp(y0 + 1, 0, srcH - 1);
        var tx = Mathf.Clamp01(x - x0);
        var ty = Mathf.Clamp01(y - y0);

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

    private static Color32 LerpColor(Color32 a, Color32 b, float t)
    {
        t = Mathf.Clamp01(t);
        return new Color32(
            (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(a.r, b.r, t)), 0, 255),
            (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(a.g, b.g, t)), 0, 255),
            (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(a.b, b.b, t)), 0, 255),
            255);
    }

    private static RectInt ClampRect(RectInt r, int w, int h)
    {
        var x0 = Mathf.Clamp(r.x, 0, Mathf.Max(0, w));
        var y0 = Mathf.Clamp(r.y, 0, Mathf.Max(0, h));
        var x1 = Mathf.Clamp(r.x + r.width, 0, Mathf.Max(0, w));
        var y1 = Mathf.Clamp(r.y + r.height, 0, Mathf.Max(0, h));
        return new RectInt(x0, y0, Mathf.Max(0, x1 - x0), Mathf.Max(0, y1 - y0));
    }

    private static RectInt FindFaceRect(Texture2D faceMask, int imgW, int imgH, float threshold)
    {
        if (faceMask == null)
            return new RectInt(0, 0, imgW, imgH);

        var maskPixels = faceMask.GetPixels32();
        var minX = int.MaxValue;
        var minY = int.MaxValue;
        var maxX = int.MinValue;
        var maxY = int.MinValue;
        var found = false;

        for (var y = 0; y < faceMask.height; y++)
        {
            for (var x = 0; x < faceMask.width; x++)
            {
                var px = maskPixels[y * faceMask.width + x];
                if (px.r / 255f >= threshold)
                {
                    found = true;
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (!found)
            return new RectInt(0, 0, imgW, imgH);

        var scaleX = (float)imgW / faceMask.width;
        var scaleY = (float)imgH / faceMask.height;
        minX = Mathf.FloorToInt(minX * scaleX);
        minY = Mathf.FloorToInt(minY * scaleY);
        maxX = Mathf.CeilToInt(maxX * scaleX);
        maxY = Mathf.CeilToInt(maxY * scaleY);
        return ClampRect(new RectInt(minX, minY, maxX - minX, maxY - minY), imgW, imgH);
    }

    private static RectInt ExpandRect(RectInt rect, int imgW, int imgH, float expand)
    {
        var dw = Mathf.RoundToInt(rect.width * expand);
        var dh = Mathf.RoundToInt(rect.height * expand);
        return ClampRect(new RectInt(rect.x - dw, rect.y - dh, rect.width + dw * 2, rect.height + dh * 2), imgW, imgH);
    }

    private static Texture2D CropTexture(Texture2D src, RectInt rect)
    {
        rect = ClampRect(rect, src.width, src.height);
        if (rect.width <= 0 || rect.height <= 0)
            return null;

        var dst = new Texture2D(rect.width, rect.height, TextureFormat.RGBA32, false, true);
        var srcPixels = src.GetPixels32();
        var dstPixels = new Color32[rect.width * rect.height];
        for (var y = 0; y < rect.height; y++)
        {
            Array.Copy(srcPixels, (rect.y + y) * src.width + rect.x, dstPixels, y * rect.width, rect.width);
        }
        dst.SetPixels32(dstPixels);
        dst.Apply(false, false);
        dst.wrapMode = TextureWrapMode.Clamp;
        dst.filterMode = FilterMode.Bilinear;
        return dst;
    }

    private static RenderTexture ResizeTextureBilinear(Texture src, int w, int h)
    {
        var ret = GetTemporaryRt(w, h, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default, false);
        Graphics.Blit(src, ret);
        return ret;
    }

    private static Texture2D ResizeTextureBilinear(Texture2D src, int w, int h)
    {
        if (src == null)
            return null;
        var rt = ResizeTextureBilinear((Texture)src, w, h);
        var result = RenderTextureToTexture2D(rt, w, h);
        ReleaseTemporaryRt(rt);
        return result;
    }

    private static Texture2D RenderTextureToTexture2D(RenderTexture rt, int w, int h)
    {
        if (rt == null)
            return null;
        var prev = RenderTexture.active;
        RenderTexture.active = rt;
        var tex = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
        tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        RenderTexture.active = prev;
        return tex;
    }

    private static Texture2D TextureToTexture2D(Texture texture, int w, int h)
    {
        if (texture == null)
            return null;
        if (texture is Texture2D tex2D && tex2D.width == w && tex2D.height == h)
            return CopyTexture(tex2D);
        if (texture is RenderTexture rt)
            return RenderTextureToTexture2D(rt, w, h);

        var tmp = ResizeTextureBilinear(texture, w, h);
        var result = RenderTextureToTexture2D(tmp, w, h);
        ReleaseTemporaryRt(tmp);
        return result;
    }

    private static RenderTexture ComposeWithMask(Texture2D src, RenderTexture restored, Texture2D mask, RectInt rect)
    {
        if (src == null || restored == null)
            return null;

        var cs = Resources.Load<ComputeShader>("ImageProcessing");
        if (cs == null)
            return null;

        int kernel;
        try { kernel = cs.FindKernel("PasteRectWithMask"); } catch { return null; }
        if (kernel < 0)
            return null;

        var rt = GetTemporaryRt(src.width, src.height, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB, true);

        cs.SetTexture(kernel, "_Source", src);
        cs.SetTexture(kernel, "_Overlay", restored);
        cs.SetTexture(kernel, "_Result", rt);
        cs.SetInts("_CropRect", rect.x, rect.y, rect.width, rect.height);

        if (mask != null && mask.width == src.width && mask.height == src.height && mask.format == TextureFormat.RHalf)
            cs.SetTexture(kernel, "_FaceMaskIn", mask);
        else
            cs.SetTexture(kernel, "_FaceMaskIn", Texture2D.blackTexture);

        cs.Dispatch(kernel, Mathf.CeilToInt(src.width / 8f), Mathf.CeilToInt(src.height / 8f), 1);
        return rt;
    }

    private static RenderTexture GetTemporaryRt(int width, int height, RenderTextureFormat format, RenderTextureReadWrite readWrite, bool randomWrite)
    {
        var desc = new RenderTextureDescriptor(Mathf.Max(1, width), Mathf.Max(1, height), format, 0)
        {
            msaaSamples = 1,
            sRGB = readWrite != RenderTextureReadWrite.Linear,
            enableRandomWrite = randomWrite
        };
        var rt = RenderTexture.GetTemporary(desc);
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.filterMode = FilterMode.Bilinear;
        return rt;
    }

    private static void ReleaseTemporaryRt(RenderTexture rt)
    {
        if (rt == null)
            return;
        try { RenderTexture.ReleaseTemporary(rt); } catch { }
    }

    private static void ReleaseTextureIfTemporary(Texture texture)
    {
        if (texture is RenderTexture rt)
            ReleaseTemporaryRt(rt);
    }
}
