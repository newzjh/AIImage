using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class CodeFormerNcnnReproRunner2 : MonoBehaviour
{
    private struct CodeFormer512RunResult
    {
        public Texture texture;
        public string error;
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

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnRepro2 _encoderRepro;
    private NcnnRepro2 _generatorRepro;
    private bool _encoderLoaded;
    private bool _generatorLoaded;
    private bool _loaded;

    private void Awake()
    {
        _ops = new NcnnOps();
        _encoderRepro = new NcnnRepro2(_ops);
        _generatorRepro = new NcnnRepro2(_ops);
        ApplyReproOptions();
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
                UnityEngine.Debug.Log("[TIMING] CodeFormer(repro2) " + r.elapsedMs + " ms | in=" + originalW0 + "x" + originalH0 + " | err=" + (r.error ?? ""));
            }
            catch
            {
            }
            return r;
        }

        try
        {
            ApplyReproOptions();
            await EnsureLoaded();
            if (_encoderRepro.Model == null || _generatorRepro.Model == null)
                return Finish(new CodeFormerResult { error = "CodeFormer(repro2) model unavailable" });

            var originalW = src.width;
            var originalH = src.height;
            var maxSide = Mathf.Max(originalW, originalH);
            var limit = Mathf.Max(256, maxInputLongSide);

            ReportProgress(0f, "Prepare input");
            await UniTask.Yield();

            Texture2D scaled = null;
            Texture2D faceMask = null;
            Texture2D faceCrop = null;
            RenderTexture face512 = null;
            Texture restored512 = null;
            RenderTexture restoredCrop = null;
            RenderTexture composed = null;
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
                var fm = GetComponent<FaceMaskGenerator>();
                if (fm != null)
                {
                    var mr = await fm.GenerateForCurrentAsync(inputTex, false, ct);
                    if (string.IsNullOrWhiteSpace(mr.error) && mr.mask != null)
                        faceMask = mr.mask;
                }

                var rect = FindFaceRect(faceMask, inputTex.width, inputTex.height, faceMaskThreshold);
                rect = ExpandRect(rect, inputTex.width, inputTex.height, faceBoxExpand);
                if (rect.width <= 8 || rect.height <= 8)
                    rect = new RectInt(inputTex.width / 4, inputTex.height / 4, inputTex.width / 2, inputTex.height / 2);

                ReportProgress(0.1f, "Crop face");
                await UniTask.Yield();
                faceCrop = CropTexture(inputTex, rect);
                if (faceCrop == null)
                    return Finish(new CodeFormerResult { error = "Crop failed" });

                face512 = ResizeTextureBilinear((Texture)faceCrop, 512, 512);
                if (face512 == null)
                    return Finish(new CodeFormerResult { error = "Resize to 512 failed" });

                ReportProgress(0.15f, "Run CodeFormer");
                await UniTask.Yield();
                var runResult = await RunCodeFormer512Async(face512, ct);
                restored512 = runResult.texture;
                if (restored512 == null)
                    return Finish(new CodeFormerResult { error = string.IsNullOrWhiteSpace(runResult.error) ? "CodeFormer(repro2) inference failed" : runResult.error });

                ReportProgress(0.85f, "Paste back");
                await UniTask.Yield();
                restoredCrop = ResizeTextureBilinear(restored512, rect.width, rect.height);
                if (restoredCrop == null)
                    return Finish(new CodeFormerResult { error = "Resize restored crop failed" });

                composed = ComposeWithMask(inputTex, restoredCrop, faceMask, rect);
                if (composed == null)
                    return Finish(new CodeFormerResult { error = "Compose failed" });

                var composedTex = RenderTextureToTexture2D(composed, inputTex.width, inputTex.height);
                if (composedTex == null)
                    return Finish(new CodeFormerResult { error = "Readback failed" });

                Texture2D finalTex = composedTex;
                if (Mathf.Abs(scaleDown - 1f) > 1e-6f)
                {
                    ReportProgress(0.95f, "Restore original size");
                    await UniTask.Yield();
                    var resized = ResizeTextureBilinear(finalTex, originalW, originalH);
                    Destroy(finalTex);
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
                if (scaled != null) Destroy(scaled);
                if (faceMask != null) Destroy(faceMask);
                if (faceCrop != null) Destroy(faceCrop);
                if (face512 != null) Destroy(face512);
                if (restored512 != null) Destroy(restored512);
                if (restoredCrop != null) Destroy(restoredCrop);
                if (composed != null)
                {
                    composed.Release();
                    Destroy(composed);
                }
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
                        + " min_encoding=" + BoolText(minEncodingTensor != null)
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
            using (var generatorResult = _generatorRepro.InferWithMultiInputs(textureInputs, bufferInputs, null))
            {
                stage = "extract generator blob out";
                var outputTex = generatorResult.ExtractTexture("out");
                if (outputTex == null)
                {
                    return new CodeFormer512RunResult { error = "CodeFormer(repro2) generator output blob 'out' is null" };
                }

                stage = "clip generator output";
                var clipTex = _generatorRepro.RentTempArray(outputTex.width, outputTex.height, 1, RenderTextureFormat.ARGBHalf);
                _ops.ClipPack4(outputTex, -1f, 1f, 1, clipTex);

                stage = "convert output to RGB";
                restored = new RenderTexture(outputTex.width, outputTex.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
                restored.enableRandomWrite = true;
                restored.wrapMode = TextureWrapMode.Clamp;
                restored.filterMode = FilterMode.Bilinear;
                restored.Create();
                _ops.Pack4ToRgb01(clipTex, restored);

                _generatorRepro.ReturnTempArray(clipTex);
            }

            return new CodeFormer512RunResult { texture = restored };
        }
        catch (OperationCanceledException)
        {
            if (restored != null) Destroy(restored);
            return default;
        }
        catch (Exception e)
        {
            if (restored != null) Destroy(restored);
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
                error = "CodeFormer(repro2) failed at " + stage + " | " + e.GetType().Name + ": " + e.Message
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

    private void ReportProgress(float progress01, string text)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(progress01), text ?? ""); } catch { }
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
        var ret = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32);
        ret.Create();
        Graphics.Blit(src, ret);
        return ret;
    }

    private static Texture2D ResizeTextureBilinear(Texture2D src, int w, int h)
    {
        if (src == null)
            return null;
        var rt = ResizeTextureBilinear((Texture)src, w, h);
        var result = RenderTextureToTexture2D(rt, w, h);
        rt.Release();
        Destroy(rt);
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

    private static RenderTexture ComposeWithMask(Texture2D src, RenderTexture restored, Texture2D mask, RectInt rect)
    {
        var compRt = new RenderTexture(src.width, src.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        compRt.wrapMode = TextureWrapMode.Clamp;
        compRt.filterMode = FilterMode.Bilinear;
        compRt.Create();

        var prev = RenderTexture.active;
        RenderTexture.active = compRt;
        GL.Clear(true, true, Color.clear);
        RenderTexture.active = prev;

        Graphics.Blit(src, compRt);

        var shader = Shader.Find("Hidden/BlendWithMask");
        if (shader == null)
        {
            Graphics.Blit(restored, compRt, new Vector2((float)rect.width / compRt.width, (float)rect.height / compRt.height), new Vector2((float)rect.x / compRt.width, (float)rect.y / compRt.height));
            return compRt;
        }

        var material = new Material(shader);
        material.SetTexture("_FaceTex", restored);
        material.SetTexture("_MaskTex", mask != null ? mask : Texture2D.whiteTexture);
        material.SetVector("_FaceRect", new Vector4((float)rect.x / compRt.width, (float)rect.y / compRt.height, (float)rect.width / compRt.width, (float)rect.height / compRt.height));

        var tmp = RenderTexture.GetTemporary(compRt.descriptor);
        Graphics.Blit(compRt, tmp);
        Graphics.Blit(tmp, compRt, material);
        RenderTexture.ReleaseTemporary(tmp);
        Destroy(material);
        return compRt;
    }
}
