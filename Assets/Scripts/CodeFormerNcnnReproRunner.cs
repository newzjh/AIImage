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

public sealed class CodeFormerNcnnReproRunner : MonoBehaviour
{
    public string encoderParamRelativePath = "CodeFormer/models/encoder.param";
    public string encoderBinRelativePath = "CodeFormer/models/encoder.bin";
    public string generatorParamRelativePath = "CodeFormer/models/generator.param";
    public string generatorBinRelativePath = "CodeFormer/models/generator.bin";
    public int maxInputLongSide = 2048;
    public float faceMaskThreshold = 0.2f;
    public float faceBoxExpand = 0.35f;
    public bool enableTempPool = false;
    public int maxPooledPerShape = 2;

    public event Action<float, string> ProgressChanged;

    private NcnnRepro _encoderRepro;
    private NcnnRepro _generatorRepro;
    private NcnnOps _ops;
    private bool _encoderLoaded;
    private bool _generatorLoaded;
    private bool _loaded;

    private void Awake()
    {
        _ops = new NcnnOps();
        _encoderRepro = new NcnnRepro(_ops);
        _generatorRepro = new NcnnRepro(_ops);
        _encoderRepro.useExperimentalIteratePath = true;
        _generatorRepro.useExperimentalIteratePath = true;
    }

    private void OnDestroy()
    {
        Release();
    }

    private void Release()
    {
        _encoderRepro?.Release();
        _generatorRepro?.Release();
        _loaded = false;
        _encoderLoaded = false;
        _generatorLoaded = false;
        try { _encoderRepro?.Dispose(); } catch { }
        _encoderRepro = null;
        try { _generatorRepro?.Dispose(); } catch { }
        _generatorRepro = null;
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
                UnityEngine.Debug.Log("[TIMING] CodeFormer(repro) " + r.elapsedMs + " ms | in=" + originalW0 + "x" + originalH0 + " | err=" + (r.error ?? ""));
            }
            catch { }
            return r;
        }

        await EnsureLoaded();
        if (_encoderRepro.Model == null || _generatorRepro.Model == null)
            return Finish(new CodeFormerResult { error = "CodeFormer(复刻) 模型不可用" });

        var originalW = src.width;
        var originalH = src.height;
        var maxSide = Mathf.Max(originalW, originalH);
        var limit = Mathf.Max(256, maxInputLongSide);

        ReportProgress(0f, "准备输入");
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
                ReportProgress(0.02f, "缩小到2k以内");
                await UniTask.Yield();
                scaleDown = (float)limit / maxSide;
                var sw = Mathf.Max(1, Mathf.RoundToInt(originalW * scaleDown));
                var sh = Mathf.Max(1, Mathf.RoundToInt(originalH * scaleDown));
                scaled = ResizeTextureBilinear(src, sw, sh);
                if (scaled == null)
                    return Finish(new CodeFormerResult { error = "缩放输入失败" });
                inputTex = scaled;
            }

            ReportProgress(0.06f, "生成脸部区域");
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

            ReportProgress(0.10f, "裁剪脸部");
            await UniTask.Yield();
            faceCrop = CropTexture(inputTex, rect);
            if (faceCrop == null)
                return Finish(new CodeFormerResult { error = "裁剪失败" });

            face512 = ResizeTextureBilinear((Texture)faceCrop, 512, 512);
            if (face512 == null)
                return Finish(new CodeFormerResult { error = "resize到512失败" });

            ReportProgress(0.15f, "推理中…");
            await UniTask.Yield();
            restored512 = await RunCodeFormer512Async(face512, ct);
            if (restored512 == null)
                return Finish(new CodeFormerResult { error = "CodeFormer(复刻) 推理失败" });

            ReportProgress(0.85f, "回贴到原图");
            await UniTask.Yield();
            restoredCrop = ResizeTextureBilinear(restored512, rect.width, rect.height);
            if (restoredCrop == null)
                return Finish(new CodeFormerResult { error = "回贴缩放失败" });

            composed = ComposeWithMask(inputTex, restoredCrop, faceMask, rect);
            if (composed == null)
                return Finish(new CodeFormerResult { error = "合成失败" });

            var composedTex = RenderTextureToTexture2D(composed, inputTex.width, inputTex.height);
            if (composedTex == null)
                return Finish(new CodeFormerResult { error = "合成回读失败" });

            Texture2D finalTex = composedTex;
            if (Mathf.Abs(scaleDown - 1f) > 1e-6f)
            {
                ReportProgress(0.95f, "回缩放到原分辨率");
                await UniTask.Yield();
                var resized = ResizeTextureBilinear(finalTex, originalW, originalH);
                Destroy(finalTex);
                finalTex = resized;
                if (finalTex == null)
                    return Finish(new CodeFormerResult { error = "回缩放失败" });
            }

            finalTex.wrapMode = TextureWrapMode.Clamp;
            finalTex.filterMode = FilterMode.Bilinear;
            finalTex.name = "CodeFormer_Repro";
            ReportProgress(1f, "完成");
            await UniTask.Yield();
            return Finish(new CodeFormerResult { texture = finalTex });
        }
        catch (Exception e)
        {
            if (IsLikelyVulkanOom(e))
                return Finish(new CodeFormerResult { error = "Vulkan - Out of device memory" });
            return Finish(new CodeFormerResult { error = e.Message });
        }
        finally
        {
            if (scaled != null) Destroy(scaled);
            if (faceMask != null) Destroy(faceMask);
            if (faceCrop != null) Destroy(faceCrop);
            if (face512 != null) Destroy(face512);
            if (restored512 != null) Destroy(restored512);
            if (restoredCrop != null) Destroy(restoredCrop);
            if (composed != null) { composed.Release(); Destroy(composed); }
            _encoderRepro.ClearTempPool();
            _generatorRepro.ClearTempPool();
        }
    }

    private async UniTask<Texture> RunCodeFormer512Async(RenderTexture face512, CancellationToken ct)
    {
        if (face512 == null || face512.width != 512 || face512.height != 512)
            return null;

        RenderTexture encoderInput = null;
        RenderTexture restored = null;
        RenderTexture styleFeatTex = null;
        RenderTexture styleFeatSplit0Tex = null;
        RenderTexture styleFeatSplit1Tex = null;
        RenderTexture lqFeatTex = null;
        RenderTexture encFeat32Tex = null;
        RenderTexture encFeat64Tex = null;
        RenderTexture encFeat128Tex = null;
        RenderTexture encFeat256Tex = null;

        try
        {
            ReportProgress(0.15f, "编码器推理…");
            await UniTask.Yield();

            encoderInput = _encoderRepro.RentTempArray(512, 512, 1, RenderTextureFormat.ARGBHalf);
            _ops.PackRgbToPack4Gfpgan(face512, 0, 0, 1, 1, encoderInput);

            var encoderOutputNames = new[] { "enc_feat_32", "enc_feat_64", "enc_feat_128", "enc_feat_256", "lq_feat", "soft_one_hot" };
            var pinned = new HashSet<string>(encoderOutputNames, StringComparer.Ordinal);

            var encoderInputName = _encoderRepro.Model.layers.Count > 0
                && _encoderRepro.Model.layers[0].topNames != null
                && _encoderRepro.Model.layers[0].topNames.Length > 0
                ? _encoderRepro.Model.layers[0].topNames[0]
                : "input";

            using (var encoderResult = _encoderRepro.Infer(encoderInput, 1, encoderInputName, pinned))
            {
                encFeat32Tex = encoderResult.ExtractTexture("enc_feat_32");
                encFeat64Tex = encoderResult.ExtractTexture("enc_feat_64");
                encFeat128Tex = encoderResult.ExtractTexture("enc_feat_128");
                encFeat256Tex = encoderResult.ExtractTexture("enc_feat_256");
                lqFeatTex = encoderResult.ExtractTexture("lq_feat");

                var softOneHot = encoderResult.GetBufferData("soft_one_hot");
                var styleFeat = ConvertSoftOneHotToStyleFeatTex(softOneHot);
                styleFeatTex = styleFeat.StyleFeat;
                styleFeatSplit0Tex = styleFeat.Split0;
                styleFeatSplit1Tex = styleFeat.Split1;
            }

            if (encFeat32Tex == null || encFeat64Tex == null || encFeat128Tex == null
                || encFeat256Tex == null || lqFeatTex == null || styleFeatTex == null
                || styleFeatSplit0Tex == null || styleFeatSplit1Tex == null)
                return null;

            ReportProgress(0.35f, "生成器推理…");
            await UniTask.Yield();

            var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
            {
                { "enc_feat_32", encFeat32Tex },
                { "enc_feat_64", encFeat64Tex },
                { "enc_feat_128", encFeat128Tex },
                { "enc_feat_256", encFeat256Tex },
                { "input", lqFeatTex },
                { "style_feat", styleFeatTex },
                { "style_feat_splitncnn_0", styleFeatSplit0Tex },
                { "style_feat_splitncnn_1", styleFeatSplit1Tex },
            };

            using (var generatorResult = _generatorRepro.InferWithMultiInputs(textureInputs, null))
            {
                var outputTex = generatorResult.ExtractTexture("out");
                if (outputTex == null)
                    return null;

                var clipTex = _encoderRepro.RentTempArray(outputTex.width, outputTex.height, 1, RenderTextureFormat.ARGBHalf);
                _ops.ClipPack4(outputTex, -1f, 1f, 1, clipTex);

                restored = new RenderTexture(outputTex.width, outputTex.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
                restored.enableRandomWrite = true;
                restored.wrapMode = TextureWrapMode.Clamp;
                restored.filterMode = FilterMode.Bilinear;
                restored.Create();
                _ops.Pack4ToRgb01(clipTex, restored);

                _encoderRepro.ReturnTempArray(clipTex);
            }

            return restored;
        }
        catch
        {
            if (restored != null) Destroy(restored);
            return null;
        }
        finally
        {
            if (encoderInput != null) _encoderRepro.ReturnTempArray(encoderInput);
            if (lqFeatTex != null) _encoderRepro.ReturnTempArray(lqFeatTex);
            if (encFeat32Tex != null) _encoderRepro.ReturnTempArray(encFeat32Tex);
            if (encFeat64Tex != null) _encoderRepro.ReturnTempArray(encFeat64Tex);
            if (encFeat128Tex != null) _encoderRepro.ReturnTempArray(encFeat128Tex);
            if (encFeat256Tex != null) _encoderRepro.ReturnTempArray(encFeat256Tex);
            try
            {
                if (styleFeatTex != null) Destroy(styleFeatTex);
                if (styleFeatSplit0Tex != null) Destroy(styleFeatSplit0Tex);
                if (styleFeatSplit1Tex != null) Destroy(styleFeatSplit1Tex);
            }
            catch { }
        }
    }

    private struct StyleFeatTexResult
    {
        public RenderTexture StyleFeat;
        public RenderTexture Split0;
        public RenderTexture Split1;
    }

    private StyleFeatTexResult ConvertSoftOneHotToStyleFeatTex(float[] softOneHot)
    {
        const int codebookSize = 1024;
        const int numTokens = 256;

        var minEncodings = new float[codebookSize * numTokens];

        for (var i = 0; i < numTokens; i++)
        {
            var baseIdx = i * codebookSize;
            var maxVal = softOneHot[baseIdx];
            var maxIdx = 0;
            for (var j = 1; j < codebookSize; j++)
            {
                var val = softOneHot[baseIdx + j];
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = j;
                }
            }
            minEncodings[i * codebookSize + maxIdx] = 1f;
        }

        var styleFeatBuf = new ComputeBuffer(codebookSize * numTokens, sizeof(float), ComputeBufferType.Structured);
        styleFeatBuf.SetData(minEncodings);

        var styleFeatTex = _encoderRepro.RentTempArray(1, 256, 256, RenderTextureFormat.ARGBHalf);
        _ops.FillPack4FromBufferCHW(styleFeatBuf, 1, 256, codebookSize, styleFeatTex);

        var splitPacks = 128;
        var split0Tex = _encoderRepro.RentTempArray(1, 256, splitPacks, RenderTextureFormat.ARGBHalf);
        var split1Tex = _encoderRepro.RentTempArray(1, 256, splitPacks, RenderTextureFormat.ARGBHalf);
        _ops.CopyPack4(styleFeatTex, 0, split0Tex, 0, splitPacks);
        _ops.CopyPack4(styleFeatTex, splitPacks, split1Tex, 0, splitPacks);

        styleFeatBuf.Dispose();
        return new StyleFeatTexResult
        {
            StyleFeat = styleFeatTex,
            Split0 = split0Tex,
            Split1 = split1Tex,
        };
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
            throw new InvalidOperationException("CodeFormer(复刻) encoder param 不存在: " + paramPath);
        if (!File.Exists(binPath))
            throw new InvalidOperationException("CodeFormer(复刻) encoder bin 不存在: " + binPath);

        ReportProgress(0.02f, "读取编码器模型…");
        await UniTask.Yield();

        var paramText = await File.ReadAllTextAsync(paramPath);
        var bytes = await File.ReadAllBytesAsync(binPath);

        ReportProgress(0.06f, "加载编码器模型…");
        await UniTask.Yield();

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
            throw new InvalidOperationException("CodeFormer(复刻) generator param 不存在: " + paramPath);
        if (!File.Exists(binPath))
            throw new InvalidOperationException("CodeFormer(复刻) generator bin 不存在: " + binPath);

        ReportProgress(0.08f, "读取生成器模型…");
        await UniTask.Yield();

        var paramText = await File.ReadAllTextAsync(paramPath);
        var bytes = await File.ReadAllBytesAsync(binPath);

        ReportProgress(0.10f, "加载生成器模型…");
        await UniTask.Yield();

        using (var ms = new MemoryStream(bytes))
        using (var br = new NcnnBinReader(ms))
        {
            _generatorRepro.LoadModel(paramText, br);
        }

        _generatorLoaded = true;
    }

    private void ReportProgress(float progress01, string text)
    {
        progress01 = Mathf.Clamp01(progress01);
        try { ProgressChanged?.Invoke(progress01, text ?? ""); } catch { }
    }

    private static bool IsLikelyVulkanOom(Exception e)
    {
        if (e == null) return false;
        var msg = e.Message ?? "";
        if (msg.IndexOf("Out of device memory", StringComparison.OrdinalIgnoreCase) >= 0) return true;
        if (msg.IndexOf("out of memory", StringComparison.OrdinalIgnoreCase) >= 0) return true;
        if (msg.IndexOf("failed to create", StringComparison.OrdinalIgnoreCase) >= 0) return true;
        return false;
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
        var expanded = new RectInt(
            rect.x - dw,
            rect.y - dh,
            rect.width + dw * 2,
            rect.height + dh * 2
        );
        return ClampRect(expanded, imgW, imgH);
    }

    private static Texture2D CropTexture(Texture2D src, RectInt rect)
    {
        rect = ClampRect(rect, src.width, src.height);
        if (rect.width <= 0 || rect.height <= 0)
            return null;
        var dst = new Texture2D(rect.width, rect.height, TextureFormat.RGBA32, false, true);
        var srcPixels = src.GetPixels32();
        var dstPixels = new Color32[rect.width * rect.height];
        var sw = src.width;
        for (var y = 0; y < rect.height; y++)
        {
            var srcRow = (rect.y + y) * sw + rect.x;
            var dstRow = y * rect.width;
            Array.Copy(srcPixels, srcRow, dstPixels, dstRow, rect.width);
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
        if (rt == null)
            return null;
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
        compRt.enableRandomWrite = false;
        compRt.wrapMode = TextureWrapMode.Clamp;
        compRt.filterMode = FilterMode.Bilinear;
        compRt.Create();

        var prev = RenderTexture.active;
        RenderTexture.active = compRt;
        GL.Clear(true, true, Color.clear);
        RenderTexture.active = prev;

        Graphics.Blit(src, compRt);

        var material = new Material(Shader.Find("Hidden/BlendWithMask"));
        if (material == null)
        {
            Graphics.Blit(restored, compRt, new Vector2(1f * rect.width / compRt.width, 1f * rect.height / compRt.height),
                new Vector2(1f * rect.x / compRt.width, 1f * rect.y / compRt.height));
            return compRt;
        }

        material.SetTexture("_RestoredTex", restored);
        material.SetTexture("_MaskTex", mask);
        material.SetFloat("_RectX", (float)rect.x / src.width);
        material.SetFloat("_RectY", (float)rect.y / src.height);
        material.SetFloat("_RectW", (float)rect.width / src.width);
        material.SetFloat("_RectH", (float)rect.height / src.height);

        Graphics.Blit(restored, compRt, material);
        UnityEngine.Object.Destroy(material);

        return compRt;
    }
}
