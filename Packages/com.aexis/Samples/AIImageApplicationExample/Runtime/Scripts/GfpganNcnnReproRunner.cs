using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Aexis.Samples.Async;
using Aexis.Ncnn;
using UnityEngine;
using UnityEngine.Rendering;
using Aexis.Execution;

public sealed class GfpganNcnnReproRunner : MonoBehaviour
{
    public string paramRelativePath = "GFPGAN/models/encoder.param";
    public string binRelativePath = "GFPGAN/models/encoder.bin";
    public string styleRelativePath = "GFPGAN/models/style.bin";
    public AexisPrecisionMode precisionMode = AexisPrecisionMode.Auto;
    public int maxInputLongSide = 2048;
    public float faceMaskThreshold = 0.2f;
    public float faceBoxExpand = 0.35f;
    public bool enableFaceRegionDebugDump = false;
    public bool disallowBufferAccess = true;
    public bool disallowBufferOutputs = true;
    public bool disallowBufferToTextureMaterialization = true;

    public event Action<float, string> ProgressChanged;

    private sealed class StyleConvWeights : IDisposable
    {
        public int inc;
        public int hidDim;
        public int numOutput;
        public ComputeBuffer selfWeight;
        public ComputeBuffer modulationW;
        public ComputeBuffer modulationB;
        public float noiseWeight;
        public ComputeBuffer bias4;

        public void Dispose()
        {
            try { selfWeight?.Dispose(); } catch { }
            try { modulationW?.Dispose(); } catch { }
            try { modulationB?.Dispose(); } catch { }
            try { bias4?.Dispose(); } catch { }
        }
    }

    private sealed class ToRgbWeights : IDisposable
    {
        public int inc;
        public int hidDim;
        public int numOutput;
        public ComputeBuffer selfWeight;
        public ComputeBuffer modulationW;
        public ComputeBuffer modulationB;
        public ComputeBuffer bias4;

        public void Dispose()
        {
            try { selfWeight?.Dispose(); } catch { }
            try { modulationW?.Dispose(); } catch { }
            try { modulationB?.Dispose(); } catch { }
            try { bias4?.Dispose(); } catch { }
        }
    }

    private sealed class EncoderOutputs
    {
        public RenderTexture styles;
        public RenderTexture[] conditions;
    }

    private AexisGraphSession _repro;
    private StyleConvWeights[] _styleConv;
    private ToRgbWeights[] _toRgb;
    private ComputeBuffer _constInputBuf;
    private readonly Dictionary<int, ComputeBuffer> _zeroBias4 = new Dictionary<int, ComputeBuffer>();
    private int _noiseSequence;
    private AexisOps _ops;
    private bool _loaded;
    private bool _hasAppliedPrecisionMode;
    private AexisPrecisionMode _appliedPrecisionMode;

    private void Awake()
    {
        EnsureRuntimeObjects();
    }

    private void OnDestroy()
    {
        Release();
    }

    private void Release()
    {
        if (_styleConv != null)
        {
            for (var i = 0; i < _styleConv.Length; i++)
                _styleConv[i]?.Dispose();
        }
        if (_toRgb != null)
        {
            for (var i = 0; i < _toRgb.Length; i++)
                _toRgb[i]?.Dispose();
        }
        _styleConv = null;
        _toRgb = null;
        try { _constInputBuf?.Dispose(); } catch { }
        _constInputBuf = null;
        foreach (var kv in _zeroBias4)
        {
            try { kv.Value?.Dispose(); } catch { }
        }
        _zeroBias4.Clear();
        _noiseSequence = 0;
        _repro?.Release();
        _loaded = false;
        try { _repro?.Dispose(); } catch { }
        _repro = null;
        try { _ops?.Dispose(); } catch { }
        _ops = null;
        _hasAppliedPrecisionMode = false;
    }

    public void ReleaseRuntimeResources()
    {
        Release();
    }

    public async UniTask<GfpganResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (src == null)
            return default;

        EnsureRuntimeObjects();
        var totalSw = Stopwatch.StartNew();
        var originalW0 = src.width;
        var originalH0 = src.height;


        GfpganResult Finish(GfpganResult r)
        {
            r.elapsedMs = totalSw.ElapsedMilliseconds;
            try
            {
                UnityEngine.Debug.Log("[TIMING] GFPGAN(repro) " + r.elapsedMs + " ms | in=" + originalW0 + "x" + originalH0 + " | err=" + (r.error ?? ""));
            }
            catch
            {
            }
            return r;
        }

        await EnsureLoaded();
        if (_repro.Model == null)
            return Finish(new GfpganResult { error = "GFPGAN(复刻) 模型不可用" });

        var originalW = src.width;
        var originalH = src.height;
        var maxSide = Mathf.Max(originalW, originalH);
        var limit = Mathf.Max(256, maxInputLongSide);

        ReportProgress(0f, "准备输入");
        await UniTask.Yield();

        NcnnFaceRegionGenerator faceRegion = null;
        Texture2D scaled = null;
        Texture2D faceMask = null;
        Texture2D faceCrop = null;
        RenderTexture face512 = null;
        Texture restored512 = null;
        RenderTexture restoredCrop = null;
        Texture2D restoredCropTex = null;
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
                    return Finish(new GfpganResult { error = "缩放输入失败" });
                inputTex = scaled;
            }

            ReportProgress(0.06f, "生成脸部区域");
            await UniTask.Yield();
            faceRegion = NcnnFaceRegionGenerator.GetSharedInstance();
            faceRegion.disallowBufferAccess = disallowBufferAccess;
            faceRegion.disallowBufferOutputs = disallowBufferOutputs;
            faceRegion.disallowBufferToTextureMaterialization = disallowBufferToTextureMaterialization;
            RectInt rect = default;
            if (faceRegion != null && faceRegion.enabled)
            {
                var rr = await faceRegion.GenerateAsync(inputTex, enableFaceRegionDebugDump, ct);
                if (string.IsNullOrWhiteSpace(rr.error) && rr.faceRect.width > 0 && rr.faceRect.height > 0)
                {
                    faceMask = rr.mask;
                    rect = rr.faceRect;
                    if (enableFaceRegionDebugDump && !string.IsNullOrWhiteSpace(rr.dumpDir))
                        UnityEngine.Debug.Log("[GFPGAN Repro] face region dump: " + rr.dumpDir);
                }
                else
                {
                    if (rr.mask != null)
                        DestroyObjectSafe(rr.mask);
                }
            }

            if (rect.width <= 0 || rect.height <= 0)
                rect = FindFaceRect(faceMask, inputTex.width, inputTex.height, faceMaskThreshold);
            rect = ExpandRect(rect, inputTex.width, inputTex.height, faceBoxExpand);
            if (rect.width <= 8 || rect.height <= 8)
                rect = new RectInt(inputTex.width / 4, inputTex.height / 4, inputTex.width / 2, inputTex.height / 2);

            ReportProgress(0.10f, "裁剪脸部");
            await UniTask.Yield();
            faceCrop = CropTexture(inputTex, rect);
            if (faceCrop == null)
                return Finish(new GfpganResult { error = "裁剪失败" });

            face512 = ResizeTextureBilinear((Texture)faceCrop, 512, 512);
            if (face512 == null)
                return Finish(new GfpganResult { error = "resize到512失败" });

            ReportProgress(0.15f, "推理中…");
            await UniTask.Yield();
            restored512 = await RunGfpgan512Async(face512, ct);
            if (restored512 == null)
                return Finish(new GfpganResult { error = "GFPGAN(复刻) 推理失败" });

            ReportProgress(0.85f, "回贴到原图");
            await UniTask.Yield();
            restoredCrop = ResizeTextureBilinear(restored512, rect.width, rect.height);
            if (restoredCrop == null)
                return Finish(new GfpganResult { error = "回贴缩放失败" });

            restoredCropTex = await RenderTextureToTexture2DAsync(restoredCrop, rect.width, rect.height, ct);
            if (restoredCropTex == null)
                return Finish(new GfpganResult { error = "回贴回读失败" });

            var composedTex = NcnnFaceRegionPaster.ComposeRectWithSoftMask(inputTex, restoredCropTex, faceMask, rect, Mathf.Clamp01(faceMaskThreshold));
            if (composedTex == null)
                return Finish(new GfpganResult { error = "合成失败" });

            Texture2D finalTex = composedTex;
            if (Mathf.Abs(scaleDown - 1f) > 1e-6f)
            {
                ReportProgress(0.95f, "回缩放到原分辨率");
                await UniTask.Yield();
                var resized = ResizeTextureBilinear(finalTex, originalW, originalH);
                DestroyObjectSafe(finalTex);
                finalTex = resized;
                if (finalTex == null)
                    return Finish(new GfpganResult { error = "回缩放失败" });
            }

            finalTex.wrapMode = TextureWrapMode.Clamp;
            finalTex.filterMode = FilterMode.Bilinear;
            finalTex.name = "GFPGAN_Repro";
            ReportProgress(1f, "完成");
            await UniTask.Yield();
            return Finish(new GfpganResult { texture = finalTex });
        }
        catch (Exception e)
        {
            if (IsLikelyVulkanOom(e))
                return Finish(new GfpganResult { error = "Vulkan - Out of device memory" });
            return Finish(new GfpganResult { error = e.Message });
        }
        finally
        {
            if (scaled != null) DestroyObjectSafe(scaled);
            if (faceMask != null) DestroyObjectSafe(faceMask);
            if (faceCrop != null) DestroyObjectSafe(faceCrop);
            if (face512 != null) DestroyObjectSafe(face512);
            if (restored512 != null) DestroyObjectSafe(restored512);
            if (restoredCrop != null) DestroyObjectSafe(restoredCrop);
            if (restoredCropTex != null) DestroyObjectSafe(restoredCropTex);
        }
    }

    private async UniTask<Texture> RunGfpgan512Async(Texture face512, CancellationToken ct)
    {
        if (face512 == null || face512.width != 512 || face512.height != 512)
            return null;

        _noiseSequence = 0;

        RenderTexture inArr = null;
        RenderTexture styles = null;
        RenderTexture[] cond = null;
        RenderTexture constIn = null;
        RenderTexture outFeat = null;
        RenderTexture skip = null;
        RenderTexture tmp = null;
        RenderTexture skipClip = null;
        RenderTexture outTex = null;

        try
        {
            await UniTask.NextFrame();
            ct.ThrowIfCancellationRequested();
            inArr = _repro.RentTempArray(512, 512, 1, RenderTextureFormat.ARGBHalf);
            await ExecuteGfpganGpuOperationAsync(
                "input-pack-rgb",
                () => _repro.Ops.PackRgbToPack4Gfpgan(face512, 0, 0, 1, 1, inArr),
                ct);
            var encoderOutputs = await RunEncoderForGfpganAsync(inArr, ct);
            styles = encoderOutputs?.styles;
            cond = encoderOutputs?.conditions;
            if (styles == null || styles.dimension != TextureDimension.Tex2D || styles.width < 512 || styles.height < 16)
                return null;
            if (cond == null || cond.Length != 14)
                return null;

            constIn = _repro.RentTempArray(4, 4, 128, RenderTextureFormat.ARGBHalf);
            await ExecuteGfpganGpuOperationAsync(
                "const-input-upload",
                () => _repro.Ops.FillPack4FromBufferCHW(_constInputBuf, 4, 4, 512, constIn),
                ct);

            outFeat = await RunStyleConvAsync(constIn, styles, 0, _styleConv[14], 0, true, ct);

            skip = await RunToRgbAsync(outFeat, styles, 1, _toRgb[7], null, ct);

            await UniTask.Yield();

            var j = 0;
            for (var i = 1; i < 14; i += 2)
            {
                outFeat = await RunStyleConvAsync(outFeat, styles, i, _styleConv[i - 1], 1, true, ct);
                tmp = _repro.RentTempArray(outFeat.width, outFeat.height, outFeat.volumeDepth, RenderTextureFormat.ARGBHalf);
                await ExecuteGfpganGpuOperationAsync(
                    "sft-" + i,
                    () => _repro.Ops.SftPack4(outFeat, cond[i - 1], cond[i], outFeat.volumeDepth, outFeat.volumeDepth / 2, tmp),
                    ct);
                _repro.ReturnTempArray(outFeat);
                outFeat = tmp;
                tmp = null;

                outFeat = await RunStyleConvAsync(outFeat, styles, i + 1, _styleConv[i], 0, true, ct);

                skip = await RunToRgbAsync(outFeat, styles, i + 2, _toRgb[j], skip, ct);
                j++;

                ReportProgress(0.15f + (float)i / 14.0f * 0.8f, "推理分块 " + i + "/" + 14);
                await UniTask.Yield();
            }

            skipClip = _repro.RentTempArray(512, 512, 1, RenderTextureFormat.ARGBHalf);
            await ExecuteGfpganGpuOperationAsync(
                "final-clip",
                () => _repro.Ops.ClipPack4(skip, -1f, 1f, 1, skipClip),
                ct);

            outTex = new RenderTexture(512, 512, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            outTex.enableRandomWrite = true;
            outTex.wrapMode = TextureWrapMode.Clamp;
            outTex.filterMode = FilterMode.Bilinear;
            outTex.Create();
            await ExecuteGfpganGpuOperationAsync(
                "final-pack-rgb",
                () => _repro.Ops.Pack4ToRgb01(skipClip, outTex),
                ct);


            //outTex = RenderTextureToTexture2D(rgb, 512, 512);
            //if (outTex == null)
            //    return null;
            //outTex.wrapMode = TextureWrapMode.Clamp;
            //outTex.filterMode = FilterMode.Bilinear;
            return outTex;
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[GFPGAN Repro] RunGfpgan512 failed: " + e.Message);
            if (outTex != null) DestroyObjectSafe(outTex);
            return null;
        }
        finally
        {
            if (inArr != null) _repro.ReturnTempArray(inArr);
            if (styles != null) _repro.ReturnTempArray(styles);
            if (cond != null)
            {
                for (var i = 0; i < cond.Length; i++)
                    if (cond[i] != null) _repro.ReturnTempArray(cond[i]);
            }
            if (constIn != null) _repro.ReturnTempArray(constIn);
            if (outFeat != null) _repro.ReturnTempArray(outFeat);
            if (skip != null) _repro.ReturnTempArray(skip);
            if (tmp != null) _repro.ReturnTempArray(tmp);
            if (skipClip != null) _repro.ReturnTempArray(skipClip);
        }
    }

    private async UniTask<EncoderOutputs> RunEncoderForGfpganAsync(RenderTexture inputPack4, CancellationToken ct)
    {
        var condNames = new[]
        {
            "440","443","463","466","486","489","509","512","532","535","555","558","578","581"
        };
        var pinned = new HashSet<string>(condNames, StringComparer.Ordinal) { "420" };

        var inputName = _repro.Model.layers.Count > 0 && _repro.Model.layers[0].topNames != null && _repro.Model.layers[0].topNames.Length > 0
            ? _repro.Model.layers[0].topNames[0]
            : "input.1";

        var outputs = new EncoderOutputs();

        try
        {
            _repro.DebugLogAllLayerHeartbeats = UnityEngine.Debug.isDebugBuild;
            _repro.DebugLog = UnityEngine.Debug.isDebugBuild
                ? line =>
                {
                    if (line.StartsWith("[LayerHeartbeat]", StringComparison.Ordinal))
                        UnityEngine.Debug.Log("[GFPGAN Encoder] " + line);
                }
                : null;
            if (UnityEngine.Debug.isDebugBuild)
                UnityEngine.Debug.Log("[GFPGAN Encoder] begin | input=" + inputPack4.GetInstanceID() + " | layers=" + _repro.Model.layers.Count);
            var inputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
            {
                { inputName, inputPack4 }
            };
            using (var result = await _repro.InferWithMultiInputsAsync(
                       inputs,
                       null,
                       pinned,
                       cancellationToken: ct,
                       yieldEveryLayers: 1))
            {
                outputs.styles = ExtractExistingGpuTexture(result, "420");

                outputs.conditions = new RenderTexture[condNames.Length];
                for (var i = 0; i < condNames.Length; i++)
                {
                    outputs.conditions[i] = ExtractExistingGpuTexture(result, condNames[i]);
                }
            }

            if (UnityEngine.Debug.isDebugBuild)
                UnityEngine.Debug.Log("[GFPGAN Encoder] complete | styles=" + outputs.styles.GetInstanceID() + " | conditions=" + outputs.conditions.Length);
            return outputs;
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[GFPGAN Repro] encoder failed: " + e.Message);
            if (outputs.conditions != null)
            {
                for (var i = 0; i < outputs.conditions.Length; i++)
                    if (outputs.conditions[i] != null) _repro.ReturnTempArray(outputs.conditions[i]);
                outputs.conditions = null;
            }
            if (outputs.styles != null)
                _repro.ReturnTempArray(outputs.styles);
            return null;
        }
    }

    private static RenderTexture ExtractExistingGpuTexture(AexisGraphSession.InferResult result, string blobName)
    {
        if (result == null || !result.TryGetExistingTexture(blobName, out var texture) || texture == null)
        {
            throw new InvalidOperationException(
                "GFPGAN Pack4-only generator rejected a non-texture encoder activation"
                + " | blob=" + blobName
                + " | rejected_fallback=buffer-or-readback");
        }
        return result.ExtractTexture(blobName);
    }

    private static async UniTask ExecuteGfpganGpuOperationAsync(string stage, Action execute, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (UnityEngine.Debug.isDebugBuild)
            UnityEngine.Debug.Log("[GFPGAN Generator] gpu-begin | stage=" + stage);
        execute();
        await UniTask.NextFrame();
        ct.ThrowIfCancellationRequested();
        if (UnityEngine.Debug.isDebugBuild)
            UnityEngine.Debug.Log("[GFPGAN Generator] gpu-frame-complete | stage=" + stage);
    }

    private async UniTask<RenderTexture> RunStyleConvAsync(
        RenderTexture x,
        RenderTexture styles,
        int styleRow,
        StyleConvWeights w,
        int sampleMode,
        bool demodulate,
        CancellationToken ct)
    {
        var inArr = x;
        if (sampleMode == 1)
        {
            var up = _repro.RentTempArray(inArr.width * 2, inArr.height * 2, inArr.volumeDepth, RenderTextureFormat.ARGBHalf);
            await ExecuteGfpganGpuOperationAsync(
                "style-" + styleRow + "-upsample",
                () => _repro.Ops.Interp2xPack4(inArr, inArr.volumeDepth, up),
                ct);
            if (inArr != x)
                _repro.ReturnTempArray(inArr);
            inArr = up;
        }

        var outPacks = (w.numOutput + 3) / 4;
        var outArr = _repro.RentTempArray(inArr.width, inArr.height, outPacks, RenderTextureFormat.ARGBHalf);
        var w4Count = outPacks * (w.hidDim / 4) * 9 * 4;
        var dynamicWeightWidth = Mathf.Min(2048, w4Count);
        var dynamicWeightHeight = Mathf.CeilToInt(w4Count / (float)dynamicWeightWidth);
        RenderTexture styleVector = null;
        RenderTexture demod = null;
        RenderTexture dynamicW4 = null;
        try
        {
            styleVector = _repro.RentTempMat(w.hidDim, 1, RenderTextureFormat.RFloat);
            await ExecuteGfpganGpuOperationAsync(
                "style-" + styleRow + "-modulation",
                () => _repro.Ops.GfpganStyleModulation(styles, styleRow, 512, w.modulationW, w.modulationB, w.hidDim, styleVector),
                ct);
            if (demodulate)
            {
                demod = _repro.RentTempMat(w.numOutput, 1, RenderTextureFormat.RFloat);
                await ExecuteGfpganGpuOperationAsync(
                    "style-" + styleRow + "-demodulation",
                    () => _repro.Ops.GfpganStyleDemod(styleVector, w.selfWeight, w.hidDim, w.numOutput, 9, demod),
                    ct);
            }
            dynamicW4 = _repro.RentTempArray(dynamicWeightWidth, dynamicWeightHeight, 1, RenderTextureFormat.ARGBFloat);
            await ExecuteGfpganGpuOperationAsync(
                "style-" + styleRow + "-dynamic-weight",
                () => _repro.Ops.GfpganBuildDynamicWeight(styleVector, demod, w.selfWeight, w.hidDim, w.numOutput, 9, demodulate, dynamicW4),
                ct);
            await ExecuteGfpganGpuOperationAsync(
                "style-" + styleRow + "-conv3x3",
                () => _repro.Ops.Conv3x3Pack4DynamicWeight(inArr, w.hidDim / 4, dynamicW4, GetZeroBias4(outPacks), outPacks, 1, 0, 0f, outArr),
                ct);
        }
        finally
        {
            if (dynamicW4 != null) _repro.ReturnTempArray(dynamicW4);
            if (demod != null) _repro.ReturnTempArray(demod);
            if (styleVector != null) _repro.ReturnTempArray(styleVector);
        }

        var scaled = _repro.RentTempArray(outArr.width, outArr.height, outArr.volumeDepth, RenderTextureFormat.ARGBHalf);
        await ExecuteGfpganGpuOperationAsync(
            "style-" + styleRow + "-scale",
            () => _repro.Ops.ScalePack4(outArr, 1.4142135381698608f, outArr.volumeDepth, scaled),
            ct);
        _repro.ReturnTempArray(outArr);
        outArr = scaled;

        await ExecuteGfpganGpuOperationAsync(
            "style-" + styleRow + "-noise",
            () => _repro.Ops.GfpganAddNoisePack4(outArr, w.noiseWeight, NextNoiseSeed(), outArr.volumeDepth),
            ct);

        var biased = _repro.RentTempArray(outArr.width, outArr.height, outArr.volumeDepth, RenderTextureFormat.ARGBHalf);
        await ExecuteGfpganGpuOperationAsync(
            "style-" + styleRow + "-bias",
            () => _repro.Ops.AddBiasPack4(outArr, w.bias4, outArr.volumeDepth, biased),
            ct);
        _repro.ReturnTempArray(outArr);
        outArr = biased;

        var act = _repro.RentTempArray(outArr.width, outArr.height, outArr.volumeDepth, RenderTextureFormat.ARGBHalf);
        await ExecuteGfpganGpuOperationAsync(
            "style-" + styleRow + "-leaky-relu",
            () => _repro.Ops.LeakyReluPack4(outArr, 0.2f, outArr.volumeDepth, act),
            ct);
        _repro.ReturnTempArray(outArr);
        outArr = act;

        if (inArr != x)
            _repro.ReturnTempArray(inArr);
        return outArr;
    }

    private async UniTask<RenderTexture> RunToRgbAsync(
        RenderTexture feat,
        RenderTexture styles,
        int styleRow,
        ToRgbWeights w,
        RenderTexture skip,
        CancellationToken ct)
    {
        var outArr = _repro.RentTempArray(feat.width, feat.height, 1, RenderTextureFormat.ARGBHalf);
        var w4Count = 1 * (w.hidDim / 4) * 4;
        var dynamicWeightWidth = Mathf.Min(2048, w4Count);
        var dynamicWeightHeight = Mathf.CeilToInt(w4Count / (float)dynamicWeightWidth);
        RenderTexture styleVector = null;
        RenderTexture dynamicW4 = null;
        try
        {
            styleVector = _repro.RentTempMat(w.hidDim, 1, RenderTextureFormat.RFloat);
            await ExecuteGfpganGpuOperationAsync(
                "rgb-" + styleRow + "-modulation",
                () => _repro.Ops.GfpganStyleModulation(styles, styleRow, 512, w.modulationW, w.modulationB, w.hidDim, styleVector),
                ct);
            dynamicW4 = _repro.RentTempArray(dynamicWeightWidth, dynamicWeightHeight, 1, RenderTextureFormat.ARGBFloat);
            await ExecuteGfpganGpuOperationAsync(
                "rgb-" + styleRow + "-dynamic-weight",
                () => _repro.Ops.GfpganBuildDynamicWeight(styleVector, null, w.selfWeight, w.hidDim, w.numOutput, 1, false, dynamicW4),
                ct);
            await ExecuteGfpganGpuOperationAsync(
                "rgb-" + styleRow + "-conv1x1",
                () => _repro.Ops.Conv1x1Pack4DynamicWeight(feat, w.hidDim / 4, dynamicW4, GetZeroBias4(1), 1, 0, 0f, outArr),
                ct);
        }
        finally
        {
            if (dynamicW4 != null) _repro.ReturnTempArray(dynamicW4);
            if (styleVector != null) _repro.ReturnTempArray(styleVector);
        }

        var biased = _repro.RentTempArray(outArr.width, outArr.height, 1, RenderTextureFormat.ARGBHalf);
        await ExecuteGfpganGpuOperationAsync(
            "rgb-" + styleRow + "-bias",
            () => _repro.Ops.AddBiasPack4(outArr, w.bias4, 1, biased),
            ct);
        _repro.ReturnTempArray(outArr);
        outArr = biased;

        if (skip == null)
            return outArr;

        var up = _repro.RentTempArray(skip.width * 2, skip.height * 2, 1, RenderTextureFormat.ARGBHalf);
        await ExecuteGfpganGpuOperationAsync(
            "rgb-" + styleRow + "-skip-upsample",
            () => _repro.Ops.Interp2xPack4(skip, 1, up),
            ct);
        _repro.ReturnTempArray(skip);

        var sum = _repro.RentTempArray(up.width, up.height, 1, RenderTextureFormat.ARGBHalf);
        await ExecuteGfpganGpuOperationAsync(
            "rgb-" + styleRow + "-skip-add",
            () => _repro.Ops.AddPack4(outArr, up, 1f, 1f, 1, sum),
            ct);
        _repro.ReturnTempArray(outArr);
        _repro.ReturnTempArray(up);
        return sum;
    }

    private async UniTask LoadStyleBin(string stylePath)
    {
        if (_styleConv != null || _toRgb != null)
            return;

        _styleConv = new StyleConvWeights[15];
        _toRgb = new ToRgbWeights[8];

        byte[] bytes;
        if (Application.isBatchMode)
            bytes = File.ReadAllBytes(stylePath);
        else
            bytes = await File.ReadAllBytesAsync(stylePath);
        MemoryStream ms = new MemoryStream(bytes);
        using (var br = new BinaryReader(ms))
        {
            var styleHidDim = new[] { 512,512,512,512,512,512,512,512,512,256,256,128,128,64,512 };
            var styleOutC = new[] { 512,512,512,512,512,512,512,512,256,256,128,128,64,64,512 };
            for (var i = 0; i < 15; i++)
            {
                var w = new StyleConvWeights();
                w.inc = 512;
                w.hidDim = styleHidDim[i];
                w.numOutput = styleOutC[i];

                var selfCount = w.numOutput * w.hidDim * 3 * 3;
                w.selfWeight = CreateFloatBuffer(ReadFloatArray(br, selfCount));
                w.modulationW = CreateFloatBuffer(ReadFloatArray(br, w.inc * w.hidDim));
                w.modulationB = CreateFloatBuffer(ReadFloatArray(br, w.hidDim));
                w.noiseWeight = br.ReadSingle();
                w.bias4 = CreateBias4Buffer(ReadFloatArray(br, w.numOutput), w.numOutput);
                _styleConv[i] = w;
            }

            var rgbHidDim = new[] { 512,512,512,512,256,128,64,512 };
            for (var i = 0; i < 8; i++)
            {
                var w = new ToRgbWeights();
                w.inc = 512;
                w.hidDim = rgbHidDim[i];
                w.numOutput = 3;
                w.selfWeight = CreateFloatBuffer(ReadFloatArray(br, w.numOutput * w.hidDim));
                w.modulationW = CreateFloatBuffer(ReadFloatArray(br, w.inc * w.hidDim));
                w.modulationB = CreateFloatBuffer(ReadFloatArray(br, w.hidDim));
                w.bias4 = CreateBias4Buffer(ReadFloatArray(br, 3), 3);
                _toRgb[i] = w;
            }

            var constInput = ReadFloatArray(br, 4 * 4 * 512);
            _constInputBuf = CreateFloatBuffer(constInput);
        }

        ms.Dispose();
    }

    private static float[] ReadFloatArray(BinaryReader br, int count)
    {
        var a = new float[count];
        for (var i = 0; i < count; i++)
            a[i] = br.ReadSingle();
        return a;
    }

    private static ComputeBuffer CreateFloatBuffer(float[] values)
    {
        if (values == null || values.Length == 0)
            throw new ArgumentException("GFPGAN immutable weight payload is empty.", nameof(values));
        var buffer = new ComputeBuffer(values.Length, sizeof(float), ComputeBufferType.Structured);
        buffer.SetData(values);
        return buffer;
    }

    private ComputeBuffer CreateBias4Buffer(float[] bias, int outC)
    {
        var outPacks = (outC + 3) / 4;
        var v4 = new Vector4[outPacks];
        for (var p = 0; p < outPacks; p++)
        {
            var c0 = p * 4 + 0;
            var c1 = p * 4 + 1;
            var c2 = p * 4 + 2;
            var c3 = p * 4 + 3;
            v4[p] = new Vector4(
                c0 < outC ? bias[c0] : 0f,
                c1 < outC ? bias[c1] : 0f,
                c2 < outC ? bias[c2] : 0f,
                c3 < outC ? bias[c3] : 0f);
        }
        var cb = new ComputeBuffer(outPacks, sizeof(float) * 4, ComputeBufferType.Structured);
        cb.SetData(v4);
        return cb;
    }

    private ComputeBuffer GetZeroBias4(int outPacks)
    {
        if (_zeroBias4.TryGetValue(outPacks, out var cb) && cb != null)
            return cb;
        var v = new Vector4[outPacks];
        cb = new ComputeBuffer(outPacks, sizeof(float) * 4, ComputeBufferType.Structured);
        cb.SetData(v);
        _zeroBias4[outPacks] = cb;
        return cb;
    }

    private int NextNoiseSeed()
    {
        return unchecked(0x4F1BBCDC + _noiseSequence++ * unchecked((int)0x9E3779B9));
    }

    private static RectInt FindFaceRect(Texture2D mask, int w, int h, float threshold01)
    {
        if (mask == null || mask.width <= 0 || mask.height <= 0)
            return new RectInt(w / 4, h / 4, w / 2, h / 2);
        var pixels = mask.GetPixels();
        var minX = w;
        var minY = h;
        var maxX = -1;
        var maxY = -1;
        for (var y = 0; y < mask.height; y++)
        {
            for (var x = 0; x < mask.width; x++)
            {
                var v = pixels[y * mask.width + x].r;
                if (v < threshold01)
                    continue;
                minX = Math.Min(minX, x);
                minY = Math.Min(minY, y);
                maxX = Math.Max(maxX, x);
                maxY = Math.Max(maxY, y);
            }
        }
        if (maxX < 0 || maxY < 0)
            return new RectInt(w / 4, h / 4, w / 2, h / 2);
        return new RectInt(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    private static RectInt ExpandRect(RectInt r, int w, int h, float expand01)
    {
        if (r.width <= 0 || r.height <= 0)
            return r;
        var cx = r.x + r.width * 0.5f;
        var cy = r.y + r.height * 0.5f;
        var ex = r.width * (1f + Mathf.Max(0f, expand01));
        var ey = r.height * (1f + Mathf.Max(0f, expand01));
        var x0 = Mathf.Clamp(Mathf.FloorToInt(cx - ex * 0.5f), 0, Mathf.Max(0, w - 1));
        var y0 = Mathf.Clamp(Mathf.FloorToInt(cy - ey * 0.5f), 0, Mathf.Max(0, h - 1));
        var x1 = Mathf.Clamp(Mathf.CeilToInt(cx + ex * 0.5f), 0, w);
        var y1 = Mathf.Clamp(Mathf.CeilToInt(cy + ey * 0.5f), 0, h);
        return new RectInt(x0, y0, Mathf.Max(1, x1 - x0), Mathf.Max(1, y1 - y0));
    }


    private static async UniTask<Texture2D> RenderTextureToTexture2DAsync(
        RenderTexture rt,
        int w,
        int h,
        CancellationToken ct)
    {
        if (rt == null || w <= 0 || h <= 0)
            return null;

        await UniTask.NextFrame();
        ct.ThrowIfCancellationRequested();
        if (UnityEngine.Debug.isDebugBuild)
            UnityEngine.Debug.Log("[GFPGAN OutputReadback] begin | texture=" + rt.GetInstanceID() + " | size=" + w + "x" + h);
        var prev = RenderTexture.active;
        try
        {
            RenderTexture.active = rt;
            var tex = new Texture2D(w, h, TextureFormat.RGBA32, false, false);
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
            tex.Apply(false, false);
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            if (UnityEngine.Debug.isDebugBuild)
                UnityEngine.Debug.Log("[GFPGAN OutputReadback] complete | texture=" + rt.GetInstanceID());
            return tex;
        }
        finally
        {
            RenderTexture.active = prev;
        }
    }

    private async UniTask EnsureLoaded()
    {
        if (_loaded)
            return;

        var paramPath = Aexis.Samples.AexisSampleStreamingAssets.ResolveFilePath(paramRelativePath);
        var binPath = Aexis.Samples.AexisSampleStreamingAssets.ResolveFilePath(binRelativePath);
        var stylePath = Aexis.Samples.AexisSampleStreamingAssets.ResolveFilePath(styleRelativePath);
        if (!File.Exists(paramPath))
            throw new InvalidOperationException("GFPGAN(复刻) param 不存在: " + paramPath);
        if (!File.Exists(binPath))
            throw new InvalidOperationException("GFPGAN(复刻) bin 不存在: " + binPath);
        if (!File.Exists(stylePath))
            throw new InvalidOperationException("GFPGAN(复刻) style 不存在: " + stylePath);

        ReportProgress(0.02f, "Reading Model File...");
        await UniTask.Yield();

        string paramText;
        byte[] bytes;
        if (Application.isBatchMode)
        {
            paramText = File.ReadAllText(paramPath);
            bytes = File.ReadAllBytes(binPath);
        }
        else
        {
            paramText = await File.ReadAllTextAsync(paramPath);
            bytes = await File.ReadAllBytesAsync(binPath);
        }

        ReportProgress(0.06f, "Loading Model...");
        await UniTask.Yield();

        MemoryStream ms = new MemoryStream(bytes);
        using (var br = new NcnnBinReader(ms))
        {
            _repro.LoadModel(paramText, br);
        }
        ms.Dispose();

        ReportProgress(0.09f, "Loading Model...");
        await UniTask.Yield();

        await LoadStyleBin(stylePath);

        ReportProgress(0.1f, "Loading Model...");
        await UniTask.Yield();

        _loaded = true;
    }

    private void ReportProgress(float progress01, string text)
    {
        progress01 = Mathf.Clamp01(progress01);
        try { ProgressChanged?.Invoke(progress01, text ?? ""); } catch { }
    }

    private void EnsureRuntimeObjects()
    {
        var effectivePrecisionMode = AexisModelManifestLoader.ResolveRunnerAppliedPrecision("gfpgan", precisionMode);
        if (_repro != null && _hasAppliedPrecisionMode && _appliedPrecisionMode != effectivePrecisionMode)
        {
            UnityEngine.Debug.Log("[NcnnPrecision] GFPGAN recreating session | from=" + _appliedPrecisionMode + " | to=" + effectivePrecisionMode);
            Release();
        }
        _ops ??= new AexisOps();
        if (_repro == null)
        {
            _repro = NcnnInferenceSessionFactory.Create(_ops, "gfpgan", precisionMode);
            _appliedPrecisionMode = _repro.AppliedPrecisionMode;
            _hasAppliedPrecisionMode = true;
        }
        // The encoder contains standard 3x3/1x1 convolutions. Enable the existing
        // Pack4 kernels before model loading so it cannot select the legacy buffer path.
        _repro.EnableGeneralTextureConvolution = true;
        _repro.EnableConv1x1TextureConvolution = true;
        _repro.EnableDepthWiseTextureConvolution = true;
        _repro.DebugLog ??= line =>
        {
            if (!string.IsNullOrEmpty(line)
                && line.StartsWith("[InferencePathAudit]", StringComparison.Ordinal))
            {
                UnityEngine.Debug.Log("[GFPGAN Repro] " + line);
            }
        };
        _repro.DisallowBufferAccess = disallowBufferAccess;
        _repro.DisallowBufferOutputs = disallowBufferOutputs;
        _repro.DisallowBufferToTextureMaterialization = disallowBufferToTextureMaterialization;
        _repro.DisallowInferenceTempComputeBuffers = disallowBufferAccess
            || disallowBufferOutputs
            || disallowBufferToTextureMaterialization;
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
        RenderTexture ret = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32);
        ret.Create();
        Graphics.Blit(src, ret);
        return ret;
    }

    private static Texture2D ResizeTextureBilinear(Texture2D src, int w, int h)
    {
        if (src == null)
            return null;
        if (w <= 0 || h <= 0)
            return null;

        var dst = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
        dst.wrapMode = TextureWrapMode.Clamp;
        dst.filterMode = FilterMode.Bilinear;

        var srcPixels = src.GetPixels32();
        var dstPixels = new Color32[w * h];
        var sw = src.width;
        var sh = src.height;
        var invW = sw > 1 ? 1f / (sw - 1f) : 0f;
        var invH = sh > 1 ? 1f / (sh - 1f) : 0f;

        for (var y = 0; y < h; y++)
        {
            var v = h > 1 ? y / (h - 1f) : 0f;
            var sy = v / Mathf.Max(1e-6f, invH);
            var y0 = Mathf.Clamp((int)sy, 0, sh - 1);
            var y1 = Mathf.Clamp(y0 + 1, 0, sh - 1);
            var ty = sy - y0;
            for (var x = 0; x < w; x++)
            {
                var u = w > 1 ? x / (w - 1f) : 0f;
                var sx = u / Mathf.Max(1e-6f, invW);
                var x0 = Mathf.Clamp((int)sx, 0, sw - 1);
                var x1 = Mathf.Clamp(x0 + 1, 0, sw - 1);
                var tx = sx - x0;

                var c00 = srcPixels[y0 * sw + x0];
                var c10 = srcPixels[y0 * sw + x1];
                var c01 = srcPixels[y1 * sw + x0];
                var c11 = srcPixels[y1 * sw + x1];

                var r0 = Mathf.Lerp(c00.r, c10.r, tx);
                var g0 = Mathf.Lerp(c00.g, c10.g, tx);
                var b0 = Mathf.Lerp(c00.b, c10.b, tx);
                var a0 = Mathf.Lerp(c00.a, c10.a, tx);

                var r1 = Mathf.Lerp(c01.r, c11.r, tx);
                var g1 = Mathf.Lerp(c01.g, c11.g, tx);
                var b1 = Mathf.Lerp(c01.b, c11.b, tx);
                var a1 = Mathf.Lerp(c01.a, c11.a, tx);

                var r2 = Mathf.Lerp(r0, r1, ty);
                var g2 = Mathf.Lerp(g0, g1, ty);
                var b2 = Mathf.Lerp(b0, b1, ty);
                var a2 = Mathf.Lerp(a0, a1, ty);

                dstPixels[y * w + x] = new Color32((byte)r2, (byte)g2, (byte)b2, (byte)a2);
            }
        }

        dst.SetPixels32(dstPixels);
        dst.Apply(false, false);
        return dst;
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
}
