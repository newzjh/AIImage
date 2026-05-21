using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;

public sealed class GpuSharpenRunner : MonoBehaviour
{
    public bool enableNoiseSuppression = true;
    public bool enableFaceMidFreqReconstruction = true;
    public float frequencyDetailStrength = 1f;
    public float faceDetailStrength = 1f;

    private ComputeShader _cs;
    private readonly Dictionary<string, int> _kernelIds = new Dictionary<string, int>(StringComparer.Ordinal);

    public ComputeShader GetComputeShader()
    {
        if (_cs == null)
            _cs = Resources.Load<ComputeShader>("GPUSharpen");
        return _cs;
    }

    public int GetKernel(string kernelName)
    {
        if (string.IsNullOrWhiteSpace(kernelName))
            return -1;
        if (_kernelIds.TryGetValue(kernelName, out var id))
            return id;
        var cs = GetComputeShader();
        if (cs == null)
            return -1;
        try
        {
            id = cs.FindKernel(kernelName);
        }
        catch
        {
            id = -1;
        }
        _kernelIds[kernelName] = id;
        return id;
    }

    public async UniTask<GpuSharpenResult> ProcessAsync(Texture2D src, Texture2D faceMaskTex, bool dumpStages, CancellationToken ct)
    {
        if (src == null)
            return default;

        var cs = GetComputeShader();
        if (cs == null)
            return new GpuSharpenResult { error = "GPUSharpen.compute not found" };

        var k1 = GetKernel("StructureAnalysis");
        var k2 = GetKernel("EdgeAwareDecompose");
        var kDown = GetKernel("GaussianDown2");
        var kUp = GetKernel("GaussianUp2");
        var kDiff = GetKernel("LaplacianDiff");
        var kEnh = GetKernel("EnhanceLaplacian");
        var kDirLap = GetKernel("DirectionalSharpenLap");
        var kAdd = GetKernel("AddScalar");
        var kResp = GetKernel("DirectionalResponse");
        var kApply = GetKernel("ApplyDirectionalSharpen");
        var kClamp = GetKernel("EdgeAwareClamp");
        var kNoise = GetKernel("NoiseSuppression");
        var kBoxH = GetKernel("BoxFilterH");
        var kBoxV = GetKernel("BoxFilterV");
        var kSquare = GetKernel("SquareScalar");
        var kGuidedAB = GetKernel("GuidedAB");
        var kApplyGuided = GetKernel("ApplyGuided");
        var kMid = GetKernel("MidFreq");
        var kHigh = GetKernel("HighFreq");
        var kBilateralNoise = GetKernel("BilateralNoise");
        var kFaceStruct = GetKernel("FaceStruct");
        var kFaceBlend = GetKernel("FaceBlend");
        var k5 = GetKernel("SkinNoiseProtect");
        if (k1 < 0 || k2 < 0 || kDown < 0 || kUp < 0 || kDiff < 0 || kEnh < 0 || kDirLap < 0 || kAdd < 0 ||
            kResp < 0 || kApply < 0 || kClamp < 0 || kNoise < 0 ||
            kBoxH < 0 || kBoxV < 0 || kSquare < 0 || kGuidedAB < 0 || kApplyGuided < 0 ||
            kMid < 0 || kHigh < 0 || kBilateralNoise < 0 || kFaceStruct < 0 || kFaceBlend < 0 ||
            k5 < 0)
            return new GpuSharpenResult { error = "GPUSharpen kernels not found" };

        RenderTexture analysis = null;
        RenderTexture mask = null;
        RenderTexture baseRt = null;
        RenderTexture detail = null;
        RenderTexture enhancedDetailFull = null;
        RenderTexture accumPing = null;
        RenderTexture fullAdd = null;
        var gaussianLevels = new List<RenderTexture>();
        var laplacians = new List<RenderTexture>();
        var enhancedLaps = new List<RenderTexture>();
        var upTemps = new List<RenderTexture>();
        RenderTexture directionalResponse = null;
        RenderTexture sharpenY = null;
        RenderTexture clampedY = null;
        RenderTexture noiseSuppressedY = null;
        RenderTexture boxTmp = null;
        RenderTexture i2 = null;
        RenderTexture meanI = null;
        RenderTexture meanII = null;
        RenderTexture aRt = null;
        RenderTexture bRt = null;
        RenderTexture meanA = null;
        RenderTexture meanB = null;
        RenderTexture guidedBase = null;
        RenderTexture blurMid = null;
        RenderTexture midFreq = null;
        RenderTexture blurHigh = null;
        RenderTexture highFreq = null;
        RenderTexture noiseFiltered = null;
        RenderTexture faceStruct = null;
        RenderTexture faceEnhancedY = null;
        RenderTexture zeroMask = null;
        RenderTexture pseudoGanY = null;
        RenderTexture pseudoTmp0 = null;
        RenderTexture pseudoTmp1 = null;
        RenderTexture pseudoTmp2 = null;
        RenderTexture pseudoTmp3 = null;
        RenderTexture pseudoTmp4 = null;
        RenderTexture pseudoTmp5 = null;
        RenderTexture finalRt = null;
        string dumpDir = null;

        try
        {
            ct.ThrowIfCancellationRequested();

            var w = src.width;
            var h = src.height;
            var gx = Mathf.CeilToInt(w / 8f);
            var gy = Mathf.CeilToInt(h / 8f);

            analysis = NewRT(w, h, RenderTextureFormat.ARGBHalf);
            mask = NewRT(w, h, RenderTextureFormat.RHalf);
            baseRt = NewRT(w, h, RenderTextureFormat.ARGBHalf);
            detail = NewRT(w, h, RenderTextureFormat.RHalf);
            sharpenY = NewRT(w, h, RenderTextureFormat.RHalf);
            finalRt = NewRT(w, h, RenderTextureFormat.ARGB32);

            cs.SetTexture(k1, "_Source", src);
            cs.SetTexture(k1, "_AnalysisOut", analysis);
            cs.SetTexture(k1, "_MaskOut", mask);
            cs.Dispatch(k1, gx, gy, 1);

            if (dumpStages)
            {
                dumpDir ??= CreateDumpDir();
                await DumpStageAsync(dumpDir, w, h, "01_edge.png", "DebugVisEdge", analysis, 1f, false, ct);
                await DumpStageAsync(dumpDir, w, h, "01_dir.png", "DebugVisDir", analysis, 1f, false, ct);
                await DumpStageAsync(dumpDir, w, h, "01_tex.png", "DebugVisTex", analysis, 1f, false, ct);
                await DumpStageAsync(dumpDir, w, h, "01_flat.png", "DebugVisScalar", mask, 1f, true, ct);
            }

            cs.SetInt("_Radius", 6);
            cs.SetFloat("_SigmaSpatial", 3.2f);
            cs.SetFloat("_SigmaRange", 0.055f);
            cs.SetTexture(k2, "_Source", src);
            cs.SetTexture(k2, "_AnalysisIn", analysis);
            cs.SetTexture(k2, "_BaseOut", baseRt);
            cs.SetTexture(k2, "_DetailOut", detail);
            cs.Dispatch(k2, gx, gy, 1);

            if (dumpStages)
            {
                dumpDir ??= CreateDumpDir();
                await DumpStageAsync(dumpDir, w, h, "02_base.png", "DebugVisCopyRgb", baseRt, 1f, false, ct);
                await DumpStageAsync(dumpDir, w, h, "02_detail.png", "DebugVisSignedScalar", detail, 4.0f, true, ct);
            }

            var maxDim = Mathf.Max(w, h);
            float[] weights;
            if (maxDim >= 4096)
                weights = new[] { 0f, 0f, 0.6f, 2.4f, 2.8f, 1.0f, 0.3f };
            else if (maxDim >= 2048)
                weights = new[] { 0f, 0.3f, 1.0f, 2.6f, 2.2f, 0.8f };
            else if (maxDim >= 1024)
                weights = new[] { 0.10f, 0.80f, 1.80f, 2.40f, 1.20f };
            else
                weights = new[] { 0.25f, 1.20f, 2.20f, 1.00f };

            var freqStrength = Mathf.Clamp(frequencyDetailStrength, 0f, 4f);
            for (int i = 0; i < weights.Length; i++)
                weights[i] *= freqStrength;

            var lapCount = weights.Length;
            gaussianLevels.Clear();
            laplacians.Clear();
            enhancedLaps.Clear();
            upTemps.Clear();

            gaussianLevels.Add(detail);
            for (int gi = 1; gi < lapCount + 1; gi++)
            {
                var pw = gaussianLevels[gi - 1].width;
                var ph = gaussianLevels[gi - 1].height;
                var nw = Mathf.Max(1, (pw + 1) / 2);
                var nh = Mathf.Max(1, (ph + 1) / 2);
                var g = NewRT(nw, nh, RenderTextureFormat.RHalf);
                gaussianLevels.Add(g);
            }

            for (int gi = 0; gi < lapCount; gi++)
            {
                var outG = gaussianLevels[gi + 1];
                var gx2 = Mathf.CeilToInt(outG.width / 8f);
                var gy2 = Mathf.CeilToInt(outG.height / 8f);
                cs.SetTexture(kDown, "_ScalarIn", gaussianLevels[gi]);
                cs.SetTexture(kDown, "_ScalarOut", outG);
                cs.Dispatch(kDown, Mathf.Max(1, gx2), Mathf.Max(1, gy2), 1);
            }

            for (int li = 0; li < lapCount; li++)
            {
                var curG = gaussianLevels[li];
                var nextG = gaussianLevels[li + 1];
                var up = NewRT(curG.width, curG.height, RenderTextureFormat.RHalf);
                var lap = NewRT(curG.width, curG.height, RenderTextureFormat.RHalf);
                var enh = NewRT(curG.width, curG.height, RenderTextureFormat.RHalf);
                var enhDir = NewRT(curG.width, curG.height, RenderTextureFormat.RHalf);
                upTemps.Add(up);
                laplacians.Add(lap);
                enhancedLaps.Add(enhDir);

                cs.SetTexture(kUp, "_LowIn", nextG);
                cs.SetTexture(kUp, "_UpOut", up);
                cs.Dispatch(kUp, Mathf.Max(1, Mathf.CeilToInt(curG.width / 8f)), Mathf.Max(1, Mathf.CeilToInt(curG.height / 8f)), 1);

                cs.SetTexture(kDiff, "_HighIn", curG);
                cs.SetTexture(kDiff, "_LowUpIn", up);
                cs.SetTexture(kDiff, "_LapOut", lap);
                cs.Dispatch(kDiff, Mathf.Max(1, Mathf.CeilToInt(curG.width / 8f)), Mathf.Max(1, Mathf.CeilToInt(curG.height / 8f)), 1);

                cs.SetFloat("_LapWeight", weights[li]);
                cs.SetTexture(kEnh, "_LapIn", lap);
                cs.SetTexture(kEnh, "_AnalysisIn", analysis);
                cs.SetTexture(kEnh, "_EnhancedOut", enh);
                cs.Dispatch(kEnh, Mathf.Max(1, Mathf.CeilToInt(curG.width / 8f)), Mathf.Max(1, Mathf.CeilToInt(curG.height / 8f)), 1);

                var dirStrength = 0.25f + 0.65f * Mathf.Clamp01(weights[li] / (2.8f * Mathf.Max(0.01f, freqStrength)));
                cs.SetFloat("_LapDirStrength", dirStrength);
                cs.SetTexture(kDirLap, "_LapIn", enh);
                cs.SetTexture(kDirLap, "_AnalysisIn", analysis);
                cs.SetTexture(kDirLap, "_SharpenLapOut", enhDir);
                cs.Dispatch(kDirLap, Mathf.Max(1, Mathf.CeilToInt(curG.width / 8f)), Mathf.Max(1, Mathf.CeilToInt(curG.height / 8f)), 1);

                if (dumpStages)
                {
                    dumpDir ??= CreateDumpDir();
                    await DumpStageAsync(dumpDir, curG.width, curG.height, "03_L" + li + ".png", "DebugVisSignedScalar", lap, 6.0f, true, ct);
                    await DumpStageAsync(dumpDir, curG.width, curG.height, "03_EL" + li + ".png", "DebugVisSignedScalar", enh, 6.0f, true, ct);
                    await DumpStageAsync(dumpDir, curG.width, curG.height, "04_DirEL" + li + ".png", "DebugVisSignedScalar", enhDir, 6.0f, true, ct);
                }
            }

            enhancedDetailFull = NewRT(w, h, RenderTextureFormat.RHalf);
            accumPing = NewRT(w, h, RenderTextureFormat.RHalf);
            fullAdd = NewRT(w, h, RenderTextureFormat.RHalf);

            var lowG = gaussianLevels[gaussianLevels.Count - 1];
            cs.SetTexture(kUp, "_LowIn", lowG);
            cs.SetTexture(kUp, "_UpOut", enhancedDetailFull);
            cs.Dispatch(kUp, gx, gy, 1);

            for (int li = 0; li < lapCount; li++)
            {
                cs.SetTexture(kUp, "_LowIn", enhancedLaps[li]);
                cs.SetTexture(kUp, "_UpOut", fullAdd);
                cs.Dispatch(kUp, gx, gy, 1);

                cs.SetTexture(kAdd, "_AccIn", enhancedDetailFull);
                cs.SetTexture(kAdd, "_AddIn", fullAdd);
                cs.SetTexture(kAdd, "_AccOut", accumPing);
                cs.Dispatch(kAdd, gx, gy, 1);

                var tmp = enhancedDetailFull;
                enhancedDetailFull = accumPing;
                accumPing = tmp;
            }

            if (dumpStages)
            {
                dumpDir ??= CreateDumpDir();
                await DumpStageAsync(dumpDir, w, h, "04_enhanced_detail.png", "DebugVisSignedScalar", enhancedDetailFull, 4.0f, true, ct);
            }

            directionalResponse = NewRT(w, h, RenderTextureFormat.RHalf);
            cs.SetTexture(kResp, "_AnalysisIn", analysis);
            cs.SetTexture(kResp, "_EnhancedDetailIn", enhancedDetailFull);
            cs.SetTexture(kResp, "_DirectionalResponseOut", directionalResponse);
            cs.Dispatch(kResp, gx, gy, 1);

            cs.SetFloat("_Wb", 10.0f * Mathf.Max(0.05f, freqStrength));
            cs.SetFloat("_Alpha", 0.6f);
            cs.SetFloat("_Beta", 0.3f);
            cs.SetTexture(kApply, "_AnalysisIn", analysis);
            cs.SetTexture(kApply, "_BaseIn", baseRt);
            cs.SetTexture(kApply, "_DirectionalResponseIn", directionalResponse);
            cs.SetTexture(kApply, "_SharpenYOut", sharpenY);
            cs.Dispatch(kApply, gx, gy, 1);

            clampedY = NewRT(w, h, RenderTextureFormat.RHalf);
            cs.SetFloat("_ClampDelta", 0.10f);
            cs.SetTexture(kClamp, "_Source", src);
            cs.SetTexture(kClamp, "_AnalysisIn", analysis);
            cs.SetTexture(kClamp, "_SharpenYIn", sharpenY);
            cs.SetTexture(kClamp, "_SharpenYOut", clampedY);
            cs.Dispatch(kClamp, gx, gy, 1);

            faceEnhancedY = clampedY;
            var faceMask = (faceMaskTex != null && faceMaskTex.width == w && faceMaskTex.height == h) ? (Texture)faceMaskTex : null;
            if (enableFaceMidFreqReconstruction && faceMask != null)
            {
                boxTmp = NewRT(w, h, RenderTextureFormat.RHalf);
                i2 = NewRT(w, h, RenderTextureFormat.RHalf);
                meanI = NewRT(w, h, RenderTextureFormat.RHalf);
                meanII = NewRT(w, h, RenderTextureFormat.RHalf);
                aRt = NewRT(w, h, RenderTextureFormat.RHalf);
                bRt = NewRT(w, h, RenderTextureFormat.RHalf);
                meanA = NewRT(w, h, RenderTextureFormat.RHalf);
                meanB = NewRT(w, h, RenderTextureFormat.RHalf);
                guidedBase = NewRT(w, h, RenderTextureFormat.RHalf);
                blurMid = NewRT(w, h, RenderTextureFormat.RHalf);
                midFreq = NewRT(w, h, RenderTextureFormat.RHalf);
                blurHigh = NewRT(w, h, RenderTextureFormat.RHalf);
                highFreq = NewRT(w, h, RenderTextureFormat.RHalf);
                noiseFiltered = NewRT(w, h, RenderTextureFormat.RHalf);
                faceStruct = NewRT(w, h, RenderTextureFormat.RHalf);
                faceEnhancedY = NewRT(w, h, RenderTextureFormat.RHalf);

                var guidedRadius = Mathf.Clamp(Mathf.RoundToInt(Mathf.Max(w, h) * 0.003f), 6, 12);
                cs.SetInt("_BoxRadius", guidedRadius);
                BoxBlur(cs, kBoxH, kBoxV, clampedY, boxTmp, meanI, gx, gy);

                cs.SetTexture(kSquare, "_ScalarIn", clampedY);
                cs.SetTexture(kSquare, "_SquareOut", i2);
                cs.Dispatch(kSquare, gx, gy, 1);

                BoxBlur(cs, kBoxH, kBoxV, i2, boxTmp, meanII, gx, gy);

                cs.SetFloat("_GuidedEps", 0.0025f);
                cs.SetTexture(kGuidedAB, "_MeanIIn", meanI);
                cs.SetTexture(kGuidedAB, "_MeanIIIn", meanII);
                cs.SetTexture(kGuidedAB, "_AOut", aRt);
                cs.SetTexture(kGuidedAB, "_BOut", bRt);
                cs.Dispatch(kGuidedAB, gx, gy, 1);

                BoxBlur(cs, kBoxH, kBoxV, aRt, boxTmp, meanA, gx, gy);
                BoxBlur(cs, kBoxH, kBoxV, bRt, boxTmp, meanB, gx, gy);

                cs.SetTexture(kApplyGuided, "_ScalarIn", clampedY);
                cs.SetTexture(kApplyGuided, "_MeanAIn", meanA);
                cs.SetTexture(kApplyGuided, "_MeanBIn", meanB);
                cs.SetTexture(kApplyGuided, "_GuidedOut", guidedBase);
                cs.Dispatch(kApplyGuided, gx, gy, 1);

                cs.SetInt("_BoxRadius", 4);
                BoxBlur(cs, kBoxH, kBoxV, clampedY, boxTmp, blurMid, gx, gy);
                cs.SetTexture(kMid, "_ScalarIn", clampedY);
                cs.SetTexture(kMid, "_BlurIn", blurMid);
                cs.SetTexture(kMid, "_MidOut", midFreq);
                cs.Dispatch(kMid, gx, gy, 1);

                cs.SetInt("_BoxRadius", 2);
                BoxBlur(cs, kBoxH, kBoxV, clampedY, boxTmp, blurHigh, gx, gy);
                cs.SetTexture(kHigh, "_ScalarIn", clampedY);
                cs.SetTexture(kHigh, "_BlurIn", blurHigh);
                cs.SetTexture(kHigh, "_NoiseOut", highFreq);
                cs.Dispatch(kHigh, gx, gy, 1);

                cs.SetFloat("_BilateralSigmaRange", 0.03f);
                cs.SetTexture(kBilateralNoise, "_NoiseIn", highFreq);
                cs.SetTexture(kBilateralNoise, "_FaceMaskIn", faceMask);
                cs.SetTexture(kBilateralNoise, "_NoiseOut", noiseFiltered);
                cs.Dispatch(kBilateralNoise, gx, gy, 1);

                var faceStrength = Mathf.Clamp(faceDetailStrength, 0f, 6f);
                cs.SetFloat("_MidW", 1.5f * faceStrength);
                cs.SetFloat("_TextureScale", 0.25f + 0.12f * faceStrength);
                cs.SetTexture(kFaceStruct, "_Source", src);
                cs.SetTexture(kFaceStruct, "_AnalysisIn", analysis);
                cs.SetTexture(kFaceStruct, "_FaceMaskIn", faceMask);
                cs.SetTexture(kFaceStruct, "_ScalarIn", clampedY);
                cs.SetTexture(kFaceStruct, "_GuidedIn", guidedBase);
                cs.SetTexture(kFaceStruct, "_MidIn", midFreq);
                cs.SetTexture(kFaceStruct, "_NoiseFilteredIn", noiseFiltered);
                cs.SetTexture(kFaceStruct, "_FaceStructOut", faceStruct);
                cs.Dispatch(kFaceStruct, gx, gy, 1);

                cs.SetTexture(kFaceBlend, "_FaceMaskIn", faceMask);
                cs.SetTexture(kFaceBlend, "_ScalarIn", clampedY);
                cs.SetTexture(kFaceBlend, "_FaceStructIn", faceStruct);
                cs.SetTexture(kFaceBlend, "_FaceEnhancedOut", faceEnhancedY);
                cs.Dispatch(kFaceBlend, gx, gy, 1);

                if (dumpStages)
                {
                    dumpDir ??= CreateDumpDir();
                    await DumpStageAsync(dumpDir, w, h, "08_face_baseY.png", "DebugVisYScalar", guidedBase, 1f, true, ct);
                    await DumpStageAsync(dumpDir, w, h, "08_face_midFreq.png", "DebugVisSignedScalar", midFreq, 4.0f, true, ct);
                    await DumpStageAsync(dumpDir, w, h, "08_face_noiseTex.png", "DebugVisSignedScalar", noiseFiltered, 4.0f, true, ct);
                    await DumpStageAsync(dumpDir, w, h, "08_face_structY.png", "DebugVisYScalar", faceStruct, 1f, true, ct);
                    await DumpStageAsync(dumpDir, w, h, "08_face_enhancedY.png", "DebugVisYScalar", faceEnhancedY, 1f, true, ct);
                }
            }

            var pseudo = GetComponent<PseudoGANPrior>();
            if (pseudo != null && pseudo.enabled && pseudo.enablePseudoGanPrior && faceMask != null && faceEnhancedY != null)
            {
                var pr = pseudo.Process(src, analysis, faceMask, faceEnhancedY, gx, gy);
                if (string.IsNullOrWhiteSpace(pr.error) && pr.outputY != null)
                {
                    pseudoGanY = pr.outputY;
                    pseudoTmp0 = pr.temp0;
                    pseudoTmp1 = pr.temp1;
                    pseudoTmp2 = pr.temp2;
                    pseudoTmp3 = pr.temp3;
                    pseudoTmp4 = pr.temp4;
                    pseudoTmp5 = pr.temp5;

                    if (faceEnhancedY != null && faceEnhancedY != clampedY)
                        SafeReleaseRT(faceEnhancedY);
                    faceEnhancedY = pseudoGanY;

                    if (dumpStages)
                    {
                        dumpDir ??= CreateDumpDir();
                        if (pseudoTmp0 != null) await DumpStageAsync(dumpDir, w, h, "09_pseudo_prior.png", "DebugVisScalar", pseudoTmp0, 1f, true, ct);
                        if (pseudoTmp1 != null) await DumpStageAsync(dumpDir, w, h, "09_pseudo_synth.png", "DebugVisSignedScalar", pseudoTmp1, 6.0f, true, ct);
                        if (pseudoTmp2 != null) await DumpStageAsync(dumpDir, w, h, "09_pseudo_contrastDelta.png", "DebugVisSignedScalar", pseudoTmp2, 60.0f, true, ct);
                        if (pseudoTmp3 != null) await DumpStageAsync(dumpDir, w, h, "09_pseudo_pseudoY.png", "DebugVisYScalar", pseudoTmp3, 1f, true, ct);
                        if (pseudoGanY != null) await DumpStageAsync(dumpDir, w, h, "09_pseudo_finalY.png", "DebugVisYScalar", pseudoGanY, 1f, true, ct);
                        if (pseudoTmp4 != null) await DumpStageAsync(dumpDir, w, h, "09_pseudo_delta_preClamp.png", "DebugVisSignedScalar", pseudoTmp4, 60.0f, true, ct);
                        if (pseudoTmp5 != null) await DumpStageAsync(dumpDir, w, h, "09_pseudo_delta_final.png", "DebugVisSignedScalar", pseudoTmp5, 60.0f, true, ct);
                    }
                }
                else
                {
                    SafeReleaseRT(pr.outputY);
                    SafeReleaseRT(pr.temp0);
                    SafeReleaseRT(pr.temp1);
                    SafeReleaseRT(pr.temp2);
                    SafeReleaseRT(pr.temp3);
                    SafeReleaseRT(pr.temp4);
                    SafeReleaseRT(pr.temp5);
                }
            }

            if (dumpStages)
            {
                dumpDir ??= CreateDumpDir();
                await DumpStageAsync(dumpDir, w, h, "05_directional_response.png", "DebugVisSignedScalar", directionalResponse, 6.0f, true, ct);
                await DumpStageAsync(dumpDir, w, h, "06_sharpenY.png", "DebugVisYScalar", sharpenY, 1f, true, ct);
                await DumpStageAsync(dumpDir, w, h, "07_clampedY.png", "DebugVisYScalar", clampedY, 1f, true, ct);
            }

            if (enableNoiseSuppression)
            {
                noiseSuppressedY = NewRT(w, h, RenderTextureFormat.RHalf);
                cs.SetTexture(kNoise, "_Source", src);
                cs.SetTexture(kNoise, "_AnalysisIn", analysis);
                cs.SetTexture(kNoise, "_SharpenYIn", faceEnhancedY);
                var maskForNoise = faceMask;
                if (maskForNoise == null)
                {
                    zeroMask = NewRT(w, h, RenderTextureFormat.RHalf);
                    ClearRT(zeroMask);
                    maskForNoise = zeroMask;
                }
                cs.SetTexture(kNoise, "_FaceMaskIn", maskForNoise);
                cs.SetTexture(kNoise, "_SharpenYOut", noiseSuppressedY);
                cs.Dispatch(kNoise, gx, gy, 1);
            }

            cs.SetTexture(k5, "_Source", src);
            cs.SetTexture(k5, "_AnalysisIn", analysis);
            cs.SetTexture(k5, "_SharpenYIn", noiseSuppressedY != null ? noiseSuppressedY : faceEnhancedY);
            cs.SetTexture(k5, "_Result", finalRt);
            cs.Dispatch(k5, gx, gy, 1);

            if (dumpStages)
            {
                dumpDir ??= CreateDumpDir();
                await DumpStageAsync(dumpDir, w, h, "05_result.png", "DebugVisCopyRgb", finalRt, 1f, false, ct);
            }

            var tex = await ReadbackTextureAsync(finalRt, w, h, ct);

            if (dumpStages && !string.IsNullOrWhiteSpace(dumpDir))
                OpenFolderInShell(dumpDir);

            return new GpuSharpenResult { texture = tex, dumpDir = dumpDir };
        }
        catch (OperationCanceledException)
        {
            return default;
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            return new GpuSharpenResult { error = e.Message, dumpDir = dumpDir };
        }
        finally
        {
            SafeReleaseRT(analysis);
            SafeReleaseRT(mask);
            SafeReleaseRT(baseRt);
            SafeReleaseRT(detail);
            for (int i = 1; i < gaussianLevels.Count; i++)
                SafeReleaseRT(gaussianLevels[i]);
            for (int i = 0; i < upTemps.Count; i++)
                SafeReleaseRT(upTemps[i]);
            for (int i = 0; i < laplacians.Count; i++)
                SafeReleaseRT(laplacians[i]);
            for (int i = 0; i < enhancedLaps.Count; i++)
                SafeReleaseRT(enhancedLaps[i]);
            SafeReleaseRT(enhancedDetailFull);
            SafeReleaseRT(accumPing);
            SafeReleaseRT(fullAdd);
            SafeReleaseRT(directionalResponse);
            SafeReleaseRT(sharpenY);
            SafeReleaseRT(clampedY);
            SafeReleaseRT(noiseSuppressedY);
            SafeReleaseRT(boxTmp);
            SafeReleaseRT(i2);
            SafeReleaseRT(meanI);
            SafeReleaseRT(meanII);
            SafeReleaseRT(aRt);
            SafeReleaseRT(bRt);
            SafeReleaseRT(meanA);
            SafeReleaseRT(meanB);
            SafeReleaseRT(guidedBase);
            SafeReleaseRT(blurMid);
            SafeReleaseRT(midFreq);
            SafeReleaseRT(blurHigh);
            SafeReleaseRT(highFreq);
            SafeReleaseRT(noiseFiltered);
            SafeReleaseRT(faceStruct);
            SafeReleaseRT(zeroMask);
            SafeReleaseRT(pseudoTmp0);
            SafeReleaseRT(pseudoTmp1);
            SafeReleaseRT(pseudoTmp2);
            SafeReleaseRT(pseudoTmp3);
            SafeReleaseRT(pseudoTmp4);
            SafeReleaseRT(pseudoTmp5);
            if (faceEnhancedY != null && faceEnhancedY != clampedY)
                SafeReleaseRT(faceEnhancedY);
            SafeReleaseRT(finalRt);
        }
    }

    private static RenderTexture NewRT(int w, int h, RenderTextureFormat fmt)
    {
        var rt = new RenderTexture(w, h, 0, fmt, RenderTextureReadWrite.Linear) { enableRandomWrite = true };
        rt.Create();
        return rt;
    }

    private static void SafeReleaseRT(RenderTexture rt)
    {
        if (rt == null)
            return;
        try { rt.Release(); } catch { }
        Destroy(rt);
    }

    private static void BoxBlur(ComputeShader cs, int kBoxH, int kBoxV, Texture input, RenderTexture tmp, RenderTexture output, int gx, int gy)
    {
        cs.SetTexture(kBoxH, "_ScalarIn", input);
        cs.SetTexture(kBoxH, "_BoxOut", tmp);
        cs.Dispatch(kBoxH, gx, gy, 1);
        cs.SetTexture(kBoxV, "_ScalarIn", tmp);
        cs.SetTexture(kBoxV, "_BoxOut", output);
        cs.Dispatch(kBoxV, gx, gy, 1);
    }

    private static void ClearRT(RenderTexture rt)
    {
        if (rt == null)
            return;
        var prev = RenderTexture.active;
        RenderTexture.active = rt;
        GL.Clear(false, true, Color.black);
        RenderTexture.active = prev;
    }

    private async UniTask DumpStageAsync(string dir, int w, int h, string fileName, string visKernel, Texture srcTex, float debugScale, bool scalarInput, CancellationToken ct)
    {
        var cs = GetComputeShader();
        if (cs == null) return;
        if (string.IsNullOrWhiteSpace(dir)) return;
        if (srcTex == null) return;

        var k = GetKernel(visKernel);
        if (k < 0) return;

        var vis = NewRT(w, h, RenderTextureFormat.ARGB32);
        try
        {
            ct.ThrowIfCancellationRequested();
            cs.SetTexture(k, scalarInput ? "_DebugScalarIn" : "_DebugIn", srcTex);
            cs.SetFloat("_DebugScale", debugScale);
            cs.SetTexture(k, "_Result", vis);
            cs.Dispatch(k, Mathf.CeilToInt(w / 8f), Mathf.CeilToInt(h / 8f), 1);
            var tex = await ReadbackTextureAsync(vis, w, h, ct);
            if (tex == null) return;
            try
            {
                var bytes = tex.EncodeToPNG();
                var path = Path.Combine(dir, fileName);
                await File.WriteAllBytesAsync(path, bytes, ct);
            }
            catch
            {
            }
            finally
            {
                Destroy(tex);
            }
        }
        finally
        {
            SafeReleaseRT(vis);
        }
    }

    private static async UniTask<Texture2D> ReadbackTextureAsync(RenderTexture rt, int w, int h, CancellationToken ct)
    {
        var tcs = new UniTaskCompletionSource<AsyncGPUReadbackRequest>();
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, req => tcs.TrySetResult(req));
        var r = await tcs.Task.AttachExternalCancellation(ct);
        if (r.hasError)
            return null;

        var data = r.GetData<byte>();
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
        var dir = Path.Combine(root, "AIImage_GPUSharpen_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        try { Directory.CreateDirectory(dir); } catch { }
        return dir;
    }

#if !UNITY_WEBGL
    private static void OpenFolderInShell(string directoryPath)
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
}

public struct GpuSharpenResult
{
    public Texture2D texture;
    public string dumpDir;
    public string error;
}
