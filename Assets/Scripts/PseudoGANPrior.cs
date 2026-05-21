using System;
using System.Collections.Generic;
using UnityEngine;

public sealed class PseudoGANPrior : MonoBehaviour
{
    public bool enablePseudoGanPrior = true;
    public float alpha = 0.90f;
    public float beta = 0.60f;
    public float synthStrength = 1.25f;
    public float boost = 2.6f;
    public float contrastK = 12.0f;
    public float maxDelta = 0.22f;
    public float clampToOriginalMax = 0.24f;
    public float edgeProtect = 0.65f;
    public int priorSmallRadius = 2;
    public int priorLargeRadius = 11;

    private ComputeShader _cs;
    private readonly Dictionary<string, int> _kernelIds = new Dictionary<string, int>(StringComparer.Ordinal);

    public ComputeShader GetComputeShader()
    {
        if (_cs == null)
            _cs = Resources.Load<ComputeShader>("PseudoGANPrior");
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

    public PseudoGanPriorResult Process(Texture2D original, RenderTexture analysis, Texture faceMask, RenderTexture faceY, int gx, int gy)
    {
        if (!enablePseudoGanPrior)
            return default;
        if (original == null || analysis == null || faceMask == null || faceY == null)
            return default;
        if (!analysis.IsCreated() || !faceY.IsCreated())
            return default;
        if (analysis.width != original.width || analysis.height != original.height)
            return default;
        if (faceY.width != original.width || faceY.height != original.height)
            return default;

        var cs = GetComputeShader();
        if (cs == null)
            return new PseudoGanPriorResult { error = "PseudoGANPrior.compute not found" };

        var kPrior = GetKernel("BuildSkinPrior");
        var kSynth = GetKernel("SynthesizeTexture");
        var kContrast = GetKernel("LocalContrast");
        var kInject = GetKernel("PseudoInject");
        var kClamp = GetKernel("IdentityClamp");
        var kDiff = GetKernel("DiffScalar");
        if (kPrior < 0 || kSynth < 0 || kContrast < 0 || kInject < 0 || kClamp < 0 || kDiff < 0)
            return new PseudoGanPriorResult { error = "PseudoGANPrior kernels not found" };

        var w = original.width;
        var h = original.height;

        var prior = NewRT(w, h, RenderTextureFormat.RHalf);
        var synth = NewRT(w, h, RenderTextureFormat.RHalf);
        var contrast = NewRT(w, h, RenderTextureFormat.RHalf);
        var pseudo = NewRT(w, h, RenderTextureFormat.RHalf);
        var preDelta = NewRT(w, h, RenderTextureFormat.RHalf);
        var finalY = NewRT(w, h, RenderTextureFormat.RHalf);
        var finalDelta = NewRT(w, h, RenderTextureFormat.RHalf);

        cs.SetInt("_PriorSmallRadius", Mathf.Clamp(priorSmallRadius, 1, 12));
        cs.SetInt("_PriorLargeRadius", Mathf.Clamp(priorLargeRadius, 3, 24));
        cs.SetFloat("_SynthStrength", Mathf.Clamp(synthStrength, 0f, 2f));
        cs.SetFloat("_Boost", Mathf.Clamp(boost, 0f, 200f));
        cs.SetFloat("_ContrastK", Mathf.Clamp(contrastK, 0.5f, 24f));
        cs.SetFloat("_Alpha", Mathf.Clamp(alpha, 0f, 2f));
        cs.SetFloat("_Beta", Mathf.Clamp(beta, 0f, 2f));
        cs.SetFloat("_MaxDelta", Mathf.Clamp(maxDelta, 0f, 0.35f));
        cs.SetFloat("_ClampToOrigMax", Mathf.Clamp(clampToOriginalMax, 0.05f, 0.95f));
        cs.SetFloat("_EdgeProtect", Mathf.Clamp01(edgeProtect));

        cs.SetTexture(kPrior, "_FaceYIn", faceY);
        cs.SetTexture(kPrior, "_FaceMaskIn", faceMask);
        cs.SetTexture(kPrior, "_PriorOut", prior);
        cs.Dispatch(kPrior, gx, gy, 1);

        cs.SetTexture(kSynth, "_AnalysisIn", analysis);
        cs.SetTexture(kSynth, "_FaceMaskIn", faceMask);
        cs.SetTexture(kSynth, "_PriorIn", prior);
        cs.SetTexture(kSynth, "_SynthOut", synth);
        cs.Dispatch(kSynth, gx, gy, 1);

        cs.SetTexture(kContrast, "_FaceYIn", faceY);
        cs.SetTexture(kContrast, "_FaceMaskIn", faceMask);
        cs.SetTexture(kContrast, "_SynthIn", synth);
        cs.SetTexture(kContrast, "_ContrastOut", contrast);
        cs.Dispatch(kContrast, gx, gy, 1);

        cs.SetTexture(kInject, "_FaceYIn", faceY);
        cs.SetTexture(kInject, "_FaceMaskIn", faceMask);
        cs.SetTexture(kInject, "_PriorIn", prior);
        cs.SetTexture(kInject, "_SynthIn", synth);
        cs.SetTexture(kInject, "_ContrastIn", contrast);
        cs.SetTexture(kInject, "_PseudoOut", pseudo);
        cs.Dispatch(kInject, gx, gy, 1);

        cs.SetTexture(kDiff, "_DiffAIn", pseudo);
        cs.SetTexture(kDiff, "_DiffBIn", faceY);
        cs.SetTexture(kDiff, "_DiffOut", preDelta);
        cs.Dispatch(kDiff, gx, gy, 1);

        cs.SetTexture(kClamp, "_Source", original);
        cs.SetTexture(kClamp, "_AnalysisIn", analysis);
        cs.SetTexture(kClamp, "_FaceMaskIn", faceMask);
        cs.SetTexture(kClamp, "_FaceYIn", faceY);
        cs.SetTexture(kClamp, "_PseudoIn", pseudo);
        cs.SetTexture(kClamp, "_FinalOut", finalY);
        cs.Dispatch(kClamp, gx, gy, 1);

        cs.SetTexture(kDiff, "_DiffAIn", finalY);
        cs.SetTexture(kDiff, "_DiffBIn", faceY);
        cs.SetTexture(kDiff, "_DiffOut", finalDelta);
        cs.Dispatch(kDiff, gx, gy, 1);

        return new PseudoGanPriorResult
        {
            outputY = finalY,
            temp0 = prior,
            temp1 = synth,
            temp2 = contrast,
            temp3 = pseudo,
            temp4 = preDelta,
            temp5 = finalDelta
        };
    }

    private static RenderTexture NewRT(int w, int h, RenderTextureFormat fmt)
    {
        var rt = new RenderTexture(w, h, 0, fmt, RenderTextureReadWrite.Linear) { enableRandomWrite = true };
        rt.Create();
        return rt;
    }
}

public struct PseudoGanPriorResult
{
    public RenderTexture outputY;
    public RenderTexture temp0;
    public RenderTexture temp1;
    public RenderTexture temp2;
    public RenderTexture temp3;
    public RenderTexture temp4;
    public RenderTexture temp5;
    public string error;
}
