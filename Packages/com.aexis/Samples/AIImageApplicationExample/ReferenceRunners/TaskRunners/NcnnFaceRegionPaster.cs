using System;
using System.Collections.Generic;
using UnityEngine;

namespace Aexis.Samples.Runners
{

public static class NcnnFaceRegionPaster
{
    private readonly struct AffineCoeffs
    {
        public readonly float m00;
        public readonly float m01;
        public readonly float m02;
        public readonly float m10;
        public readonly float m11;
        public readonly float m12;

        public AffineCoeffs(float m00, float m01, float m02, float m10, float m11, float m12)
        {
            this.m00 = m00;
            this.m01 = m01;
            this.m02 = m02;
            this.m10 = m10;
            this.m11 = m11;
            this.m12 = m12;
        }
    }

    public static void PasteAlignedFaceWithSoftMask(
        Texture2D baseTex,
        Texture2D restoredFace,
        float alignedToSourceM00,
        float alignedToSourceM01,
        float alignedToSourceM02,
        float alignedToSourceM10,
        float alignedToSourceM11,
        float alignedToSourceM12)
    {
        if (baseTex == null || restoredFace == null)
            return;

        var alignedToSource = new AffineCoeffs(
            alignedToSourceM00,
            alignedToSourceM01,
            alignedToSourceM02,
            alignedToSourceM10,
            alignedToSourceM11,
            alignedToSourceM12);
        if (!TryInvert(alignedToSource, out var pasteSampleTransform))
            return;

        var imageW = baseTex.width;
        var imageH = baseTex.height;
        var warpedBounds = ComputeWarpedAlignedBounds(alignedToSource, imageW, imageH);
        if (warpedBounds.width <= 0 || warpedBounds.height <= 0)
            return;

        var basePixels = baseTex.GetPixels32();
        var facePixels = restoredFace.GetPixels32();
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
        MaskWarpedFace(invRestored, invMaskErosion, pastedFace, firstErodeBounds, imageW);

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
        var softMask = new float[imageW * imageH];
        GaussianBlurMaskToFloat(invMaskCenter, imageW, imageH, blurKernelSize, blendBounds, softMask);

        BlendPastedFace(basePixels, pastedFace, softMask, blendBounds, imageW);
        baseTex.SetPixels32(basePixels);
        baseTex.Apply(false, false);
    }

    public static Texture2D ComposeRectWithSoftMask(Texture2D baseTex, Texture2D overlayCrop, Texture2D mask, RectInt rect, float maskThreshold01)
    {
        if (baseTex == null || overlayCrop == null)
            return null;

        rect = ClampRect(rect, baseTex.width, baseTex.height);
        if (rect.width <= 0 || rect.height <= 0)
            return null;

        var imageW = baseTex.width;
        var imageH = baseTex.height;
        var basePixels = baseTex.GetPixels32();
        var overlayPixels = overlayCrop.GetPixels32();
        if (overlayPixels == null || overlayPixels.Length == 0)
            return null;

        var baseMask = BuildBaseMask(mask, imageW, imageH, rect, maskThreshold01, out var maskBounds);
        if (baseMask == null)
            return null;

        var invMaskErosion = new byte[imageW * imageH];
        var firstErodeBounds = ExpandRectPixels(rect, imageW, imageH, 4);
        ErodeMask(baseMask, invMaskErosion, imageW, imageH, BuildEllipseKernelOffsets(4, 4), firstErodeBounds);

        var totalFaceArea = CountNonZero(invMaskErosion, firstErodeBounds, imageW);
        if (totalFaceArea <= 0)
            totalFaceArea = Mathf.Max(1, rect.width * rect.height);

        var pastedFace = new Color32[imageW * imageH];
        FillRectPastedFace(overlayPixels, overlayCrop.width, overlayCrop.height, rect, imageW, pastedFace);
        ApplyMaskToPastedFace(pastedFace, invMaskErosion, firstErodeBounds, imageW);

        var wEdge = Mathf.Max(0, Mathf.FloorToInt(Mathf.Sqrt(totalFaceArea) / 20f));
        var erosionRadius = wEdge * 2;
        var invMaskCenter = new byte[imageW * imageH];
        if (erosionRadius >= 2)
        {
            var centerBounds = ExpandRectPixels(maskBounds, imageW, imageH, erosionRadius + 2);
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
        var blendBounds = ExpandRectPixels(maskBounds, imageW, imageH, blurRadius * 2 + 4);
        var softMask = new float[imageW * imageH];
        GaussianBlurMaskToFloat(invMaskCenter, imageW, imageH, blurKernelSize, blendBounds, softMask);

        BlendPastedFace(basePixels, pastedFace, softMask, blendBounds, imageW);

        var tex = new Texture2D(imageW, imageH, TextureFormat.RGBA32, false, true);
        tex.SetPixels32(basePixels);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
    }

    private static RectInt ComputeWarpedAlignedBounds(AffineCoeffs alignedToSource, int imageW, int imageH)
    {
        var corners = new[]
        {
            Transform(alignedToSource, new Vector2(0f, 0f)),
            Transform(alignedToSource, new Vector2(511f, 0f)),
            Transform(alignedToSource, new Vector2(0f, 511f)),
            Transform(alignedToSource, new Vector2(511f, 511f))
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
        AffineCoeffs pasteSampleTransform,
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
                var alignedPos = Transform(pasteSampleTransform, new Vector2(x + 0.5f, y + 0.5f));
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

    private static void MaskWarpedFace(Color32[] invRestored, byte[] invMask, Color32[] pastedFace, RectInt bounds, int imageW)
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

    private static byte[] BuildBaseMask(Texture2D mask, int imageW, int imageH, RectInt rect, float threshold01, out RectInt maskBounds)
    {
        var baseMask = new byte[imageW * imageH];
        maskBounds = rect;

        if (mask != null && mask.width == imageW && mask.height == imageH)
        {
            var pixels = mask.GetPixels();
            var minX = imageW;
            var minY = imageH;
            var maxX = -1;
            var maxY = -1;
            for (var y = 0; y < imageH; y++)
            {
                for (var x = 0; x < imageW; x++)
                {
                    var value = pixels[y * imageW + x].r;
                    if (value <= 0f)
                        continue;

                    baseMask[y * imageW + x] = 255;
                    if (value >= threshold01)
                    {
                        minX = Mathf.Min(minX, x);
                        minY = Mathf.Min(minY, y);
                        maxX = Mathf.Max(maxX, x);
                        maxY = Mathf.Max(maxY, y);
                    }
                }
            }

            if (maxX >= minX && maxY >= minY)
                maskBounds = new RectInt(minX, minY, maxX - minX + 1, maxY - minY + 1);
            else
                FillRectMask(baseMask, rect, imageW);
        }
        else
        {
            FillRectMask(baseMask, rect, imageW);
        }

        return baseMask;
    }

    private static void FillRectMask(byte[] baseMask, RectInt rect, int imageW)
    {
        if (baseMask == null || rect.width <= 0 || rect.height <= 0)
            return;

        for (var y = rect.yMin; y < rect.yMax; y++)
        {
            for (var x = rect.xMin; x < rect.xMax; x++)
                baseMask[y * imageW + x] = 255;
        }
    }

    private static void FillRectPastedFace(Color32[] overlayPixels, int overlayW, int overlayH, RectInt rect, int imageW, Color32[] pastedFace)
    {
        if (overlayPixels == null || pastedFace == null)
            return;

        for (var y = 0; y < rect.height; y++)
        {
            var srcY = Mathf.Clamp(y, 0, overlayH - 1);
            var dstY = rect.y + y;
            for (var x = 0; x < rect.width; x++)
            {
                var srcX = Mathf.Clamp(x, 0, overlayW - 1);
                pastedFace[dstY * imageW + rect.x + x] = overlayPixels[srcY * overlayW + srcX];
            }
        }
    }

    private static void ApplyMaskToPastedFace(Color32[] pastedFace, byte[] mask, RectInt bounds, int imageW)
    {
        if (pastedFace == null || mask == null)
            return;

        for (var y = bounds.yMin; y < bounds.yMax; y++)
        {
            for (var x = bounds.xMin; x < bounds.xMax; x++)
            {
                var idx = y * imageW + x;
                if (mask[idx] == 0)
                    pastedFace[idx] = default;
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
            var weight = Mathf.Exp(-(x * x) * invTwoSigmaSq);
            kernel[i] = weight;
            sum += weight;
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

    private static bool TryInvert(AffineCoeffs src, out AffineCoeffs inverse)
    {
        var det = src.m00 * src.m11 - src.m01 * src.m10;
        if (Mathf.Abs(det) < 1e-8f)
        {
            inverse = default;
            return false;
        }

        var inv = 1f / det;
        var i00 = src.m11 * inv;
        var i01 = -src.m01 * inv;
        var i10 = -src.m10 * inv;
        var i11 = src.m00 * inv;
        var i02 = -(i00 * src.m02 + i01 * src.m12);
        var i12 = -(i10 * src.m02 + i11 * src.m12);
        inverse = new AffineCoeffs(i00, i01, i02, i10, i11, i12);
        return true;
    }

    private static Vector2 Transform(AffineCoeffs affine, Vector2 p)
    {
        return new Vector2(
            affine.m00 * p.x + affine.m01 * p.y + affine.m02,
            affine.m10 * p.x + affine.m11 * p.y + affine.m12);
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

    private static RectInt ClampRect(RectInt rect, int imageW, int imageH)
    {
        var x0 = Mathf.Clamp(rect.x, 0, Mathf.Max(0, imageW));
        var y0 = Mathf.Clamp(rect.y, 0, Mathf.Max(0, imageH));
        var x1 = Mathf.Clamp(rect.x + rect.width, 0, Mathf.Max(0, imageW));
        var y1 = Mathf.Clamp(rect.y + rect.height, 0, Mathf.Max(0, imageH));
        return new RectInt(x0, y0, Mathf.Max(0, x1 - x0), Mathf.Max(0, y1 - y0));
    }
}

}