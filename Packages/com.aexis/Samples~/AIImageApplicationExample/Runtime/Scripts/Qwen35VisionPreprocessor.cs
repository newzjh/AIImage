using System;
using UnityEngine;

namespace AIImage.Qwen35
{
    public static class Qwen35VisionPreprocessor
    {
        public static Vector2Int TargetImageSize(int height, int width, int patchSize = 16, int maxPatches = 49152)
        {
            if (height <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(height));
            var effective = patchSize * 2; var scale = 1f;
            while (true)
            {
                var targetH = Mathf.Max(effective, Mathf.CeilToInt(height * scale / effective) * effective);
                var targetW = Mathf.Max(effective, Mathf.CeilToInt(width * scale / effective) * effective);
                if ((targetH / patchSize) * (targetW / patchSize) <= maxPatches) return new Vector2Int(targetW, targetH);
                scale -= .02f; if (scale <= 0f) throw new InvalidOperationException("image cannot fit Qwen3.5 patch budget");
            }
        }

        public static float[] ResizeNormalize(Texture2D source, int width, int height)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width));

            // Unity exposes Texture2D pixels bottom-up, while stb_image and the ncnn
            // reference path are top-down interleaved RGB/BGR byte images.
            var pixels = source.GetPixels32();
            var sourceRgb = new byte[source.width * source.height * 3];
            for (var y = 0; y < source.height; y++)
            {
                var sourceRow = (source.height - 1 - y) * source.width;
                var targetRow = y * source.width * 3;
                for (var x = 0; x < source.width; x++)
                {
                    var pixel = pixels[sourceRow + x];
                    var target = targetRow + x * 3;
                    sourceRgb[target] = pixel.r;
                    sourceRgb[target + 1] = pixel.g;
                    sourceRgb[target + 2] = pixel.b;
                }
            }

            var resizedRgb = new byte[width * height * 3];
            ResizeBilinearC3Ncnn(
                sourceRgb,
                source.width,
                source.height,
                source.width * 3,
                resizedRgb,
                width,
                height,
                width * 3);

            var result = new float[resizedRgb.Length];
            for (var i = 0; i < resizedRgb.Length; i++)
                result[i] = (resizedRgb[i] / 255.5f - .5f) / .5f;
            return result;
        }

        public static float[] BuildDuplicatedPatches(float[] rgb, int width, int height, int patchSize = 16, int merge = 2)
        {
            if (rgb == null || rgb.Length != width * height * 3) throw new ArgumentException("rgb shape mismatch");
            var hp = height / patchSize; var wp = width / patchSize; if (hp % merge != 0 || wp % merge != 0) throw new ArgumentException("patch grid must be divisible by merge");
            var patchCount = hp * wp; var patchValues = new float[patchCount * 3 * 2 * patchSize * patchSize]; var outIndex = 0;
            for (var gh = 0; gh < hp / merge; gh++) for (var gw = 0; gw < wp / merge; gw++) for (var mh = 0; mh < merge; mh++) for (var mw = 0; mw < merge; mw++)
            {
                var ph = gh * merge + mh; var pw = gw * merge + mw;
                for (var c = 0; c < 3; c++) for (var t = 0; t < 2; t++) for (var py = 0; py < patchSize; py++) for (var px = 0; px < patchSize; px++)
                    patchValues[outIndex++] = rgb[((ph * patchSize + py) * width + pw * patchSize + px) * 3 + c];
            }
            return patchValues;
        }

        public static void BuildVisionRope2D(
            int heightPatches,
            int widthPatches,
            out float[] cos,
            out float[] sin,
            int merge = 2,
            float theta = 10000f,
            int heightSection = 16,
            int widthSection = 16,
            bool duplicateSections = true)
        {
            if (heightPatches <= 0 || widthPatches <= 0) throw new ArgumentOutOfRangeException(nameof(heightPatches));
            if (merge <= 0 || heightPatches % merge != 0 || widthPatches % merge != 0)
                throw new ArgumentException("vision RoPE patch grid must be divisible by merge");
            if (theta <= 0f || heightSection <= 0 || widthSection <= 0)
                throw new ArgumentOutOfRangeException(nameof(theta));

            var ropeDim = heightSection + widthSection;
            var outputDim = duplicateSections ? ropeDim * 2 : ropeDim;
            var rowCount = heightPatches * widthPatches;
            var inverseHeight = new float[heightSection];
            var inverseWidth = new float[widthSection];
            for (var i = 0; i < heightSection; i++) inverseHeight[i] = 1f / Mathf.Pow(theta, (float)(i * 2) / ropeDim);
            for (var i = 0; i < widthSection; i++) inverseWidth[i] = 1f / Mathf.Pow(theta, (float)(i * 2) / ropeDim);

            cos = new float[rowCount * outputDim];
            sin = new float[rowCount * outputDim];
            var row = 0;
            for (var groupY = 0; groupY < heightPatches / merge; groupY++)
            for (var groupX = 0; groupX < widthPatches / merge; groupX++)
            for (var localY = 0; localY < merge; localY++)
            for (var localX = 0; localX < merge; localX++)
            {
                var currentY = groupY * merge + localY;
                var currentX = groupX * merge + localX;
                var offset = row * outputDim;
                for (var i = 0; i < heightSection; i++)
                {
                    var angle = currentY * inverseHeight[i];
                    var cosine = Mathf.Cos(angle);
                    var sine = Mathf.Sin(angle);
                    cos[offset + i] = cosine;
                    sin[offset + i] = sine;
                    if (duplicateSections) { cos[offset + ropeDim + i] = cosine; sin[offset + ropeDim + i] = sine; }
                }
                for (var i = 0; i < widthSection; i++)
                {
                    var angle = currentX * inverseWidth[i];
                    var cosine = Mathf.Cos(angle);
                    var sine = Mathf.Sin(angle);
                    cos[offset + heightSection + i] = cosine;
                    sin[offset + heightSection + i] = sine;
                    if (duplicateSections) { cos[offset + ropeDim + heightSection + i] = cosine; sin[offset + ropeDim + heightSection + i] = sine; }
                }
                row++;
            }
        }

        private static void ResizeBilinearC3Ncnn(
            byte[] source,
            int sourceWidth,
            int sourceHeight,
            int sourceStride,
            byte[] target,
            int targetWidth,
            int targetHeight,
            int targetStride)
        {
            if (sourceWidth == targetWidth && sourceHeight == targetHeight)
            {
                for (var y = 0; y < targetHeight; y++)
                    Buffer.BlockCopy(source, y * sourceStride, target, y * targetStride, targetWidth * 3);
                return;
            }

            if (sourceWidth < 2 || sourceHeight < 2)
            {
                for (var y = 0; y < targetHeight; y++)
                {
                    var sy = Mathf.Clamp((int)((long)y * sourceHeight / targetHeight), 0, sourceHeight - 1);
                    for (var x = 0; x < targetWidth; x++)
                    {
                        var sx = Mathf.Clamp((int)((long)x * sourceWidth / targetWidth), 0, sourceWidth - 1);
                        Buffer.BlockCopy(source, sy * sourceStride + sx * 3, target, y * targetStride + x * 3, 3);
                    }
                }
                return;
            }

            const int coefficientBits = 11;
            const int coefficientScale = 1 << coefficientBits;
            var scaleX = (double)sourceWidth / targetWidth;
            var scaleY = (double)sourceHeight / targetHeight;
            var xOffsets = new int[targetWidth];
            var yOffsets = new int[targetHeight];
            var alpha = new short[targetWidth * 2];
            var beta = new short[targetHeight * 2];

            for (var x = 0; x < targetWidth; x++)
            {
                var fx = (float)((x + .5) * scaleX - .5);
                var sx = (int)Math.Floor(fx);
                fx -= sx;
                if (sx < 0) { sx = 0; fx = 0f; }
                if (sx >= sourceWidth - 1) { sx = sourceWidth - 2; fx = 1f; }
                xOffsets[x] = sx * 3;
                alpha[x * 2] = SaturateCastShort((1f - fx) * coefficientScale);
                alpha[x * 2 + 1] = SaturateCastShort(fx * coefficientScale);
            }

            for (var y = 0; y < targetHeight; y++)
            {
                var fy = (float)((y + .5) * scaleY - .5);
                var sy = (int)Math.Floor(fy);
                fy -= sy;
                if (sy < 0) { sy = 0; fy = 0f; }
                if (sy >= sourceHeight - 1) { sy = sourceHeight - 2; fy = 1f; }
                yOffsets[y] = sy;
                beta[y * 2] = SaturateCastShort((1f - fy) * coefficientScale);
                beta[y * 2 + 1] = SaturateCastShort(fy * coefficientScale);
            }

            for (var y = 0; y < targetHeight; y++)
            {
                var sourceRow0 = yOffsets[y] * sourceStride;
                var sourceRow1 = (yOffsets[y] + 1) * sourceStride;
                var targetRow = y * targetStride;
                var b0 = beta[y * 2];
                var b1 = beta[y * 2 + 1];
                for (var x = 0; x < targetWidth; x++)
                {
                    var sourceX = xOffsets[x];
                    var a0 = alpha[x * 2];
                    var a1 = alpha[x * 2 + 1];
                    var source0 = sourceRow0 + sourceX;
                    var source1 = sourceRow1 + sourceX;
                    var targetIndex = targetRow + x * 3;
                    for (var channel = 0; channel < 3; channel++)
                    {
                        var row0 = (source[source0 + channel] * a0 + source[source0 + channel + 3] * a1) >> 4;
                        var row1 = (source[source1 + channel] * a0 + source[source1 + channel + 3] * a1) >> 4;
                        var value = (((b0 * row0) >> 16) + ((b1 * row1) >> 16) + 2) >> 2;
                        target[targetIndex + channel] = (byte)Mathf.Clamp(value, 0, 255);
                    }
                }
            }
        }

        private static short SaturateCastShort(float value)
        {
            var rounded = (int)(value + (value >= 0f ? .5f : -.5f));
            return (short)Mathf.Clamp(rounded, short.MinValue, short.MaxValue);
        }
    }
}
