using UnityEngine;

namespace Aexis.Ncnn
{
    /// <summary>
    /// Winograd F(2,3) kernel transform and pack4 weight layout matching ncnn-vulkan
    /// convolution_pack4_3x3s1d1_winograd23 (non-coopmat path).
    /// </summary>
    public static class NcnnWinograd23
    {
        private static readonly float[,] Ktm =
        {
            { 1.0f, 0.0f, 0.0f },
            { 0.5f, 0.5f, 0.5f },
            { 0.5f, -0.5f, 0.5f },
            { 0.0f, 0.0f, 1.0f },
        };

        public static bool CanUse(int kernel, int pad, int inPacks, int outPacks)
        {
            return kernel == 3 && pad == 1 && inPacks >= 4 && outPacks >= 4;
        }

        public static bool ShouldPreferForShape(int w, int h, int inPacks, int outPacks)
        {
            // Runtime evidence on the repro path shows F(2,3) is a clear win for the
            // repeated 48->16 pack4 bottleneck shapes, but regresses the common ->8
            // output-pack shapes. Keep the first policy conservative and only enable
            // Winograd23 on the shape family with a strong measured benefit.
            if (w <= 0 || h <= 0)
                return false;

            return inPacks >= 48 && outPacks >= 16;
        }

        public static int BlockX(int w) => (w + 1) / 2;
        public static int BlockY(int h) => (h + 1) / 2;

        public static int BottomTmCount(int w, int h, int inPacks) => inPacks * 16 * BlockX(w) * BlockY(h);
        public static int TopTmCount(int w, int h, int outPacks) => outPacks * 16 * BlockX(w) * BlockY(h);

        public static Vector4[] PackWeightTm23(float[] w, int outC, int inC, int outPacks, int inPacks)
        {
            var tm = new float[outC * inC * 16];
            for (var oc = 0; oc < outC; oc++)
            {
                for (var ic = 0; ic < inC; ic++)
                {
                    var k9 = new float[9];
                    for (var i = 0; i < 9; i++)
                        k9[i] = w[(oc * inC + ic) * 9 + i];
                    var t16 = TransformKernel3x3(k9);
                    for (var i = 0; i < 16; i++)
                        tm[(oc * inC + ic) * 16 + i] = t16[i];
                }
            }

            // ncnn: dst = inch/4-outch/4-16, channel k holds out_pack x in_pack blocks of 4x4 lanes.
            var packed = new Vector4[16 * outPacks * inPacks * 4];
            var idx = 0;
            for (var k = 0; k < 16; k++)
            {
                for (var op = 0; op < outPacks; op++)
                {
                    for (var ip = 0; ip < inPacks; ip++)
                    {
                        for (var ol = 0; ol < 4; ol++)
                        {
                            var oc = op * 4 + ol;
                            var x0 = 0f;
                            var x1 = 0f;
                            var x2 = 0f;
                            var x3 = 0f;
                            for (var il = 0; il < 4; il++)
                            {
                                var ic = ip * 4 + il;
                                var val = oc < outC && ic < inC ? tm[(oc * inC + ic) * 16 + k] : 0f;
                                if (il == 0) x0 = val;
                                else if (il == 1) x1 = val;
                                else if (il == 2) x2 = val;
                                else x3 = val;
                            }
                            packed[idx++] = new Vector4(x0, x1, x2, x3);
                        }
                    }
                }
            }
            return packed;
        }

        private static float[] TransformKernel3x3(float[] k9)
        {
            var tmp = new float[4, 3];
            for (var i = 0; i < 4; i++)
            {
                for (var m = 0; m < 3; m++)
                {
                    tmp[i, m] = k9[m] * Ktm[i, 0] + k9[3 + m] * Ktm[i, 1] + k9[6 + m] * Ktm[i, 2];
                }
            }

            var r = new float[16];
            for (var j = 0; j < 4; j++)
            {
                for (var i = 0; i < 4; i++)
                {
                    r[j * 4 + i] = tmp[j, 0] * Ktm[i, 0] + tmp[j, 1] * Ktm[i, 1] + tmp[j, 2] * Ktm[i, 2];
                }
            }
            return r;
        }
    }
}
