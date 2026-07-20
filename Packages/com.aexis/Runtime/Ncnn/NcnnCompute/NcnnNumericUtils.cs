using System.Runtime.InteropServices;

namespace Aexis.Ncnn
{
    internal static class NcnnNumericUtils
    {
        [StructLayout(LayoutKind.Explicit)]
        private struct UIntFloat
        {
            [FieldOffset(0)] public uint u;
            [FieldOffset(0)] public float f;
        }

        internal static float QuantizeInt8(float value)
        {
            var rounded = value >= 0f ? value + 0.5f : value - 0.5f;
            if (rounded > 127f) return 127f;
            if (rounded < -127f) return -127f;
            return (int)rounded;
        }

        internal static float CastInt8(float value)
        {
            var rounded = value >= 0f ? value + 0.5f : value - 0.5f;
            if (rounded > 127f) return 127f;
            if (rounded < -128f) return -128f;
            return (int)rounded;
        }

        internal static float ToHalfRoundedFloat(float value)
        {
            return HalfBitsToFloat(FloatToHalfBits(value));
        }

        internal static float ToBFloat16RoundedFloat(float value)
        {
            var bits = FloatToUInt32Bits(value);
            var lsb = (bits >> 16) & 1u;
            bits += 0x7FFFu + lsb;
            bits &= 0xFFFF0000u;
            return UInt32BitsToFloat(bits);
        }

        internal static float ApplyCast(float value, int typeFrom, int typeTo)
        {
            if (typeFrom == typeTo)
                return value;

            if (typeFrom == 1 && typeTo == 2)
                return ToHalfRoundedFloat(value);
            if (typeFrom == 1 && typeTo == 4)
                return ToBFloat16RoundedFloat(value);
            if (typeFrom == 1 && typeTo == 3)
                return CastInt8(value);

            if ((typeFrom == 2 || typeFrom == 3 || typeFrom == 4) && typeTo == 1)
                return value;

            return value;
        }

        private static uint FloatToUInt32Bits(float value)
        {
            return new UIntFloat { f = value }.u;
        }

        private static float UInt32BitsToFloat(uint value)
        {
            return new UIntFloat { u = value }.f;
        }

        private static ushort FloatToHalfBits(float value)
        {
            var bits = FloatToUInt32Bits(value);
            var sign = (bits >> 16) & 0x8000u;
            var mantissa = bits & 0x007FFFFFu;
            var exponent = (int)((bits >> 23) & 0xFFu);

            if (exponent == 255)
            {
                if (mantissa != 0)
                    return (ushort)(sign | 0x7C00u | (mantissa >> 13) | 1u);
                return (ushort)(sign | 0x7C00u);
            }

            var halfExp = exponent - 127 + 15;
            if (halfExp >= 31)
                return (ushort)(sign | 0x7C00u);

            if (halfExp <= 0)
            {
                if (halfExp < -10)
                    return (ushort)sign;

                mantissa |= 0x00800000u;
                var shift = 14 - halfExp;
                var halfMantissa = mantissa >> shift;
                var roundBit = (mantissa >> (shift - 1)) & 1u;
                if (roundBit != 0)
                    halfMantissa++;
                return (ushort)(sign | halfMantissa);
            }

            var roundedMantissa = mantissa + 0x00001000u;
            if ((roundedMantissa & 0x00800000u) != 0)
            {
                roundedMantissa = 0;
                halfExp++;
                if (halfExp >= 31)
                    return (ushort)(sign | 0x7C00u);
            }

            return (ushort)(sign | ((uint)halfExp << 10) | (roundedMantissa >> 13));
        }

        private static float HalfBitsToFloat(ushort value)
        {
            var sign = (uint)(value & 0x8000u) << 16;
            var exponent = (uint)(value & 0x7C00u) >> 10;
            var mantissa = (uint)(value & 0x03FFu);

            if (exponent == 0)
            {
                if (mantissa == 0)
                    return UInt32BitsToFloat(sign);

                while ((mantissa & 0x0400u) == 0)
                {
                    mantissa <<= 1;
                    exponent--;
                }

                exponent++;
                mantissa &= ~0x0400u;
            }
            else if (exponent == 31)
            {
                return UInt32BitsToFloat(sign | 0x7F800000u | (mantissa << 13));
            }

            exponent = exponent + (127u - 15u);
            mantissa <<= 13;
            return UInt32BitsToFloat(sign | (exponent << 23) | mantissa);
        }
    }
}
