using System;
using System.IO;
using System.Runtime.InteropServices;

namespace NcnnCompute
{
    public sealed class NcnnBinReader : IDisposable
    {
        private const uint TagFp16 = 0x01306B47;
        private const uint TagInt8 = 0x000D4B38;
        private const uint TagFloat32ExtraScale = 0x0002C056;

        private static readonly float[] HalfToSingleTable = BuildHalfToSingleTable();

        private readonly Stream _stream;
        private readonly BinaryReader _br;

        public long Position => _stream.CanSeek ? _stream.Position : 0;

        public NcnnBinReader(Stream stream)
        {
            _stream = stream ?? throw new ArgumentNullException(nameof(stream));
            _br = new BinaryReader(_stream);
        }

        public int ReadInt32()
        {
            return _br.ReadInt32();
        }

        public float[] ReadFloat32Array(int count)
        {
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (count == 0)
                return Array.Empty<float>();

            var bytes = ReadBytesChecked(checked(count * sizeof(float)));
            var a = new float[count];
            if (BitConverter.IsLittleEndian)
            {
                Buffer.BlockCopy(bytes, 0, a, 0, bytes.Length);
                return a;
            }

            for (var i = 0; i < count; i++)
                a[i] = BitConverter.ToSingle(bytes, i * sizeof(float));
            return a;
        }

        public float[] ReadFp16ArrayAsFloat32(int count, bool align4 = false)
        {
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            var readCount = align4 ? Align4(count * 2) : count * 2;
            var bytes = ReadBytesChecked(readCount);
            var a = new float[count];
            for (var i = 0; i < count; i++)
            {
                var h = (ushort)(bytes[i * 2 + 0] | (bytes[i * 2 + 1] << 8));
                a[i] = HalfToSingle(h);
            }
            return a;
        }

        public float[] ReadNcnnMatAsFloat32(int w, int h, int d, int c, int loadType)
        {
            if (w < 0 || h < 0 || d < 0 || c < 0)
                throw new ArgumentOutOfRangeException(nameof(w));

            int count;
            if (d != 0) count = checked(w * h * d * c);
            else if (c != 0) count = checked(w * h * c);
            else if (h != 0) count = checked(w * h);
            else if (w != 0) count = w;
            else count = 1;

            return ReadNcnnArrayAsFloat32(count, loadType);
        }

        public void SkipNcnnMat(int w, int h, int d, int c, int loadType)
        {
            if (w < 0 || h < 0 || d < 0 || c < 0)
                throw new ArgumentOutOfRangeException(nameof(w));

            int count;
            if (d != 0) count = checked(w * h * d * c);
            else if (c != 0) count = checked(w * h * c);
            else if (h != 0) count = checked(w * h);
            else if (w != 0) count = w;
            else count = 1;

            SkipNcnnArray(count, loadType);
        }

        public void SkipNcnnArray(int count, int loadType)
        {
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));

            if (loadType == 1)
            {
                Skip((long)count * 4);
                return;
            }

            if (loadType != 0)
                throw new NotSupportedException("unsupported loadType: " + loadType);

            var flag = ReadUInt32();
            var f0 = (byte)(flag & 0xFF);
            var f1 = (byte)((flag >> 8) & 0xFF);
            var f2 = (byte)((flag >> 16) & 0xFF);
            var f3 = (byte)((flag >> 24) & 0xFF);
            var sum = f0 + f1 + f2 + f3;

            if (flag == TagFp16)
            {
                Skip(Align4((long)count * 2));
                return;
            }

            if (flag == TagInt8)
            {
                Skip(Align4((long)count));
                return;
            }

            if (flag == TagFloat32ExtraScale)
            {
                Skip((long)count * 4);
                return;
            }

            if (sum != 0)
            {
                Skip(256L * 4);
                Skip(Align4((long)count));
                return;
            }

            if (f0 == 0)
            {
                Skip((long)count * 4);
                return;
            }
        }

        public byte[] ReadBytes(int count)
        {
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            return _br.ReadBytes(count);
        }

        public void Skip(long byteCount)
        {
            if (byteCount < 0)
                throw new ArgumentOutOfRangeException(nameof(byteCount));
            if (byteCount == 0)
                return;
            if (_stream.CanSeek)
                _stream.Seek(byteCount, SeekOrigin.Current);
            else
                _br.ReadBytes((int)byteCount);
        }

        public void Dispose()
        {
            try { _br?.Dispose(); } catch { }
        }

        private static float HalfToSingle(ushort h)
        {
            return HalfToSingleTable[h];
        }

        private static float[] BuildHalfToSingleTable()
        {
            var table = new float[1 << 16];
            for (var i = 0; i < table.Length; i++)
                table[i] = UInt32BitsToSingle(HalfToSingleBits((ushort)i));
            return table;
        }

        private static uint HalfToSingleBits(ushort h)
        {
            var sign = (uint)(h & 0x8000) << 16;
            var exp = (uint)(h & 0x7C00) >> 10;
            var mant = (uint)(h & 0x03FF);

            if (exp == 0)
            {
                if (mant == 0)
                    return sign;

                var e = -14;
                while ((mant & 0x0400) == 0)
                {
                    mant <<= 1;
                    e--;
                }

                mant &= 0x03FF;
                return sign | (uint)(e + 127) << 23 | mant << 13;
            }

            if (exp == 0x1F)
                return sign | 0x7F800000u | mant << 13;

            return sign | (exp + 112u) << 23 | mant << 13;
        }

        private static float UInt32BitsToSingle(uint bits)
        {
            return new UIntFloat { UInt = bits }.Float;
        }

        [StructLayout(LayoutKind.Explicit)]
        private struct UIntFloat
        {
            [FieldOffset(0)] public uint UInt;
            [FieldOffset(0)] public float Float;
        }

        private float[] ReadNcnnArrayAsFloat32(int count, int loadType)
        {
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));

            if (loadType == 1)
                return ReadFloat32Array(count);

            if (loadType != 0)
                throw new NotSupportedException("unsupported loadType: " + loadType);

            var flag = ReadUInt32();
            var f0 = (byte)(flag & 0xFF);
            var f1 = (byte)((flag >> 8) & 0xFF);
            var f2 = (byte)((flag >> 16) & 0xFF);
            var f3 = (byte)((flag >> 24) & 0xFF);
            var sum = f0 + f1 + f2 + f3;

            if (flag == TagFp16)
            {
                var bytes = ReadBytesChecked(Align4(count * 2));
                var a = new float[count];
                for (var i = 0; i < count; i++)
                {
                    var lo = bytes[i * 2 + 0];
                    var hi = bytes[i * 2 + 1];
                    var h = (ushort)(lo | (hi << 8));
                    a[i] = HalfToSingle(h);
                }
                return a;
            }

            if (flag == TagInt8)
            {
                var bytes = ReadBytesChecked(Align4(count));
                var a = new float[count];
                for (var i = 0; i < count; i++)
                    a[i] = unchecked((sbyte)bytes[i]);
                return a;
            }

            if (flag == TagFloat32ExtraScale)
            {
                return ReadFloat32Array(count);
            }

            if (sum != 0)
            {
                var table = ReadFloat32Array(256);
                var idxBytes = ReadBytesChecked(Align4(count));
                var a = new float[count];
                for (var i = 0; i < count; i++)
                    a[i] = table[idxBytes[i]];
                return a;
            }

            if (f0 == 0)
                return ReadFloat32Array(count);

            throw new InvalidDataException("unsupported ncnn weight encoding flag: 0x" + flag.ToString("X8"));
        }

        public uint ReadUInt32()
        {
            return _br.ReadUInt32();
        }

        public void Seek(long position)
        {
            if (!_stream.CanSeek)
                throw new NotSupportedException("stream does not support seeking");
            _stream.Seek(position, SeekOrigin.Begin);
        }

        private byte[] ReadBytesChecked(int count)
        {
            var b = _br.ReadBytes(count);
            if (b.Length != count)
                throw new EndOfStreamException("ReadBytes(" + count + ") got " + b.Length);
            return b;
        }

        private static int Align4(int bytes)
        {
            return (bytes + 3) & ~3;
        }

        private static long Align4(long bytes)
        {
            return (bytes + 3) & ~3;
        }
    }
}
