using System;
using System.IO;

namespace NcnnCompute
{
    public sealed class NcnnBinReader : IDisposable
    {
        private const uint TagFp16 = 0x01306B47;
        private const uint TagInt8 = 0x000D4B38;
        private const uint TagFloat32ExtraScale = 0x0002C056;

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
            var a = new float[count];
            for (var i = 0; i < count; i++)
                a[i] = _br.ReadSingle();
            return a;
        }

        public float[] ReadFp16ArrayAsFloat32(int count)
        {
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            var a = new float[count];
            for (var i = 0; i < count; i++)
            {
                var h = _br.ReadUInt16();
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

            Skip((long)count * 4);
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
            var sign = (h >> 15) & 1;
            var exp = (h >> 10) & 0x1F;
            var mant = h & 0x03FF;

            if (exp == 0)
            {
                if (mant == 0)
                    return sign != 0 ? -0f : 0f;
                var v = mant / 1024f;
                var f = (float)Math.Pow(2, -14) * v;
                return sign != 0 ? -f : f;
            }

            if (exp == 31)
            {
                if (mant == 0)
                    return sign != 0 ? float.NegativeInfinity : float.PositiveInfinity;
                return float.NaN;
            }

            var baseV = 1f + mant / 1024f;
            var scale = (float)Math.Pow(2, exp - 15);
            var r = baseV * scale;
            return sign != 0 ? -r : r;
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

            return ReadFloat32Array(count);
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
