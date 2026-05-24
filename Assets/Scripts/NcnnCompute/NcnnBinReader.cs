using System;
using System.IO;

namespace NcnnCompute
{
    public sealed class NcnnBinReader : IDisposable
    {
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
    }
}
