using System;
using System.IO;
using System.Runtime.InteropServices;
using Aexis.Execution;

namespace Aexis.Ncnn
{
    public sealed class NcnnBinReader : AexisWeightReader
    {
        private const uint TagFp16 = 0x01306B47;
        private const uint TagInt8 = 0x000D4B38;
        private const uint TagFloat32ExtraScale = 0x0002C056;

        private static readonly float[] HalfToSingleTable = BuildHalfToSingleTable();

        private readonly Stream _stream;
        private readonly BinaryReader _br;
        private readonly NcnnQ8ArchiveReader _q8Reader;
        private readonly NcnnQ8ArchiveWriter _q8Capture;

        public override long Position => _stream.CanSeek ? _stream.Position : 0;

        public NcnnBinReader(Stream stream)
            : this(stream, null)
        {
        }

        public NcnnBinReader(Stream stream, NcnnQ8ArchiveWriter q8Capture)
        {
            _stream = stream ?? throw new ArgumentNullException(nameof(stream));
            _br = new BinaryReader(_stream);
            _q8Capture = q8Capture;
            if (ProbeQ8Archive(stream))
                _q8Reader = new NcnnQ8ArchiveReader(stream);
        }

        public bool IsQ8Archive => _q8Reader != null && _q8Reader.QuantizationBits == 8;
        public bool IsQ4Archive => _q8Reader != null && _q8Reader.QuantizationBits == 4;

        public int ReadInt32()
        {
            if (_q8Reader != null)
            {
                var bytes = _q8Reader.ReadRaw(sizeof(int));
                _q8Capture?.WriteRaw(bytes);
                return BitConverter.ToInt32(bytes, 0);
            }
            var value = _br.ReadInt32();
            _q8Capture?.WriteRaw(BitConverter.GetBytes(value));
            return value;
        }

        public override float[] ReadFloat32Array(int count)
        {
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (count == 0)
                return Array.Empty<float>();

            if (_q8Reader != null)
            {
                var values = _q8Reader.ReadArray(count, requireRaw: true);
                _q8Capture?.WriteFloat32Array(values);
                return values;
            }
            var bytes = ReadBytesChecked(checked(count * sizeof(float)));
            var a = new float[count];
            if (BitConverter.IsLittleEndian)
            {
                Buffer.BlockCopy(bytes, 0, a, 0, bytes.Length);
                _q8Capture?.WriteFloat32Array(a);
                return a;
            }

            for (var i = 0; i < count; i++)
                a[i] = BitConverter.ToSingle(bytes, i * sizeof(float));
            _q8Capture?.WriteFloat32Array(a);
            return a;
        }

        public float[] ReadFp16ArrayAsFloat32(int count, bool align4 = false)
        {
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (_q8Reader != null)
            {
                var values = _q8Reader.ReadArray(count, requireRaw: false);
                _q8Capture?.WriteNcnnArray(values);
                return values;
            }
            var readCount = align4 ? Align4(count * 2) : count * 2;
            var bytes = ReadBytesChecked(readCount);
            var a = new float[count];
            for (var i = 0; i < count; i++)
            {
                var h = (ushort)(bytes[i * 2 + 0] | (bytes[i * 2 + 1] << 8));
                a[i] = HalfToSingle(h);
            }
            _q8Capture?.WriteNcnnArray(a);
            return a;
        }

        public override float[] ReadTensorAsFloat32(int w, int h, int d, int c, int loadType)
        {
            if (w < 0 || h < 0 || d < 0 || c < 0)
                throw new ArgumentOutOfRangeException(nameof(w));

            int count;
            if (d != 0) count = checked(w * h * d * c);
            else if (c != 0) count = checked(w * h * c);
            else if (h != 0) count = checked(w * h);
            else if (w != 0) count = w;
            else count = 1;

            var values = ReadNcnnArrayAsFloat32(count, loadType);
            var captureBlockSize = loadType == 0 && h > 0 && d == 0 && c == 0
                ? ResolveCaptureBlockSize(w)
                : 0;
            _q8Capture?.WriteNcnnArray(values, captureBlockSize, forceFp32: loadType == 1);
            return values;
        }

        public override void SkipTensor(int w, int h, int d, int c, int loadType)
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

            if (_q8Reader != null)
            {
                if (_q8Capture == null)
                    _q8Reader.SkipRecord(count);
                else
                {
                    var values = _q8Reader.ReadArray(count, requireRaw: loadType == 1);
                    _q8Capture.WriteNcnnArray(values, forceFp32: loadType == 1);
                }
                return;
            }

            if (_q8Capture != null)
            {
                var values = ReadNcnnArrayAsFloat32(count, loadType);
                _q8Capture.WriteNcnnArray(values, forceFp32: loadType == 1);
                return;
            }

            if (loadType == 1)
            {
                Skip((long)count * 4);
                return;
            }

            if (loadType != 0)
                throw new NotSupportedException("unsupported loadType: " + loadType);

            var flag = _br.ReadUInt32();
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
            if (_q8Reader != null)
            {
                var rawBytes = _q8Reader.ReadRaw(count);
                _q8Capture?.WriteRaw(rawBytes);
                return rawBytes;
            }
            var bytes = _br.ReadBytes(count);
            _q8Capture?.WriteRaw(bytes);
            return bytes;
        }

        public void Skip(long byteCount)
        {
            if (byteCount < 0)
                throw new ArgumentOutOfRangeException(nameof(byteCount));
            if (byteCount == 0)
                return;
            if (byteCount > int.MaxValue) throw new NotSupportedException("A single Q8 raw record cannot exceed Int32.MaxValue bytes.");
            if (_q8Reader != null)
            {
                if (_q8Capture == null)
                    _q8Reader.SkipRecord((int)byteCount);
                else
                    _q8Capture.WriteRaw(_q8Reader.ReadRaw((int)byteCount));
                return;
            }
            if (_q8Capture != null)
            {
                var bytes = ReadBytesChecked((int)byteCount);
                _q8Capture.WriteRaw(bytes);
                return;
            }
            if (_stream.CanSeek)
                _stream.Seek(byteCount, SeekOrigin.Current);
            else
                _br.ReadBytes((int)byteCount);
        }

        public override void Dispose()
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

            if (_q8Reader != null)
                return _q8Reader.ReadArray(count, requireRaw: loadType == 1);

            if (loadType == 1)
                return ReadFloat32ArraySource(count);

            if (loadType != 0)
                throw new NotSupportedException("unsupported loadType: " + loadType);

            var flag = _br.ReadUInt32();
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
                return ReadFloat32ArraySource(count);
            }

            if (sum != 0)
            {
                var table = ReadFloat32ArraySource(256);
                var idxBytes = ReadBytesChecked(Align4(count));
                var a = new float[count];
                for (var i = 0; i < count; i++)
                    a[i] = table[idxBytes[i]];
                return a;
            }

            if (f0 == 0)
                return ReadFloat32ArraySource(count);

            throw new InvalidDataException("unsupported ncnn weight encoding flag: 0x" + flag.ToString("X8"));
        }

        public uint ReadUInt32()
        {
            if (_q8Reader != null)
            {
                var bytes = _q8Reader.ReadRaw(sizeof(uint));
                _q8Capture?.WriteRaw(bytes);
                return BitConverter.ToUInt32(bytes, 0);
            }
            var value = _br.ReadUInt32();
            _q8Capture?.WriteRaw(BitConverter.GetBytes(value));
            return value;
        }

        public NcnnQ8PackedArray ReadQ8NcnnMatPacked(int count, int expectedBlockSize)
        {
            if (_q8Reader == null) throw new InvalidOperationException("Packed Q8 reads require an AIImage Q8 archive.");
            return _q8Reader.ReadPackedArray(count, expectedBlockSize);
        }

        public bool TryReadQ8NcnnMatPacked(int count, int expectedBlockSize, out NcnnQ8PackedArray packed)
        {
            if (_q8Reader == null)
            {
                packed = null;
                return false;
            }
            return _q8Reader.TryReadPackedArray(count, expectedBlockSize, out packed);
        }

        public NcnnQ4PackedArray ReadQ4NcnnMatPacked(int count, int expectedBlockSize)
        {
            if (_q8Reader == null) throw new InvalidOperationException("Packed Q4 reads require an AIImage Q4 archive.");
            return _q8Reader.ReadQ4PackedArray(count, expectedBlockSize);
        }

        public bool TryReadQ4NcnnMatPacked(int count, int expectedBlockSize, out NcnnQ4PackedArray packed)
        {
            if (_q8Reader == null)
            {
                packed = null;
                return false;
            }
            return _q8Reader.TryReadQ4PackedArray(count, expectedBlockSize, out packed);
        }

        public float[] ReadNcnnMatAsFloat32(int w, int h, int d, int c, int loadType)
        {
            return ReadTensorAsFloat32(w, h, d, c, loadType);
        }

        public void SkipNcnnMat(int w, int h, int d, int c, int loadType)
        {
            SkipTensor(w, h, d, c, loadType);
        }

        public override bool TryReadQuantizedTensor(int count, int expectedBlockSize, out AexisQuantizedTensor packed)
        {
            var result = TryReadQ8NcnnMatPacked(count, expectedBlockSize, out var ncnnPacked);
            packed = ncnnPacked;
            return result;
        }

        public override bool TryReadInt4QuantizedTensor(int count, int expectedBlockSize, out AexisQuantizedTensor packed)
        {
            // Q4 Gemm uses group-wise scales. The caller validates that each group
            // stays within an output channel after it has resolved K/N dimensions.
            var result = TryReadQ4NcnnMatPacked(count, 0, out var ncnnPacked);
            packed = ncnnPacked;
            return result;
        }

        private int ResolveCaptureBlockSize(int rowWidth)
        {
            if (_q8Capture == null || _q8Capture.QuantizationBits != 4)
                return rowWidth;

            // Keep quantization groups within a 2D Gemm output row.  The group
            // width comes from the Q4 archive builder; using a hard-coded 64
            // here used to produce 64-wide Gemm records while the generated
            // manifest claimed its requested (for example 16-wide) grouping.
            var preferredGroupSize = _q8Capture.DefaultBlockSize;
            for (var groupSize = Math.Min(preferredGroupSize, rowWidth); groupSize >= 4; groupSize >>= 1)
            {
                if (rowWidth % groupSize == 0)
                    return groupSize;
            }

            return rowWidth;
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

        private float[] ReadFloat32ArraySource(int count)
        {
            var bytes = ReadBytesChecked(checked(count * sizeof(float)));
            var result = new float[count];
            if (BitConverter.IsLittleEndian)
            {
                Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
                return result;
            }
            for (var i = 0; i < count; i++) result[i] = BitConverter.ToSingle(bytes, i * sizeof(float));
            return result;
        }

        private static bool ProbeQ8Archive(Stream stream)
        {
            if (!stream.CanSeek || stream.Length - stream.Position < sizeof(ulong)) return false;
            var position = stream.Position;
            var bytes = new byte[sizeof(ulong)];
            NcnnQ8ArchiveWriter.ReadFully(stream, bytes, 0, bytes.Length);
            stream.Seek(position, SeekOrigin.Begin);
            return BitConverter.ToUInt64(bytes, 0) == NcnnQ8ArchiveWriter.Magic;
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
