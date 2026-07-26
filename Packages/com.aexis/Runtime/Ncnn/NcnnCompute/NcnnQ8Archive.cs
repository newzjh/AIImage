using System;
using System.IO;
using Aexis.Execution;
using UnityEngine;

namespace Aexis.Ncnn
{
    public sealed class NcnnQ8PackedArray : AexisQuantizedTensor
    {
        internal NcnnQ8PackedArray(int elementCount, int blockSize, uint[] packedValues, float[] scales)
            : base(elementCount, blockSize, packedValues, scales)
        {
        }
    }

    public sealed class NcnnQ4PackedArray : AexisQuantizedTensor
    {
        internal NcnnQ4PackedArray(int elementCount, int blockSize, uint[] packedValues, float[] scales)
            : base(elementCount, blockSize, packedValues, scales)
        {
        }
    }

    public sealed class NcnnQ8ArchiveWriter : IDisposable
    {
        internal const ulong Magic = 0x0038544547535141UL; // "AQSGE T8" as a little-endian binary marker.
        internal const uint Version = 1;
        internal const uint RecordMagic = 0x31523851; // Q8R1
        internal const int HeaderBytes = 32;
        internal const int RecordHeaderBytes = 24;
        internal const int RecordQ8 = 1;
        internal const int RecordRaw = 2;
        internal const int RecordQ4 = 3;

        private readonly Stream _stream;
        private readonly BinaryWriter _writer;
        private readonly int _defaultBlockSize;
        private readonly int _fp32Threshold;
        private readonly int _quantizationBits;
        private int _recordCount;
        private bool _disposed;

        public NcnnQ8ArchiveWriter(
            Stream stream,
            long sourceBytes,
            int defaultBlockSize = 256,
            int fp32Threshold = 4096,
            int quantizationBits = 8)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            if (!stream.CanWrite || !stream.CanSeek) throw new ArgumentException("Q8 archive output must be writable and seekable.", nameof(stream));
            if (defaultBlockSize <= 0) throw new ArgumentOutOfRangeException(nameof(defaultBlockSize));
            if (fp32Threshold < 0) throw new ArgumentOutOfRangeException(nameof(fp32Threshold));
            if (quantizationBits != 4 && quantizationBits != 8) throw new ArgumentOutOfRangeException(nameof(quantizationBits));
            _stream = stream;
            _writer = new BinaryWriter(stream);
            _defaultBlockSize = defaultBlockSize;
            _fp32Threshold = fp32Threshold;
            _quantizationBits = quantizationBits;
            _writer.Write(Magic);
            _writer.Write(Version);
            _writer.Write((uint)quantizationBits);
            _writer.Write(defaultBlockSize);
            _writer.Write(0);
            _writer.Write(sourceBytes);
        }

        public int RecordCount => _recordCount;
        public int QuantizationBits => _quantizationBits;
        /// <summary>
        /// Default quantization block width requested by the archive producer.
        /// Readers that know a tensor row width may reduce this value to keep a
        /// block inside a row, but must never silently replace it with a wider
        /// grouping.  That would make the precision manifest lie about the
        /// stored model and materially change Q4 accuracy.
        /// </summary>
        public int DefaultBlockSize => _defaultBlockSize;

        public void WriteNcnnArray(float[] values, int blockSize = 0, bool forceFp32 = false)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (forceFp32 || values.Length <= _fp32Threshold)
            {
                WriteFloat32Array(values);
                return;
            }

            if (_quantizationBits == 4)
                WriteQ4(values, blockSize > 0 ? blockSize : _defaultBlockSize);
            else
                WriteQ8(values, blockSize > 0 ? blockSize : _defaultBlockSize);
        }

        public void WriteQ8(float[] values, int blockSize)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
            WriteRecordHeader(RecordQ8, values.Length, blockSize, checked(values.Length + BlockCount(values.Length, blockSize) * sizeof(float)));
            var block = new byte[Math.Min(blockSize, Math.Max(1, values.Length))];
            for (var offset = 0; offset < values.Length; offset += blockSize)
            {
                var length = Math.Min(blockSize, values.Length - offset);
                var maxAbs = 0f;
                for (var i = 0; i < length; i++) maxAbs = Mathf.Max(maxAbs, Mathf.Abs(values[offset + i]));
                var scale = maxAbs > 0f ? maxAbs / 127f : 1f;
                _writer.Write(scale);
                for (var i = 0; i < length; i++)
                {
                    var quantized = Mathf.Clamp(Mathf.RoundToInt(values[offset + i] / scale), -127, 127);
                    block[i] = unchecked((byte)(sbyte)quantized);
                }
                _writer.Write(block, 0, length);
            }
        }

        public void WriteQ4(float[] values, int blockSize)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
            WriteRecordHeader(RecordQ4, values.Length, blockSize, Q4PayloadBytes(values.Length, blockSize));
            var block = new byte[Math.Max(1, (Math.Min(blockSize, Math.Max(1, values.Length)) + 1) / 2)];
            for (var offset = 0; offset < values.Length; offset += blockSize)
            {
                var length = Math.Min(blockSize, values.Length - offset);
                var maxAbs = 0f;
                for (var i = 0; i < length; i++) maxAbs = Mathf.Max(maxAbs, Mathf.Abs(values[offset + i]));
                var scale = maxAbs > 0f ? maxAbs / 7f : 1f;
                _writer.Write(scale);
                Array.Clear(block, 0, block.Length);
                for (var i = 0; i < length; i++)
                {
                    var quantized = Mathf.Clamp(Mathf.RoundToInt(values[offset + i] / scale), -7, 7);
                    var byteIndex = i >> 1;
                    var shift = (i & 1) * 4;
                    block[byteIndex] |= (byte)(((uint)quantized & 0xfu) << shift);
                }
                _writer.Write(block, 0, (length + 1) / 2);
            }
        }

        public void WriteQ8FromFloat32Rows(BinaryReader source, int rowCount, int rowSize)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (rowCount <= 0 || rowSize <= 0) throw new ArgumentOutOfRangeException(nameof(rowCount));
            var count = checked(rowCount * rowSize);
            WriteRecordHeader(RecordQ8, count, rowSize, checked(count + rowCount * sizeof(float)));
            var byteRow = new byte[checked(rowSize * sizeof(float))];
            var values = new float[rowSize];
            var quantized = new byte[rowSize];
            for (var row = 0; row < rowCount; row++)
            {
                ReadFully(source.BaseStream, byteRow, 0, byteRow.Length);
                Buffer.BlockCopy(byteRow, 0, values, 0, byteRow.Length);
                var maxAbs = 0f;
                for (var i = 0; i < rowSize; i++) maxAbs = Mathf.Max(maxAbs, Mathf.Abs(values[i]));
                var scale = maxAbs > 0f ? maxAbs / 127f : 1f;
                _writer.Write(scale);
                for (var i = 0; i < rowSize; i++)
                {
                    var q = Mathf.Clamp(Mathf.RoundToInt(values[i] / scale), -127, 127);
                    quantized[i] = unchecked((byte)(sbyte)q);
                }
                _writer.Write(quantized);
            }
        }

        public void WriteQ4FromQ8Rows(NcnnQ8PackedArray source, int rowCount, int rowSize, int groupSize)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (rowCount <= 0 || rowSize <= 0 || source.ElementCount != checked(rowCount * rowSize))
                throw new ArgumentException("Q8 row source shape is invalid.", nameof(source));
            if (source.BlockSize != rowSize || source.Scales.Length != rowCount)
                throw new InvalidDataException("Q8 row source must contain one scale per embedding row.");
            if (groupSize <= 0 || rowSize % groupSize != 0)
                throw new ArgumentException("Q4 embedding group size must divide the embedding row width.", nameof(groupSize));

            var count = source.ElementCount;
            WriteRecordHeader(RecordQ4, count, groupSize, Q4PayloadBytes(count, groupSize));
            var packedGroup = new byte[(groupSize + 1) / 2];
            for (var row = 0; row < rowCount; row++)
            {
                var rowStart = row * rowSize;
                for (var groupStart = 0; groupStart < rowSize; groupStart += groupSize)
                {
                    var maxAbs = 0;
                    for (var column = 0; column < groupSize; column++)
                    {
                        var index = rowStart + groupStart + column;
                        var packed = source.PackedValues[index >> 2];
                        var value = unchecked((sbyte)((packed >> ((index & 3) * 8)) & 0xffu));
                        maxAbs = Math.Max(maxAbs, Math.Abs((int)value));
                    }
                    var scale = maxAbs > 0 ? source.Scales[row] * maxAbs / 7f : 1f;
                    _writer.Write(scale);
                    Array.Clear(packedGroup, 0, packedGroup.Length);
                    for (var column = 0; column < groupSize; column++)
                    {
                        var index = rowStart + groupStart + column;
                        var sourcePacked = source.PackedValues[index >> 2];
                        var q8 = unchecked((sbyte)((sourcePacked >> ((index & 3) * 8)) & 0xffu));
                        var q4 = maxAbs > 0 ? Mathf.Clamp(Mathf.RoundToInt(q8 * 7f / maxAbs), -7, 7) : 0;
                        packedGroup[column >> 1] |= (byte)(((uint)q4 & 0xfu) << ((column & 1) * 4));
                    }
                    _writer.Write(packedGroup);
                }
            }
        }

        public void WriteRaw(byte[] bytes)
        {
            if (bytes == null) throw new ArgumentNullException(nameof(bytes));
            WriteRecordHeader(RecordRaw, bytes.Length, 1, bytes.Length);
            _writer.Write(bytes);
        }

        public void WriteFloat32Array(float[] values)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            var bytes = new byte[checked(values.Length * sizeof(float))];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            WriteRecordHeader(RecordRaw, values.Length, 1, bytes.Length);
            _writer.Write(bytes);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            var end = _stream.Position;
            _stream.Seek(20, SeekOrigin.Begin);
            _writer.Write(_recordCount);
            _stream.Seek(end, SeekOrigin.Begin);
            _writer.Flush();
        }

        private void WriteRecordHeader(int kind, int elementCount, int blockSize, int payloadBytes)
        {
            _writer.Write(RecordMagic);
            _writer.Write(kind);
            _writer.Write(elementCount);
            _writer.Write(blockSize);
            _writer.Write(payloadBytes);
            _writer.Write(0);
            _recordCount++;
        }

        private static int BlockCount(int count, int blockSize) => count == 0 ? 0 : checked((count + blockSize - 1) / blockSize);

        private static int Q4PayloadBytes(int count, int blockSize)
        {
            var bytes = 0;
            for (var offset = 0; offset < count; offset += blockSize)
                bytes = checked(bytes + sizeof(float) + (Math.Min(blockSize, count - offset) + 1) / 2);
            return bytes;
        }

        internal static void ReadFully(Stream stream, byte[] bytes, int offset, int count)
        {
            while (count > 0)
            {
                var read = stream.Read(bytes, offset, count);
                if (read <= 0) throw new EndOfStreamException("Unexpected end of Q8 archive source.");
                offset += read;
                count -= read;
            }
        }
    }

    internal sealed class NcnnQ8ArchiveReader
    {
        private readonly Stream _stream;
        private readonly BinaryReader _reader;
        private readonly int _recordCount;
        private readonly int _quantizationBits;
        private int _recordsRead;

        public NcnnQ8ArchiveReader(Stream stream)
        {
            _stream = stream ?? throw new ArgumentNullException(nameof(stream));
            _reader = new BinaryReader(stream);
            var magic = _reader.ReadUInt64();
            if (magic != NcnnQ8ArchiveWriter.Magic) throw new InvalidDataException("Invalid AIImage Q8 archive magic.");
            var version = _reader.ReadUInt32();
            if (version != NcnnQ8ArchiveWriter.Version) throw new InvalidDataException("Unsupported AIImage Q8 archive version: " + version);
            _quantizationBits = (int)_reader.ReadUInt32();
            var defaultBlockSize = _reader.ReadInt32();
            _recordCount = _reader.ReadInt32();
            _reader.ReadInt64();
            if (defaultBlockSize <= 0 || _recordCount < 0 || (_quantizationBits != 0 && _quantizationBits != 4 && _quantizationBits != 8))
                throw new InvalidDataException("Invalid AIImage quantized archive header.");
        }

        public int QuantizationBits => _quantizationBits == 0 ? 8 : _quantizationBits;

        public float[] ReadArray(int expectedCount, bool requireRaw)
        {
            var header = ReadHeader(expectedCount);
            if (header.Kind == NcnnQ8ArchiveWriter.RecordRaw)
            {
                if (header.PayloadBytes != checked(expectedCount * sizeof(float)))
                    throw new InvalidDataException("Q8 raw float record size mismatch.");
                var bytes = ReadBytesChecked(header.PayloadBytes);
                var values = new float[expectedCount];
                Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
                return values;
            }
            if (requireRaw) throw new InvalidDataException("Expected a raw FP32 quantized archive record.");
            if (header.Kind != NcnnQ8ArchiveWriter.RecordQ8 && header.Kind != NcnnQ8ArchiveWriter.RecordQ4)
                throw new InvalidDataException("Expected a quantized array record.");
            var result = new float[expectedCount];
            var bytesConsumed = 0;
            var block = new byte[Math.Min(header.BlockSize, Math.Max(1, expectedCount))];
            for (var offset = 0; offset < expectedCount; offset += header.BlockSize)
            {
                var length = Math.Min(header.BlockSize, expectedCount - offset);
                var scale = _reader.ReadSingle();
                bytesConsumed += sizeof(float);
                if (header.Kind == NcnnQ8ArchiveWriter.RecordQ8)
                {
                    NcnnQ8ArchiveWriter.ReadFully(_stream, block, 0, length);
                    bytesConsumed += length;
                    for (var i = 0; i < length; i++) result[offset + i] = unchecked((sbyte)block[i]) * scale;
                }
                else
                {
                    var packedBytes = (length + 1) / 2;
                    NcnnQ8ArchiveWriter.ReadFully(_stream, block, 0, packedBytes);
                    bytesConsumed += packedBytes;
                    for (var i = 0; i < length; i++)
                    {
                        var raw = (block[i >> 1] >> ((i & 1) * 4)) & 0xf;
                        var signed = raw >= 8 ? raw - 16 : raw;
                        result[offset + i] = signed * scale;
                    }
                }
            }
            ValidatePayload(header, bytesConsumed);
            return result;
        }

        public NcnnQ8PackedArray ReadPackedArray(int expectedCount, int expectedBlockSize)
        {
            var header = ReadHeader(expectedCount);
            if (header.Kind != NcnnQ8ArchiveWriter.RecordQ8) throw new InvalidDataException("Expected a Q8 packed array record.");
            if (expectedBlockSize > 0 && header.BlockSize != expectedBlockSize)
                throw new InvalidDataException("Q8 block size mismatch. expected=" + expectedBlockSize + " actual=" + header.BlockSize);
            var packed = new uint[Math.Max(1, checked((expectedCount + 3) / 4))];
            var scales = new float[BlockCount(expectedCount, header.BlockSize)];
            var block = new byte[Math.Min(header.BlockSize, Math.Max(1, expectedCount))];
            var bytesConsumed = 0;
            for (var blockIndex = 0; blockIndex < scales.Length; blockIndex++)
            {
                var offset = blockIndex * header.BlockSize;
                var length = Math.Min(header.BlockSize, expectedCount - offset);
                scales[blockIndex] = _reader.ReadSingle();
                bytesConsumed += sizeof(float);
                NcnnQ8ArchiveWriter.ReadFully(_stream, block, 0, length);
                bytesConsumed += length;
                for (var i = 0; i < length; i++)
                {
                    var index = offset + i;
                    packed[index >> 2] |= (uint)block[i] << ((index & 3) * 8);
                }
            }
            ValidatePayload(header, bytesConsumed);
            return new NcnnQ8PackedArray(expectedCount, header.BlockSize, packed, scales);
        }

        public bool TryReadPackedArray(int expectedCount, int expectedBlockSize, out NcnnQ8PackedArray packed)
        {
            var position = _stream.Position;
            var recordsRead = _recordsRead;
            var header = ReadHeader(expectedCount);
            _stream.Seek(position, SeekOrigin.Begin);
            _recordsRead = recordsRead;
            if (header.Kind != NcnnQ8ArchiveWriter.RecordQ8)
            {
                packed = null;
                return false;
            }
            packed = ReadPackedArray(expectedCount, expectedBlockSize);
            return true;
        }

        public NcnnQ4PackedArray ReadQ4PackedArray(int expectedCount, int expectedBlockSize)
        {
            var header = ReadHeader(expectedCount);
            if (header.Kind != NcnnQ8ArchiveWriter.RecordQ4) throw new InvalidDataException("Expected a Q4 packed array record.");
            if (expectedBlockSize > 0 && header.BlockSize != expectedBlockSize)
                throw new InvalidDataException("Q4 block size mismatch. expected=" + expectedBlockSize + " actual=" + header.BlockSize);
            var packed = new uint[Math.Max(1, checked((expectedCount + 7) / 8))];
            var scales = new float[BlockCount(expectedCount, header.BlockSize)];
            var block = new byte[(Math.Min(header.BlockSize, Math.Max(1, expectedCount)) + 1) / 2];
            var bytesConsumed = 0;
            for (var blockIndex = 0; blockIndex < scales.Length; blockIndex++)
            {
                var offset = blockIndex * header.BlockSize;
                var length = Math.Min(header.BlockSize, expectedCount - offset);
                scales[blockIndex] = _reader.ReadSingle();
                bytesConsumed += sizeof(float);
                var packedBytes = (length + 1) / 2;
                NcnnQ8ArchiveWriter.ReadFully(_stream, block, 0, packedBytes);
                bytesConsumed += packedBytes;
                for (var i = 0; i < length; i++)
                {
                    var index = offset + i;
                    var raw = (uint)((block[i >> 1] >> ((i & 1) * 4)) & 0xf);
                    packed[index >> 3] |= raw << ((index & 7) * 4);
                }
            }
            ValidatePayload(header, bytesConsumed);
            return new NcnnQ4PackedArray(expectedCount, header.BlockSize, packed, scales);
        }

        public bool TryReadQ4PackedArray(int expectedCount, int expectedBlockSize, out NcnnQ4PackedArray packed)
        {
            var position = _stream.Position;
            var recordsRead = _recordsRead;
            var header = ReadHeader(expectedCount);
            _stream.Seek(position, SeekOrigin.Begin);
            _recordsRead = recordsRead;
            if (header.Kind != NcnnQ8ArchiveWriter.RecordQ4)
            {
                packed = null;
                return false;
            }
            packed = ReadQ4PackedArray(expectedCount, expectedBlockSize);
            return true;
        }

        public byte[] ReadRaw(int expectedBytes)
        {
            var header = ReadHeader(expectedBytes);
            if (header.Kind != NcnnQ8ArchiveWriter.RecordRaw || header.PayloadBytes != expectedBytes)
                throw new InvalidDataException("Q8 raw record mismatch.");
            return ReadBytesChecked(expectedBytes);
        }

        public void SkipRecord(int expectedCount)
        {
            var header = ReadHeader(expectedCount);
            SkipExactly(header.PayloadBytes);
        }

        private RecordHeader ReadHeader(int expectedCount)
        {
            if (_recordsRead >= _recordCount) throw new EndOfStreamException("Q8 archive has no remaining records.");
            var magic = _reader.ReadUInt32();
            if (magic != NcnnQ8ArchiveWriter.RecordMagic) throw new InvalidDataException("Invalid Q8 record magic at " + (_stream.Position - sizeof(uint)) + ".");
            var header = new RecordHeader
            {
                Kind = _reader.ReadInt32(),
                ElementCount = _reader.ReadInt32(),
                BlockSize = _reader.ReadInt32(),
                PayloadBytes = _reader.ReadInt32()
            };
            _reader.ReadInt32();
            _recordsRead++;
            if (header.ElementCount != expectedCount)
                throw new InvalidDataException("Q8 record element count mismatch. expected=" + expectedCount + " actual=" + header.ElementCount + " record=" + _recordsRead + ".");
            if (header.BlockSize <= 0 || header.PayloadBytes < 0) throw new InvalidDataException("Invalid Q8 record header.");
            return header;
        }

        private void ValidatePayload(RecordHeader header, int consumed)
        {
            if (consumed != header.PayloadBytes)
                throw new InvalidDataException("Q8 payload size mismatch. expected=" + header.PayloadBytes + " actual=" + consumed + ".");
        }

        private byte[] ReadBytesChecked(int count)
        {
            var result = new byte[count];
            NcnnQ8ArchiveWriter.ReadFully(_stream, result, 0, count);
            return result;
        }

        private void SkipExactly(int count)
        {
            if (_stream.CanSeek)
            {
                _stream.Seek(count, SeekOrigin.Current);
                return;
            }
            var scratch = new byte[Math.Min(8192, Math.Max(1, count))];
            while (count > 0)
            {
                var read = Math.Min(count, scratch.Length);
                NcnnQ8ArchiveWriter.ReadFully(_stream, scratch, 0, read);
                count -= read;
            }
        }

        private static int BlockCount(int count, int blockSize) => count == 0 ? 0 : checked((count + blockSize - 1) / blockSize);

        private struct RecordHeader
        {
            public int Kind;
            public int ElementCount;
            public int BlockSize;
            public int PayloadBytes;
        }
    }
}
