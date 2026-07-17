using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace AIImage.Inference.Core
{
    [Serializable]
    public sealed class OnnxModel
    {
        public int opset;
        public OnnxGraph graph = new OnnxGraph();
    }

    [Serializable]
    public sealed class OnnxGraph
    {
        public string name = string.Empty;
        public readonly List<OnnxNode> nodes = new List<OnnxNode>();
        public readonly List<OnnxValueInfo> inputs = new List<OnnxValueInfo>();
        public readonly List<OnnxValueInfo> outputs = new List<OnnxValueInfo>();
        public readonly List<OnnxValueInfo> valueInfos = new List<OnnxValueInfo>();
        public readonly Dictionary<string, OnnxTensor> initializers = new Dictionary<string, OnnxTensor>(StringComparer.Ordinal);

        public int CountNodes(string opType)
        {
            var count = 0;
            if (string.IsNullOrEmpty(opType))
                return count;
            for (var i = 0; i < nodes.Count; i++)
            {
                if (string.Equals(nodes[i]?.opType, opType, StringComparison.Ordinal))
                    count++;
            }
            return count;
        }
    }

    [Serializable]
    public sealed class OnnxNode
    {
        public string name = string.Empty;
        public string opType = string.Empty;
        public string domain = string.Empty;
        public readonly List<string> inputs = new List<string>();
        public readonly List<string> outputs = new List<string>();
        public readonly Dictionary<string, OnnxAttribute> attributes = new Dictionary<string, OnnxAttribute>(StringComparer.Ordinal);
    }

    [Serializable]
    public sealed class OnnxAttribute
    {
        public string name = string.Empty;
        public int type;
        public float f;
        public long i;
        public byte[] s = Array.Empty<byte>();
        public readonly List<float> floats = new List<float>();
        public readonly List<long> ints = new List<long>();
        public readonly List<string> strings = new List<string>();
        public OnnxTensor tensor;

        public string GetUtf8String()
        {
            return s == null || s.Length == 0 ? string.Empty : Encoding.UTF8.GetString(s);
        }
    }

    [Serializable]
    public sealed class OnnxValueInfo
    {
        public string name = string.Empty;
        public TensorDataType dataType = TensorDataType.Unknown;
        public long[] dims = Array.Empty<long>();
    }

    [Serializable]
    public sealed class OnnxTensor
    {
        public string name = string.Empty;
        public TensorDataType dataType = TensorDataType.Unknown;
        public int onnxDataType;
        public long[] dims = Array.Empty<long>();
        public byte[] rawData = Array.Empty<byte>();
        public float[] floatData = Array.Empty<float>();
        public int[] int32Data = Array.Empty<int>();
        public long[] int64Data = Array.Empty<long>();

        public long ElementCount
        {
            get
            {
                if (dims == null || dims.Length == 0)
                    return 1;
                long count = 1;
                for (var i = 0; i < dims.Length; i++)
                {
                    if (dims[i] < 0)
                        return -1;
                    count = checked(count * Math.Max(1, dims[i]));
                }
                return count;
            }
        }

        public byte[] GetFloat32LittleEndianBytes()
        {
            if (dataType != TensorDataType.Float32)
                throw new InvalidDataException("ONNX tensor is not float32: " + name);
            var count = ElementCount;
            if (count < 0 || count > int.MaxValue / sizeof(float))
                throw new InvalidDataException("Invalid float32 tensor element count: " + name);
            var byteCount = checked((int)count * sizeof(float));
            if (rawData != null && rawData.Length == byteCount)
                return rawData;
            if (floatData != null && floatData.Length == count)
            {
                var bytes = new byte[byteCount];
                Buffer.BlockCopy(floatData, 0, bytes, 0, bytes.Length);
                if (!BitConverter.IsLittleEndian)
                {
                    for (var i = 0; i < bytes.Length; i += 4)
                    {
                        var b0 = bytes[i + 0];
                        bytes[i + 0] = bytes[i + 3];
                        bytes[i + 3] = b0;
                        var b1 = bytes[i + 1];
                        bytes[i + 1] = bytes[i + 2];
                        bytes[i + 2] = b1;
                    }
                }
                return bytes;
            }
            throw new InvalidDataException("Float32 tensor has no usable data: " + name);
        }
    }

    // Minimal ONNX protobuf reader for import planning and model lowering.
    // It does not execute ONNX and intentionally keeps unsupported protobuf
    // constructs as parse errors instead of silently guessing.
    public static class OnnxModelReader
    {
        public static OnnxModel Read(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("An ONNX path is required.", nameof(path));
            return Read(File.ReadAllBytes(path));
        }

        public static OnnxModel Read(byte[] modelBytes)
        {
            if (modelBytes == null || modelBytes.Length == 0)
                throw new ArgumentException("ONNX model bytes are required.", nameof(modelBytes));

            var model = new OnnxModel();
            var reader = new ProtoReader(modelBytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 7 && wire == 2)
                    model.graph = ParseGraph(reader.ReadBytes());
                else if (field == 8 && wire == 2)
                    ParseOpset(reader.ReadBytes(), model);
                else
                    reader.Skip(wire);
            }

            if (model.graph == null || (model.graph.nodes.Count == 0 && model.graph.initializers.Count == 0))
                throw new InvalidDataException("ONNX ModelProto has no GraphProto.");
            return model;
        }

        private static void ParseOpset(byte[] bytes, OnnxModel model)
        {
            var reader = new ProtoReader(bytes);
            var domain = string.Empty;
            var version = 0;
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2)
                    domain = reader.ReadString();
                else if (field == 2 && wire == 0)
                    version = (int)reader.ReadVarint();
                else
                    reader.Skip(wire);
            }

            if (string.IsNullOrEmpty(domain) || domain == "ai.onnx")
                model.opset = version;
        }

        private static OnnxGraph ParseGraph(byte[] bytes)
        {
            var graph = new OnnxGraph();
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2)
                    graph.nodes.Add(ParseNode(reader.ReadBytes()));
                else if (field == 2 && wire == 2)
                    graph.name = reader.ReadString();
                else if (field == 5 && wire == 2)
                {
                    var tensor = ParseTensor(reader.ReadBytes());
                    if (!string.IsNullOrEmpty(tensor.name))
                        graph.initializers[tensor.name] = tensor;
                }
                else if (field == 11 && wire == 2)
                    graph.inputs.Add(ParseValueInfo(reader.ReadBytes()));
                else if (field == 12 && wire == 2)
                    graph.outputs.Add(ParseValueInfo(reader.ReadBytes()));
                else if (field == 13 && wire == 2)
                    graph.valueInfos.Add(ParseValueInfo(reader.ReadBytes()));
                else
                    reader.Skip(wire);
            }
            return graph;
        }

        private static OnnxNode ParseNode(byte[] bytes)
        {
            var node = new OnnxNode();
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2)
                    node.inputs.Add(reader.ReadString());
                else if (field == 2 && wire == 2)
                    node.outputs.Add(reader.ReadString());
                else if (field == 3 && wire == 2)
                    node.name = reader.ReadString();
                else if (field == 4 && wire == 2)
                    node.opType = reader.ReadString();
                else if (field == 5 && wire == 2)
                {
                    var attr = ParseAttribute(reader.ReadBytes());
                    if (!string.IsNullOrEmpty(attr.name))
                        node.attributes[attr.name] = attr;
                }
                else if (field == 7 && wire == 2)
                    node.domain = reader.ReadString();
                else
                    reader.Skip(wire);
            }
            return node;
        }

        private static OnnxAttribute ParseAttribute(byte[] bytes)
        {
            var attr = new OnnxAttribute();
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2)
                    attr.name = reader.ReadString();
                else if (field == 2 && wire == 5)
                    attr.f = reader.ReadFloat32();
                else if (field == 3 && wire == 0)
                    attr.i = reader.ReadInt64Varint();
                else if (field == 4 && wire == 2)
                    attr.s = reader.ReadBytes();
                else if (field == 5 && wire == 2)
                    attr.tensor = ParseTensor(reader.ReadBytes());
                else if (field == 7)
                    ReadFloatField(reader, wire, attr.floats);
                else if (field == 8)
                    ReadInt64Field(reader, wire, attr.ints);
                else if (field == 9 && wire == 2)
                    attr.strings.Add(reader.ReadString());
                else if (field == 20 && wire == 0)
                    attr.type = (int)reader.ReadVarint();
                else
                    reader.Skip(wire);
            }
            return attr;
        }

        private static OnnxTensor ParseTensor(byte[] bytes)
        {
            var tensor = new OnnxTensor();
            var floats = new List<float>();
            var int32s = new List<int>();
            var int64s = new List<long>();
            var dims = new List<long>();
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1)
                    ReadInt64Field(reader, wire, dims);
                else if (field == 2 && wire == 0)
                {
                    tensor.onnxDataType = (int)reader.ReadVarint();
                    tensor.dataType = ToType(tensor.onnxDataType);
                }
                else if (field == 4)
                    ReadFloatField(reader, wire, floats);
                else if (field == 5)
                    ReadInt32Field(reader, wire, int32s);
                else if (field == 7)
                    ReadInt64Field(reader, wire, int64s);
                else if (field == 8 && wire == 2)
                    tensor.name = reader.ReadString();
                else if (field == 9 && wire == 2)
                    tensor.rawData = reader.ReadBytes();
                else
                    reader.Skip(wire);
            }

            tensor.dims = dims.ToArray();
            tensor.floatData = floats.ToArray();
            tensor.int32Data = int32s.ToArray();
            tensor.int64Data = int64s.ToArray();
            return tensor;
        }

        private static OnnxValueInfo ParseValueInfo(byte[] bytes)
        {
            var value = new OnnxValueInfo();
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2)
                    value.name = reader.ReadString();
                else if (field == 2 && wire == 2)
                    ParseType(reader.ReadBytes(), value);
                else
                    reader.Skip(wire);
            }
            return value;
        }

        private static void ParseType(byte[] bytes, OnnxValueInfo value)
        {
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2)
                    ParseTensorType(reader.ReadBytes(), value);
                else
                    reader.Skip(wire);
            }
        }

        private static void ParseTensorType(byte[] bytes, OnnxValueInfo value)
        {
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 0)
                    value.dataType = ToType((int)reader.ReadVarint());
                else if (field == 2 && wire == 2)
                    value.dims = ParseShape(reader.ReadBytes());
                else
                    reader.Skip(wire);
            }
        }

        private static long[] ParseShape(byte[] bytes)
        {
            var dims = new List<long>();
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2)
                    dims.Add(ParseDimension(reader.ReadBytes()));
                else
                    reader.Skip(wire);
            }
            return dims.ToArray();
        }

        private static long ParseDimension(byte[] bytes)
        {
            var reader = new ProtoReader(bytes);
            long value = -1;
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 0)
                    value = reader.ReadInt64Varint();
                else if (field == 2 && wire == 2)
                {
                    reader.ReadString();
                    if (value < 0)
                        value = -1;
                }
                else
                    reader.Skip(wire);
            }
            return value;
        }

        private static void ReadFloatField(ProtoReader reader, int wire, List<float> dst)
        {
            if (wire == 5)
            {
                dst.Add(reader.ReadFloat32());
                return;
            }
            if (wire == 2)
            {
                var packed = new ProtoReader(reader.ReadBytes());
                while (!packed.Eof)
                    dst.Add(packed.ReadFloat32());
                return;
            }
            reader.Skip(wire);
        }

        private static void ReadInt32Field(ProtoReader reader, int wire, List<int> dst)
        {
            if (wire == 0)
            {
                dst.Add((int)reader.ReadInt64Varint());
                return;
            }
            if (wire == 2)
            {
                var packed = new ProtoReader(reader.ReadBytes());
                while (!packed.Eof)
                    dst.Add((int)packed.ReadInt64Varint());
                return;
            }
            reader.Skip(wire);
        }

        private static void ReadInt64Field(ProtoReader reader, int wire, List<long> dst)
        {
            if (wire == 0)
            {
                dst.Add(reader.ReadInt64Varint());
                return;
            }
            if (wire == 2)
            {
                var packed = new ProtoReader(reader.ReadBytes());
                while (!packed.Eof)
                    dst.Add(packed.ReadInt64Varint());
                return;
            }
            reader.Skip(wire);
        }

        private static TensorDataType ToType(int onnxType)
        {
            switch (onnxType)
            {
                case 1: return TensorDataType.Float32;
                case 10: return TensorDataType.Float16;
                case 3: return TensorDataType.Int8;
                case 2: return TensorDataType.UInt8;
                case 6: return TensorDataType.Int32;
                default: return TensorDataType.Unknown;
            }
        }

        private sealed class ProtoReader
        {
            private readonly byte[] data;
            private int position;

            public ProtoReader(byte[] data)
            {
                this.data = data ?? throw new ArgumentNullException(nameof(data));
            }

            public bool Eof => position >= data.Length;

            public bool TryRead(out int field, out int wire)
            {
                if (position >= data.Length)
                {
                    field = 0;
                    wire = 0;
                    return false;
                }
                var key = ReadVarint();
                field = (int)(key >> 3);
                wire = (int)(key & 7);
                if (field == 0)
                    throw new InvalidDataException("Invalid protobuf field 0.");
                return true;
            }

            public ulong ReadVarint()
            {
                ulong value = 0;
                for (var shift = 0; shift < 64; shift += 7)
                {
                    if (position >= data.Length)
                        throw new InvalidDataException("Truncated protobuf varint.");
                    var b = data[position++];
                    value |= (ulong)(b & 0x7F) << shift;
                    if ((b & 0x80) == 0)
                        return value;
                }
                throw new InvalidDataException("Invalid protobuf varint.");
            }

            public long ReadInt64Varint()
            {
                return unchecked((long)ReadVarint());
            }

            public float ReadFloat32()
            {
                if (position + 4 > data.Length)
                    throw new InvalidDataException("Truncated protobuf fixed32.");
                var value = BitConverter.ToSingle(data, position);
                position += 4;
                return value;
            }

            public byte[] ReadBytes()
            {
                var length = (int)ReadVarint();
                if (length < 0 || position + length > data.Length)
                    throw new InvalidDataException("Invalid protobuf length.");
                var value = new byte[length];
                Buffer.BlockCopy(data, position, value, 0, length);
                position += length;
                return value;
            }

            public string ReadString()
            {
                return Encoding.UTF8.GetString(ReadBytes());
            }

            public void Skip(int wire)
            {
                switch (wire)
                {
                    case 0:
                        ReadVarint();
                        break;
                    case 1:
                        SkipBytes(8);
                        break;
                    case 2:
                        SkipBytes((int)ReadVarint());
                        break;
                    case 5:
                        SkipBytes(4);
                        break;
                    default:
                        throw new InvalidDataException("Unsupported protobuf wire type " + wire + ".");
                }
            }

            private void SkipBytes(int count)
            {
                if (count < 0 || position + count > data.Length)
                    throw new InvalidDataException("Truncated protobuf field.");
                position += count;
            }
        }
    }
}
