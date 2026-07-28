using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Aexis;

namespace Aexis.Onnx
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
        // SparseTensorProto is normalized to an immutable dense initializer at
        // import time. Keep the source record for validation/reporting while
        // execution consumes only the one texture-native immutable upload.
        public readonly Dictionary<string, OnnxSparseTensor> sparseInitializers = new Dictionary<string, OnnxSparseTensor>(StringComparer.Ordinal);

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
        // Graph-valued attributes are used by ONNX control-flow operators. Keep
        // them in the parsed model so strict lowering can prove and inline a
        // bounded graph before any GPU execution plan is created.
        public OnnxGraph graph;
        public readonly List<OnnxGraph> graphs = new List<OnnxGraph>();

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
        // Preserve the source ONNX element type so lowering can distinguish
        // INT64 from the narrowed Aexis Int32 logical texture contract.
        public int onnxDataType;
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
        public int dataLocation;
        public readonly Dictionary<string, string> externalData = new Dictionary<string, string>(StringComparer.Ordinal);

        public bool UsesExternalData => dataLocation == 1 || externalData.Count != 0;

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
                    count = checked(count * dims[i]);
                }
                return count;
            }
        }

        public bool TryValidatePayload(out string reason)
        {
            long count;
            try
            {
                count = ElementCount;
            }
            catch (OverflowException)
            {
                reason = "element count overflows Int64";
                return false;
            }
            if (count < 0)
            {
                reason = "shape contains a dynamic or negative extent";
                return false;
            }
            if (count > int.MaxValue)
            {
                reason = "element count exceeds the managed importer limit";
                return false;
            }

            var elementCount = (int)count;
            if (rawData != null && rawData.Length != 0)
            {
                if (!TryGetElementByteWidth(onnxDataType, out var width))
                {
                    reason = "raw payload uses unsupported ONNX elem_type " + onnxDataType.ToString(CultureInfo.InvariantCulture);
                    return false;
                }
                var expected = checked((long)elementCount * width);
                if (rawData.LongLength != expected)
                {
                    reason = "raw payload length is " + rawData.LongLength.ToString(CultureInfo.InvariantCulture)
                        + " bytes; expected " + expected.ToString(CultureInfo.InvariantCulture);
                    return false;
                }
                reason = null;
                return true;
            }

            if (elementCount == 0)
            {
                reason = null;
                return true;
            }

            switch (onnxDataType)
            {
                case 1:
                    if (floatData != null && floatData.Length == elementCount) { reason = null; return true; }
                    reason = "float_data length does not match the tensor shape";
                    return false;
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 9:
                case 10:
                case 12:
                    if (int32Data != null && int32Data.Length == elementCount) { reason = null; return true; }
                    reason = "int32_data length does not match the tensor shape";
                    return false;
                case 7:
                case 11:
                case 13:
                    if (int64Data != null && int64Data.Length == elementCount) { reason = null; return true; }
                    reason = "int64_data length does not match the tensor shape";
                    return false;
                default:
                    reason = "ONNX elem_type " + onnxDataType.ToString(CultureInfo.InvariantCulture) + " is not supported by the payload validator";
                    return false;
            }
        }

        internal static bool TryGetElementByteWidth(int onnxType, out int width)
        {
            switch (onnxType)
            {
                case 2:
                case 3:
                case 9:
                    width = 1;
                    return true;
                case 10:
                case 16:
                    width = 2;
                    return true;
                case 1:
                case 4:
                case 5:
                case 6:
                case 12:
                    width = 4;
                    return true;
                case 7:
                case 11:
                case 13:
                    width = 8;
                    return true;
                default:
                    width = 0;
                    return false;
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

    [Serializable]
    public sealed class OnnxSparseTensor
    {
        // This is an import/upload bound, never an activation allocation. It
        // prevents a malformed sparse initializer from silently requesting an
        // unbounded CPU staging allocation before the immutable texture upload.
        public const int MaximumDenseElementCount = 16 * 1024 * 1024;

        public OnnxTensor values;
        public OnnxTensor indices;
        public long[] dims = Array.Empty<long>();

        public bool TryMaterializeDense(out OnnxTensor dense, out string reason)
        {
            dense = null;
            if (values == null || indices == null)
            {
                reason = "values and indices tensors are required";
                return false;
            }
            if (string.IsNullOrWhiteSpace(values.name))
            {
                reason = "values tensor has no initializer name";
                return false;
            }
            if (dims == null || dims.Length < 1 || dims.Length > 4)
            {
                reason = "dense shape rank must be in [1,4]";
                return false;
            }

            long denseCount = 1;
            try
            {
                for (var axis = 0; axis < dims.Length; axis++)
                {
                    if (dims[axis] <= 0)
                    {
                        reason = "dense shape contains a non-positive extent";
                        return false;
                    }
                    denseCount = checked(denseCount * dims[axis]);
                }
            }
            catch (OverflowException)
            {
                reason = "dense shape element count overflows Int64";
                return false;
            }
            if (denseCount > MaximumDenseElementCount)
            {
                reason = "dense shape exceeds the immutable sparse upload capacity of "
                    + MaximumDenseElementCount.ToString(CultureInfo.InvariantCulture) + " elements";
                return false;
            }
            if (!values.TryValidatePayload(out reason))
                return false;
            if (!indices.TryValidatePayload(out reason))
                return false;
            if (values.dataType != TensorDataType.Float32 && values.dataType != TensorDataType.Int32)
            {
                reason = "values elem_type must be Float32, Int32, Int64, or Bool for the immutable texture upload";
                return false;
            }
            if (indices.onnxDataType != 7)
            {
                reason = "indices elem_type must be ONNX INT64";
                return false;
            }

            var nnz = values.ElementCount;
            if (nnz < 0 || nnz > int.MaxValue)
            {
                reason = "values count is outside the importer range";
                return false;
            }
            if (!TryGetPayloadBytes(values, out var valueBytes, out var elementWidth, out reason))
                return false;
            if (!TryValidateIndexShape(indices, (int)nnz, dims.Length, out var coordinateIndices, out reason))
                return false;

            byte[] denseBytes;
            try
            {
                denseBytes = new byte[checked((int)denseCount * elementWidth)];
            }
            catch (OverflowException)
            {
                reason = "dense immutable upload byte count overflows Int32";
                return false;
            }

            var visited = new HashSet<long>();
            for (var item = 0; item < (int)nnz; item++)
            {
                if (!TryResolveDenseIndex(indices, item, dims, coordinateIndices, out var denseIndex, out reason))
                    return false;
                if (!visited.Add(denseIndex))
                {
                    reason = "indices contain a duplicate dense coordinate; sparse write ordering is not a strict immutable contract";
                    return false;
                }
                Buffer.BlockCopy(valueBytes, item * elementWidth, denseBytes, checked((int)denseIndex * elementWidth), elementWidth);
            }

            dense = new OnnxTensor
            {
                name = values.name,
                dataType = values.dataType,
                onnxDataType = values.onnxDataType,
                dims = (long[])dims.Clone(),
                rawData = denseBytes
            };
            reason = null;
            return true;
        }

        private static bool TryValidateIndexShape(OnnxTensor tensor, int nnz, int rank, out bool coordinateIndices, out string reason)
        {
            coordinateIndices = false;
            if (tensor.dims == null)
            {
                reason = "indices shape is missing";
                return false;
            }
            if (tensor.dims.Length == 1 && tensor.dims[0] == nnz)
            {
                reason = null;
                return true;
            }
            if (tensor.dims.Length == 2 && tensor.dims[0] == nnz && tensor.dims[1] == rank)
            {
                coordinateIndices = true;
                reason = null;
                return true;
            }
            reason = "indices shape must be [nnz] or [nnz,rank]";
            return false;
        }

        private static bool TryResolveDenseIndex(
            OnnxTensor tensor,
            int item,
            long[] denseShape,
            bool coordinateIndices,
            out long denseIndex,
            out string reason)
        {
            denseIndex = 0;
            if (!coordinateIndices)
            {
                if (!TryGetInt64(tensor, item, out denseIndex))
                {
                    reason = "indices payload is not a readable INT64 tensor";
                    return false;
                }
                long denseCount = 1;
                for (var axis = 0; axis < denseShape.Length; axis++) denseCount = checked(denseCount * denseShape[axis]);
                if (denseIndex < 0 || denseIndex >= denseCount)
                {
                    reason = "linear index is outside the dense shape";
                    return false;
                }
                reason = null;
                return true;
            }

            try
            {
                for (var axis = 0; axis < denseShape.Length; axis++)
                {
                    if (!TryGetInt64(tensor, item * denseShape.Length + axis, out var coordinate))
                    {
                        reason = "indices payload is not a readable INT64 tensor";
                        return false;
                    }
                    if (coordinate < 0 || coordinate >= denseShape[axis])
                    {
                        reason = "coordinate index is outside the dense shape";
                        return false;
                    }
                    denseIndex = checked(denseIndex * denseShape[axis] + coordinate);
                }
            }
            catch (OverflowException)
            {
                reason = "coordinate index calculation overflows Int64";
                return false;
            }
            reason = null;
            return true;
        }

        private static bool TryGetInt64(OnnxTensor tensor, int index, out long value)
        {
            value = 0;
            if (tensor?.int64Data != null && index >= 0 && index < tensor.int64Data.Length)
            {
                value = tensor.int64Data[index];
                return true;
            }
            if (tensor?.rawData == null || index < 0 || tensor.rawData.Length != checked((int)tensor.ElementCount * sizeof(long)))
                return false;
            value = BitConverter.ToInt64(tensor.rawData, index * sizeof(long));
            return true;
        }

        private static bool TryGetPayloadBytes(OnnxTensor tensor, out byte[] bytes, out int elementWidth, out string reason)
        {
            bytes = null;
            elementWidth = 0;
            reason = null;
            if (!OnnxTensor.TryGetElementByteWidth(tensor.onnxDataType, out elementWidth))
            {
                reason = "values elem_type has no fixed byte width";
                return false;
            }
            var count = tensor.ElementCount;
            if (count < 0 || count > int.MaxValue || checked(count * elementWidth) > int.MaxValue)
            {
                reason = "values payload is outside the importer byte range";
                return false;
            }
            var byteCount = checked((int)count * elementWidth);
            if (tensor.rawData != null && tensor.rawData.Length == byteCount)
            {
                bytes = (byte[])tensor.rawData.Clone();
                return true;
            }
            bytes = new byte[byteCount];
            if (tensor.onnxDataType == 1 && tensor.floatData != null && tensor.floatData.Length == count)
            {
                Buffer.BlockCopy(tensor.floatData, 0, bytes, 0, byteCount);
                return true;
            }
            if (tensor.onnxDataType == 6 && tensor.int32Data != null && tensor.int32Data.Length == count)
            {
                Buffer.BlockCopy(tensor.int32Data, 0, bytes, 0, byteCount);
                return true;
            }
            if (tensor.onnxDataType == 7 && tensor.int64Data != null && tensor.int64Data.Length == count)
            {
                Buffer.BlockCopy(tensor.int64Data, 0, bytes, 0, byteCount);
                return true;
            }
            if (tensor.onnxDataType == 9 && tensor.int32Data != null && tensor.int32Data.Length == count)
            {
                for (var i = 0; i < tensor.int32Data.Length; i++) bytes[i] = tensor.int32Data[i] == 0 ? (byte)0 : (byte)1;
                return true;
            }
            reason = "values payload has no decoded representation";
            return false;
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
            var fullPath = Path.GetFullPath(path);
            var model = Parse(File.ReadAllBytes(fullPath));
            ResolveExternalData(model, fullPath);
            ValidateTensorPayloads(model);
            return model;
        }

        public static OnnxModel Read(byte[] modelBytes)
        {
            if (modelBytes == null || modelBytes.Length == 0)
                throw new ArgumentException("ONNX model bytes are required.", nameof(modelBytes));

            var model = Parse(modelBytes);
            foreach (var tensor in EnumerateTensors(model))
            {
                if (tensor.UsesExternalData)
                    throw new InvalidDataException("ONNX tensor " + TensorName(tensor) + " uses external_data. Import from a model path so relative payload files can be resolved safely.");
            }
            ValidateTensorPayloads(model);
            return model;
        }

        private static OnnxModel Parse(byte[] modelBytes)
        {
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

        private static void ResolveExternalData(OnnxModel model, string modelPath)
        {
            var root = Path.GetFullPath(Path.GetDirectoryName(modelPath) ?? string.Empty);
            var rootPrefix = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            foreach (var tensor in EnumerateTensors(model))
            {
                if (!tensor.UsesExternalData)
                    continue;
                if (!tensor.externalData.TryGetValue("location", out var location) || string.IsNullOrWhiteSpace(location))
                    throw new InvalidDataException("ONNX tensor " + TensorName(tensor) + " declares external_data without a location.");
                if (Path.IsPathRooted(location))
                    throw new InvalidDataException("ONNX tensor " + TensorName(tensor) + " uses an absolute external_data location, which is not allowed: " + location);

                var payloadPath = Path.GetFullPath(Path.Combine(root, location.Replace('/', Path.DirectorySeparatorChar)));
                if (!payloadPath.StartsWith(rootPrefix, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("ONNX tensor " + TensorName(tensor) + " external_data escapes the model directory: " + location);
                if (!File.Exists(payloadPath))
                    throw new FileNotFoundException("External ONNX tensor payload was not found for " + TensorName(tensor) + ": " + payloadPath, payloadPath);

                var offset = ParseExternalRange(tensor, "offset", 0L);
                long requestedLength;
                if (tensor.externalData.ContainsKey("length"))
                {
                    requestedLength = ParseExternalRange(tensor, "length", -1L);
                }
                else
                {
                    if (!OnnxTensor.TryGetElementByteWidth(tensor.onnxDataType, out var width))
                        throw new InvalidDataException("Cannot infer external_data length for tensor " + TensorName(tensor) + " with ONNX elem_type " + tensor.onnxDataType.ToString(CultureInfo.InvariantCulture) + ".");
                    requestedLength = checked(tensor.ElementCount * width);
                }
                if (requestedLength < 0 || requestedLength > int.MaxValue)
                    throw new InvalidDataException("External ONNX tensor payload length is outside the supported range for " + TensorName(tensor) + ".");

                using var stream = new FileStream(payloadPath, FileMode.Open, FileAccess.Read, FileShare.Read);
                if (offset < 0 || offset > stream.Length || requestedLength > stream.Length - offset)
                    throw new EndOfStreamException("External ONNX tensor payload is truncated for " + TensorName(tensor)
                        + ": offset=" + offset.ToString(CultureInfo.InvariantCulture)
                        + " length=" + requestedLength.ToString(CultureInfo.InvariantCulture)
                        + " fileLength=" + stream.Length.ToString(CultureInfo.InvariantCulture) + ".");
                stream.Position = offset;
                tensor.rawData = new byte[(int)requestedLength];
                var read = 0;
                while (read < tensor.rawData.Length)
                {
                    var current = stream.Read(tensor.rawData, read, tensor.rawData.Length - read);
                    if (current <= 0)
                        throw new EndOfStreamException("External ONNX tensor payload ended while reading " + TensorName(tensor) + ".");
                    read += current;
                }
            }
        }

        private static long ParseExternalRange(OnnxTensor tensor, string key, long defaultValue)
        {
            if (!tensor.externalData.TryGetValue(key, out var text) || string.IsNullOrWhiteSpace(text))
                return defaultValue;
            if (!long.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var value) || value < 0)
                throw new InvalidDataException("ONNX tensor " + TensorName(tensor) + " has invalid external_data " + key + "=" + text + ".");
            return value;
        }

        private static void ValidateTensorPayloads(OnnxModel model)
        {
            foreach (var tensor in EnumerateTensors(model))
            {
                if (!tensor.TryValidatePayload(out var reason))
                    throw new InvalidDataException("ONNX tensor " + TensorName(tensor) + " has an invalid payload: " + reason + ".");
            }
        }

        private static IEnumerable<OnnxTensor> EnumerateTensors(OnnxModel model)
        {
            if (model?.graph == null)
                yield break;
            foreach (var tensor in EnumerateGraphTensors(model.graph))
                yield return tensor;
        }

        private static IEnumerable<OnnxTensor> EnumerateGraphTensors(OnnxGraph graph)
        {
            if (graph == null)
                yield break;
            foreach (var pair in graph.initializers)
                if (pair.Value != null)
                    yield return pair.Value;
            foreach (var pair in graph.sparseInitializers)
            {
                if (pair.Value?.values != null)
                    yield return pair.Value.values;
                if (pair.Value?.indices != null)
                    yield return pair.Value.indices;
            }
            foreach (var node in graph.nodes)
                if (node != null)
                    foreach (var pair in node.attributes)
                    {
                        var attribute = pair.Value;
                        if (attribute?.tensor != null)
                            yield return attribute.tensor;
                        if (attribute?.graph != null)
                            foreach (var tensor in EnumerateGraphTensors(attribute.graph))
                                yield return tensor;
                        if (attribute?.graphs != null)
                            foreach (var nestedGraph in attribute.graphs)
                                foreach (var tensor in EnumerateGraphTensors(nestedGraph))
                                    yield return tensor;
                    }
        }

        private static string TensorName(OnnxTensor tensor)
        {
            return string.IsNullOrEmpty(tensor?.name) ? "<unnamed>" : tensor.name;
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
                else if (field == 15 && wire == 2)
                {
                    var sparse = ParseSparseTensor(reader.ReadBytes());
                    if (!sparse.TryMaterializeDense(out var tensor, out var reason))
                        throw new InvalidDataException("ONNX sparse initializer is not a bounded immutable texture upload: " + reason + ".");
                    if (graph.initializers.ContainsKey(tensor.name) || graph.sparseInitializers.ContainsKey(tensor.name))
                        throw new InvalidDataException("ONNX graph has duplicate dense/sparse initializer name: " + tensor.name + ".");
                    graph.sparseInitializers[tensor.name] = sparse;
                    graph.initializers[tensor.name] = tensor;
                }
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
                else if (field == 6 && wire == 2)
                    attr.graph = ParseGraph(reader.ReadBytes());
                else if (field == 7)
                    ReadFloatField(reader, wire, attr.floats);
                else if (field == 8)
                    ReadInt64Field(reader, wire, attr.ints);
                else if (field == 9 && wire == 2)
                    attr.strings.Add(reader.ReadString());
                else if (field == 11 && wire == 2)
                    attr.graphs.Add(ParseGraph(reader.ReadBytes()));
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
                else if (field == 13 && wire == 2)
                {
                    ParseExternalDataEntry(reader.ReadBytes(), out var key, out var value);
                    if (!string.IsNullOrEmpty(key))
                        tensor.externalData[key] = value;
                }
                else if (field == 14 && wire == 0)
                    tensor.dataLocation = (int)reader.ReadVarint();
                else
                    reader.Skip(wire);
            }

            tensor.dims = dims.ToArray();
            tensor.floatData = floats.ToArray();
            tensor.int32Data = int32s.ToArray();
            tensor.int64Data = int64s.ToArray();
            return tensor;
        }

        private static OnnxSparseTensor ParseSparseTensor(byte[] bytes)
        {
            var sparse = new OnnxSparseTensor();
            var dimensions = new List<long>();
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2)
                    sparse.values = ParseTensor(reader.ReadBytes());
                else if (field == 2 && wire == 2)
                    sparse.indices = ParseTensor(reader.ReadBytes());
                else if (field == 3)
                    ReadInt64Field(reader, wire, dimensions);
                else
                    reader.Skip(wire);
            }
            sparse.dims = dimensions.ToArray();
            return sparse;
        }

        private static void ParseExternalDataEntry(byte[] bytes, out string key, out string value)
        {
            key = string.Empty;
            value = string.Empty;
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2)
                    key = reader.ReadString();
                else if (field == 2 && wire == 2)
                    value = reader.ReadString();
                else
                    reader.Skip(wire);
            }
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
                {
                    value.onnxDataType = (int)reader.ReadVarint();
                    value.dataType = ToType(value.onnxDataType);
                }
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
                // Aexis' texture index contract is Int32. ONNX INT64/BOOL constants
                // are range-checked and narrowed by the graph compiler before upload.
                case 7: return TensorDataType.Int32;
                case 9: return TensorDataType.Int32;
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
