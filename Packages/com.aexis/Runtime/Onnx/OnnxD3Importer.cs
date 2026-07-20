using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Aexis;

namespace Aexis.Onnx
{
    [Serializable]
    public sealed class OnnxD3ImportOptions
    {
        // Capacities are physical texture capacities, never a CPU observation of GPU data.
        public readonly Dictionary<string, int> outputCapacities = new Dictionary<string, int>(StringComparer.Ordinal);
        public readonly HashSet<string> provablyUniqueScatterNodes = new HashSet<string>(StringComparer.Ordinal);
    }

    [Serializable]
    public sealed class OnnxD3ImportResult
    {
        public int opset;
        public CanonicalShapeIndexNode[] nodes = Array.Empty<CanonicalShapeIndexNode>();
        public OnnxExecutionPlanDiagnostic[] diagnostics = Array.Empty<OnnxExecutionPlanDiagnostic>();
        public bool IsPlanEligible => diagnostics.Length == 0;
    }

    // Self-contained ONNX protobuf reader for the D3 shape/index subset. It intentionally
    // reads only the schema fields required by import planning and never executes ONNX.
    public static class OnnxD3Importer
    {
        public static OnnxD3ImportResult Import(string path, OnnxD3ImportOptions options = null)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("An ONNX path is required.", nameof(path));
            return Import(File.ReadAllBytes(path), options);
        }

        public static OnnxD3ImportResult Import(byte[] modelBytes, OnnxD3ImportOptions options = null)
        {
            if (modelBytes == null || modelBytes.Length == 0) throw new ArgumentException("ONNX model bytes are required.", nameof(modelBytes));
            options = options ?? new OnnxD3ImportOptions();
            var model = ParseModel(modelBytes);
            var shapes = new Dictionary<string, TensorInfo>(model.tensors, StringComparer.Ordinal);
            var nodes = new List<CanonicalShapeIndexNode>();
            var diagnostics = new List<OnnxExecutionPlanDiagnostic>();

            foreach (var source in model.nodes)
            {
                if (!IsD3Operator(source.opType)) continue;
                var imported = BuildImportedNode(source, shapes, options);
                if (!OnnxExecutionAdapter.TryAdapt(imported, out var canonical, out var diagnostic))
                {
                    diagnostics.Add(WithNode(diagnostic, source.name, source.opType));
                    continue;
                }
                nodes.Add(canonical);
                foreach (var output in source.outputs) shapes[output] = new TensorInfo { rank = OutputRank(source, imported), dataType = OutputDataType(source) };
            }
            return new OnnxD3ImportResult { opset = model.opset, nodes = nodes.ToArray(), diagnostics = diagnostics.ToArray() };
        }

        private static OnnxExecutionImportedNode BuildImportedNode(NodeInfo node, Dictionary<string, TensorInfo> tensors, OnnxD3ImportOptions options)
        {
            var data = node.inputs.Count > 0 && tensors.TryGetValue(node.inputs[0], out var info) ? info : new TensorInfo { rank = 1, dataType = TensorDataType.Unknown };
            var parameterInput = node.opType == "TopK" || node.opType == "OneHot" ? 1 : -1;
            var parameterDynamic = parameterInput >= 0 && (node.inputs.Count <= parameterInput || !tensors.TryGetValue(node.inputs[parameterInput], out var parameter) || !parameter.isInitializer);
            var output = node.outputs.Count > 0 ? node.outputs[0] : node.name;
            options.outputCapacities.TryGetValue(output, out var capacity);
            return new OnnxExecutionImportedNode
            {
                name = string.IsNullOrEmpty(node.name) ? node.opType + "_" + output : node.name,
                opType = node.opType,
                inputs = node.inputs.ToArray(), outputs = node.outputs.ToArray(), inputRank = data.rank,
                axis = node.attributes.TryGetValue("axis", out var axis) ? (int)axis : 0,
                batchDims = node.attributes.TryGetValue("batch_dims", out var batch) ? (int)batch : 0,
                indexDataType = IndexDataType(node, tensors),
                parameterComesFromTensor = parameterDynamic,
                uniqueIndices = options.provablyUniqueScatterNodes.Contains(node.name),
                scatterConflictPolicy = "reject", outputCapacity = capacity
            };
        }

        private static TensorDataType IndexDataType(NodeInfo node, Dictionary<string, TensorInfo> tensors)
        {
            var input = node.opType == "GatherND" ? 1 : (node.opType.StartsWith("Scatter", StringComparison.Ordinal) ? 1 : 0);
            return node.inputs.Count > input && tensors.TryGetValue(node.inputs[input], out var value) ? value.dataType : TensorDataType.Int32;
        }
        private static int OutputRank(NodeInfo node, OnnxExecutionImportedNode imported) => node.opType == "NonZero" ? 2 : (node.opType == "OneHot" ? imported.inputRank + 1 : imported.inputRank);
        private static TensorDataType OutputDataType(NodeInfo node) => node.opType == "NonZero" ? TensorDataType.Int32 : TensorDataType.Float32;
        private static bool IsD3Operator(string op) => op == "Shape" || op == "Size" || op == "Rank" || op == "TopK" || op == "OneHot" || op == "NonZero" || op == "Compress" || op == "GatherND" || op == "Scatter" || op == "ScatterElements" || op == "ScatterND";
        private static OnnxExecutionPlanDiagnostic WithNode(OnnxExecutionPlanDiagnostic diagnostic, string name, string op) => diagnostic ?? new OnnxExecutionPlanDiagnostic { code = "import-failed", message = op + " node " + name + " could not be imported.", recommendedAction = "Check the D3 importer contract." };

        private static ModelInfo ParseModel(byte[] bytes)
        {
            var model = new ModelInfo(); var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 7 && wire == 2) ParseGraph(reader.ReadBytes(), model);
                else if (field == 8 && wire == 2) ParseOpset(reader.ReadBytes(), model);
                else reader.Skip(wire);
            }
            if (model.nodes.Count == 0 && model.tensors.Count == 0) throw new InvalidDataException("ONNX ModelProto has no GraphProto.");
            return model;
        }
        private static void ParseOpset(byte[] bytes, ModelInfo model)
        {
            var reader = new ProtoReader(bytes); string domain = string.Empty; int version = 0;
            while (reader.TryRead(out var field, out var wire)) { if (field == 1 && wire == 2) domain = reader.ReadString(); else if (field == 2 && wire == 0) version = (int)reader.ReadVarint(); else reader.Skip(wire); }
            if (string.IsNullOrEmpty(domain) || domain == "ai.onnx") model.opset = version;
        }
        private static void ParseGraph(byte[] bytes, ModelInfo model)
        {
            var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2) model.nodes.Add(ParseNode(reader.ReadBytes()));
                else if ((field == 5 || field == 11 || field == 12 || field == 13) && wire == 2)
                {
                    var tensor = field == 5 ? ParseInitializer(reader.ReadBytes()) : ParseValueInfo(reader.ReadBytes());
                    if (string.IsNullOrEmpty(tensor.name)) continue;
                    if (model.tensors.TryGetValue(tensor.name, out var existing) && existing.isInitializer && !tensor.isInitializer)
                    {
                        existing.rank = tensor.rank > 0 ? tensor.rank : existing.rank;
                        existing.dataType = tensor.dataType != TensorDataType.Unknown ? tensor.dataType : existing.dataType;
                    }
                    else model.tensors[tensor.name] = tensor;
                }
                else reader.Skip(wire);
            }
        }
        private static NodeInfo ParseNode(byte[] bytes)
        {
            var node = new NodeInfo(); var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 2) node.inputs.Add(reader.ReadString());
                else if (field == 2 && wire == 2) node.outputs.Add(reader.ReadString());
                else if (field == 3 && wire == 2) node.name = reader.ReadString();
                else if (field == 4 && wire == 2) node.opType = reader.ReadString();
                else if (field == 5 && wire == 2) ParseAttribute(reader.ReadBytes(), node.attributes);
                else reader.Skip(wire);
            }
            return node;
        }
        private static void ParseAttribute(byte[] bytes, Dictionary<string, long> attributes)
        {
            var reader = new ProtoReader(bytes); string name = null; long value = 0; var hasValue = false;
            while (reader.TryRead(out var field, out var wire)) { if (field == 1 && wire == 2) name = reader.ReadString(); else if (field == 3 && wire == 0) { value = (long)reader.ReadVarint(); hasValue = true; } else reader.Skip(wire); }
            if (!string.IsNullOrEmpty(name) && hasValue) attributes[name] = value;
        }
        private static TensorInfo ParseInitializer(byte[] bytes)
        {
            var tensor = new TensorInfo { isInitializer = true }; var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire))
            {
                if (field == 1 && wire == 0) { reader.ReadVarint(); tensor.rank++; }
                else if (field == 1 && wire == 2) { var dimensions = new ProtoReader(reader.ReadBytes()); while (dimensions.TryReadPackedVarint()) tensor.rank++; }
                else if (field == 2 && wire == 0) tensor.dataType = ToType((int)reader.ReadVarint());
                else if (field == 8 && wire == 2) tensor.name = reader.ReadString();
                else reader.Skip(wire);
            }
            return tensor;
        }
        private static TensorInfo ParseValueInfo(byte[] bytes)
        {
            var tensor = new TensorInfo(); var reader = new ProtoReader(bytes);
            while (reader.TryRead(out var field, out var wire)) { if (field == 1 && wire == 2) tensor.name = reader.ReadString(); else if (field == 2 && wire == 2) ParseType(reader.ReadBytes(), tensor); else reader.Skip(wire); }
            return tensor;
        }
        private static void ParseType(byte[] bytes, TensorInfo tensor)
        {
            var reader = new ProtoReader(bytes); while (reader.TryRead(out var field, out var wire)) { if (field == 1 && wire == 2) ParseTensorType(reader.ReadBytes(), tensor); else reader.Skip(wire); }
        }
        private static void ParseTensorType(byte[] bytes, TensorInfo tensor)
        {
            var reader = new ProtoReader(bytes); while (reader.TryRead(out var field, out var wire)) { if (field == 1 && wire == 0) tensor.dataType = ToType((int)reader.ReadVarint()); else if (field == 2 && wire == 2) ParseShape(reader.ReadBytes(), tensor); else reader.Skip(wire); }
        }
        private static void ParseShape(byte[] bytes, TensorInfo tensor)
        {
            var reader = new ProtoReader(bytes); while (reader.TryRead(out var field, out var wire)) { if (field == 1 && wire == 2) { reader.ReadBytes(); tensor.rank++; } else reader.Skip(wire); }
        }
        private static TensorDataType ToType(int onnx) { switch (onnx) { case 1: return TensorDataType.Float32; case 10: return TensorDataType.Float16; case 3: return TensorDataType.Int8; case 2: return TensorDataType.UInt8; case 6: return TensorDataType.Int32; default: return TensorDataType.Unknown; } }

        private sealed class ModelInfo { public int opset; public readonly List<NodeInfo> nodes = new List<NodeInfo>(); public readonly Dictionary<string, TensorInfo> tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal); }
        private sealed class NodeInfo { public string name = string.Empty; public string opType = string.Empty; public readonly List<string> inputs = new List<string>(); public readonly List<string> outputs = new List<string>(); public readonly Dictionary<string, long> attributes = new Dictionary<string, long>(StringComparer.Ordinal); }
        private sealed class TensorInfo { public string name = string.Empty; public int rank; public TensorDataType dataType; public bool isInitializer; }

        private sealed class ProtoReader
        {
            private readonly byte[] data; private int position;
            public ProtoReader(byte[] data) { this.data = data ?? throw new ArgumentNullException(nameof(data)); }
            public bool TryRead(out int field, out int wire) { if (position >= data.Length) { field = wire = 0; return false; } var key = ReadVarint(); field = (int)(key >> 3); wire = (int)(key & 7); if (field == 0) throw new InvalidDataException("Invalid protobuf field 0."); return true; }
            public ulong ReadVarint() { ulong value = 0; for (var shift = 0; shift < 64; shift += 7) { if (position >= data.Length) throw new InvalidDataException("Truncated protobuf varint."); var b = data[position++]; value |= (ulong)(b & 127) << shift; if ((b & 128) == 0) return value; } throw new InvalidDataException("Invalid protobuf varint."); }
            public byte[] ReadBytes() { var length = (int)ReadVarint(); if (length < 0 || position + length > data.Length) throw new InvalidDataException("Invalid protobuf length."); var value = new byte[length]; Buffer.BlockCopy(data, position, value, 0, length); position += length; return value; }
            public string ReadString() => Encoding.UTF8.GetString(ReadBytes());
            public bool TryReadPackedVarint() { if (position >= data.Length) return false; ReadVarint(); return true; }
            public void Skip(int wire) { switch (wire) { case 0: ReadVarint(); break; case 1: SkipBytes(8); break; case 2: SkipBytes((int)ReadVarint()); break; case 5: SkipBytes(4); break; default: throw new InvalidDataException("Unsupported protobuf wire type " + wire + "."); } }
            private void SkipBytes(int count) { if (count < 0 || position + count > data.Length) throw new InvalidDataException("Truncated protobuf field."); position += count; }
        }
    }
}
