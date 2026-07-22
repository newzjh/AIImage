using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Aexis.Execution;

namespace Aexis.Ncnn
{
    // The text .param format remains accepted for compatibility. This format makes
    // graph parsing deterministic at import time and is embedded in AEXM archives.
    public static class AexisNcnnBinaryParam
    {
        public const uint Magic = 0x31504241; // "ABP1" little-endian
        public const int Version = 1;
        private const int MaxCollectionLength = 1_000_000;
        private const int MaxStringLength = 16 * 1024 * 1024;

        public static byte[] Serialize(AexisGraphModel graph)
        {
            using (var stream = new MemoryStream())
            {
                Write(stream, graph);
                return stream.ToArray();
            }
        }

        public static AexisGraphModel Deserialize(byte[] bytes)
        {
            if (bytes == null)
                throw new ArgumentNullException(nameof(bytes));
            using (var stream = new MemoryStream(bytes, writable: false))
                return Read(stream);
        }

        public static void Write(Stream stream, AexisGraphModel graph)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (graph == null)
                throw new ArgumentNullException(nameof(graph));

            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                writer.Write(Magic);
                writer.Write(Version);
                WriteString(writer, graph.magic);
                writer.Write(graph.layerCount);
                writer.Write(graph.blobCount);
                var layers = graph.layers ?? new List<AexisGraphModel.Layer>();
                writer.Write(layers.Count);
                foreach (var layer in layers)
                {
                    if (layer == null)
                        throw new InvalidDataException("Aexis graph cannot serialize a null layer.");
                    WriteString(writer, layer.typeName);
                    WriteString(writer, layer.name);
                    WriteStrings(writer, layer.bottomNames);
                    WriteStrings(writer, layer.topNames);
                    WriteParameters(writer, layer.intParams);
                    WriteNamedParameters(writer, layer.stringParams);
                }

                var declarations = graph.extensionDeclarations ?? Array.Empty<AexisModelExtensionDeclaration>();
                writer.Write(declarations.Length);
                foreach (var declaration in declarations)
                {
                    if (declaration == null)
                        throw new InvalidDataException("Aexis graph cannot serialize a null extension declaration.");
                    WriteString(writer, declaration.typeName);
                    writer.Write(declaration.schemaVersion);
                    WriteString(writer, declaration.kernelId);
                    writer.Write(declaration.textureNativeRequired);
                }
            }
        }

        public static AexisGraphModel Read(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
            {
                if (reader.ReadUInt32() != Magic)
                    throw new InvalidDataException("Not an Aexis binary param stream.");
                var version = reader.ReadInt32();
                if (version != Version)
                    throw new InvalidDataException("Unsupported Aexis binary param version: " + version.ToString(CultureInfo.InvariantCulture));

                var graph = new AexisGraphModel
                {
                    magic = ReadString(reader),
                    layerCount = reader.ReadInt32(),
                    blobCount = reader.ReadInt32()
                };
                var layerCount = ReadCount(reader, "layer");
                for (var index = 0; index < layerCount; index++)
                {
                    var typeName = ReadString(reader);
                    var layer = new AexisGraphModel.Layer
                    {
                        typeName = typeName,
                        type = AexisLayerTypeKey.FromString(typeName),
                        name = ReadString(reader),
                        bottomNames = ReadStrings(reader, "bottom"),
                        topNames = ReadStrings(reader, "top"),
                        intParams = ReadParameters(reader),
                        stringParams = ReadNamedParameters(reader)
                    };
                    layer.bottoms = layer.bottomNames.Length;
                    layer.tops = layer.topNames.Length;
                    graph.layers.Add(layer);
                }

                var declarationCount = ReadCount(reader, "extension declaration");
                var declarations = new AexisModelExtensionDeclaration[declarationCount];
                for (var index = 0; index < declarationCount; index++)
                {
                    declarations[index] = new AexisModelExtensionDeclaration
                    {
                        typeName = ReadString(reader),
                        schemaVersion = reader.ReadInt32(),
                        kernelId = ReadString(reader),
                        textureNativeRequired = reader.ReadBoolean()
                    };
                }
                graph.extensionDeclarations = declarations;
                if (stream.Position != stream.Length)
                    throw new InvalidDataException("Aexis binary param contains trailing bytes.");
                return graph;
            }
        }

        private static void WriteStrings(BinaryWriter writer, string[] values)
        {
            var source = values ?? Array.Empty<string>();
            writer.Write(source.Length);
            foreach (var value in source)
                WriteString(writer, value);
        }

        private static string[] ReadStrings(BinaryReader reader, string kind)
        {
            var count = ReadCount(reader, kind);
            var values = new string[count];
            for (var index = 0; index < count; index++)
                values[index] = ReadString(reader);
            return values;
        }

        private static void WriteParameters(BinaryWriter writer, Dictionary<int, string> parameters)
        {
            var source = parameters ?? new Dictionary<int, string>();
            writer.Write(source.Count);
            foreach (var pair in source.OrderBy(pair => pair.Key))
            {
                writer.Write(pair.Key);
                WriteString(writer, pair.Value);
            }
        }

        private static Dictionary<int, string> ReadParameters(BinaryReader reader)
        {
            var count = ReadCount(reader, "integer parameter");
            var result = new Dictionary<int, string>(count);
            for (var index = 0; index < count; index++)
            {
                var key = reader.ReadInt32();
                if (result.ContainsKey(key))
                    throw new InvalidDataException("Aexis binary param contains a duplicate integer parameter.");
                result.Add(key, ReadString(reader));
            }
            return result;
        }

        private static void WriteNamedParameters(BinaryWriter writer, Dictionary<string, string> parameters)
        {
            var source = parameters ?? new Dictionary<string, string>(StringComparer.Ordinal);
            writer.Write(source.Count);
            foreach (var pair in source.OrderBy(pair => pair.Key, StringComparer.Ordinal))
            {
                WriteString(writer, pair.Key);
                WriteString(writer, pair.Value);
            }
        }

        private static Dictionary<string, string> ReadNamedParameters(BinaryReader reader)
        {
            var count = ReadCount(reader, "named parameter");
            var result = new Dictionary<string, string>(count, StringComparer.Ordinal);
            for (var index = 0; index < count; index++)
            {
                var key = ReadString(reader);
                if (string.IsNullOrWhiteSpace(key) || result.ContainsKey(key))
                    throw new InvalidDataException("Aexis binary param contains an invalid or duplicate named parameter.");
                result.Add(key, ReadString(reader));
            }
            return result;
        }

        private static void WriteString(BinaryWriter writer, string value)
        {
            var bytes = Encoding.UTF8.GetBytes(value ?? string.Empty);
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }

        private static string ReadString(BinaryReader reader)
        {
            var length = reader.ReadInt32();
            if (length < 0 || length > MaxStringLength)
                throw new InvalidDataException("Aexis binary param contains an invalid string length.");
            var bytes = reader.ReadBytes(length);
            if (bytes.Length != length)
                throw new EndOfStreamException("Aexis binary param ended while reading a string.");
            return Encoding.UTF8.GetString(bytes);
        }

        private static int ReadCount(BinaryReader reader, string kind)
        {
            var count = reader.ReadInt32();
            if (count < 0 || count > MaxCollectionLength)
                throw new InvalidDataException("Aexis binary param has an invalid " + kind + " count.");
            return count;
        }
    }

    [Serializable]
    public sealed class AexisCompiledModel
    {
        public const uint Magic = 0x314D5841; // "AXM1" little-endian
        public const int FormatVersion = 1;

        public string modelId = string.Empty;
        public string sourceFormat = string.Empty;
        public string compilerVersion = "aexis-model/v1";
        public bool eligible;
        public byte[] binaryParam = Array.Empty<byte>();
        public byte[] weights = Array.Empty<byte>();
        public byte[] source = Array.Empty<byte>();
        public string manifestJson = string.Empty;
        public string diagnosticJson = string.Empty;

        public AexisGraphModel ReadGraph()
        {
            return AexisNcnnBinaryParam.Deserialize(binaryParam ?? Array.Empty<byte>());
        }
    }

    public static class AexisModelArchive
    {
        private const int MaxSectionLength = 1024 * 1024 * 1024;

        public static byte[] Serialize(AexisCompiledModel model)
        {
            using (var stream = new MemoryStream())
            {
                Write(stream, model);
                return stream.ToArray();
            }
        }

        public static AexisCompiledModel Deserialize(byte[] bytes)
        {
            if (bytes == null)
                throw new ArgumentNullException(nameof(bytes));
            using (var stream = new MemoryStream(bytes, writable: false))
                return Read(stream);
        }

        public static void Write(Stream stream, AexisCompiledModel model)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (model.binaryParam == null || model.binaryParam.Length == 0)
                throw new InvalidDataException("An Aexis compiled model requires binaryParam.");

            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                writer.Write(AexisCompiledModel.Magic);
                writer.Write(AexisCompiledModel.FormatVersion);
                WriteString(writer, model.modelId);
                WriteString(writer, model.sourceFormat);
                WriteString(writer, model.compilerVersion);
                writer.Write(model.eligible);
                WriteBytes(writer, model.binaryParam);
                WriteBytes(writer, model.weights);
                WriteBytes(writer, model.source);
                WriteString(writer, model.manifestJson);
                WriteString(writer, model.diagnosticJson);
            }
        }

        public static AexisCompiledModel Read(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
            {
                if (reader.ReadUInt32() != AexisCompiledModel.Magic)
                    throw new InvalidDataException("Not an Aexis compiled model archive.");
                var version = reader.ReadInt32();
                if (version != AexisCompiledModel.FormatVersion)
                    throw new InvalidDataException("Unsupported Aexis model archive version: " + version.ToString(CultureInfo.InvariantCulture));
                var model = new AexisCompiledModel
                {
                    modelId = ReadString(reader),
                    sourceFormat = ReadString(reader),
                    compilerVersion = ReadString(reader),
                    eligible = reader.ReadBoolean(),
                    binaryParam = ReadBytes(reader),
                    weights = ReadBytes(reader),
                    source = ReadBytes(reader),
                    manifestJson = ReadString(reader),
                    diagnosticJson = ReadString(reader)
                };
                if (model.binaryParam.Length == 0)
                    throw new InvalidDataException("Aexis model archive contains no graph.");
                if (stream.Position != stream.Length)
                    throw new InvalidDataException("Aexis model archive contains trailing bytes.");
                return model;
            }
        }

        private static void WriteBytes(BinaryWriter writer, byte[] bytes)
        {
            var data = bytes ?? Array.Empty<byte>();
            writer.Write(data.Length);
            writer.Write(data);
        }

        private static byte[] ReadBytes(BinaryReader reader)
        {
            var length = reader.ReadInt32();
            if (length < 0 || length > MaxSectionLength)
                throw new InvalidDataException("Aexis model archive contains an invalid section length.");
            var data = reader.ReadBytes(length);
            if (data.Length != length)
                throw new EndOfStreamException("Aexis model archive ended while reading a section.");
            return data;
        }

        private static void WriteString(BinaryWriter writer, string value)
        {
            var bytes = Encoding.UTF8.GetBytes(value ?? string.Empty);
            WriteBytes(writer, bytes);
        }

        private static string ReadString(BinaryReader reader)
        {
            return Encoding.UTF8.GetString(ReadBytes(reader));
        }
    }
}
