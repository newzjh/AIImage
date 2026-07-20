using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using Aexis;
using Aexis.Onnx;

namespace Aexis.Ncnn
{
    [Serializable]
    public sealed class DeepFillV2OnnxNcnnImportReport
    {
        public string status = "unknown";
        public string modelVariant = string.Empty;
        public string onnxPath = string.Empty;
        public string paramTemplatePath = string.Empty;
        public long onnxBytes;
        public int opset;
        public int nodeCount;
        public int initializerCount;
        public int convNodeCount;
        public int ncnnConvLayerCount;
        public int extractImagePatchesNodeCount;
        public int extractPatchesLayerCount;
        public int contextualAttentionNodeCount;
        public int contextualAttentionLayerCount;
        public int generatedBinBytes;
        public string generatedBinSha256 = string.Empty;
    }

    public sealed class DeepFillV2OnnxNcnnImportResult
    {
        public string paramText = string.Empty;
        public byte[] ncnnBinBytes = Array.Empty<byte>();
        public DeepFillV2OnnxNcnnImportReport report = new DeepFillV2OnnxNcnnImportReport();
    }

    // DeepFillV2 HiFill-specific ONNX direct reader/lowerer.  It deliberately
    // does not invoke Sentis or an external converter at runtime.  The compact
    // NCNN param is a 30KB topology/lowering template; all learned weights come
    // from the small source ONNX initializers, while ExtractImagePatches is
    // represented by the repo-owned native ExtractPatches runtime layer.
    public static class DeepFillV2OnnxNcnnImporter
    {
        private const int ExpectedOpset = 13;
        private const int ExpectedConvCount = 102;
        private const int ExpectedExtractImagePatchesCount = 8;
        private const int ExpectedExtractPatchesLayerCount = 8;
        private const int ExpectedCase1Opset = 17;
        private const int ExpectedCase1ConvCount = 82;

        public static DeepFillV2OnnxNcnnImportResult Import(string onnxPath, string paramTemplatePath)
        {
            if (string.IsNullOrWhiteSpace(onnxPath))
                throw new ArgumentException("DeepFillV2 ONNX path is required.", nameof(onnxPath));
            if (string.IsNullOrWhiteSpace(paramTemplatePath))
                throw new ArgumentException("DeepFillV2 NCNN param template path is required.", nameof(paramTemplatePath));
            if (!File.Exists(onnxPath))
                throw new FileNotFoundException("DeepFillV2 ONNX file not found.", onnxPath);
            if (!File.Exists(paramTemplatePath))
                throw new FileNotFoundException("DeepFillV2 NCNN param template not found.", paramTemplatePath);

            var onnx = OnnxModelReader.Read(onnxPath);
            var paramText = File.ReadAllText(paramTemplatePath);
            var paramModel = NcnnParamParser.Parse(paramText);
            if (onnx.graph.CountNodes("DeepFillV2ContextualAttention") == 1)
                return ImportCase1Official(onnx, onnxPath, paramText, paramTemplatePath, paramModel);
            ValidateOnnx(onnx, onnxPath);

            var convNodes = new List<OnnxNode>(ExpectedConvCount);
            for (var i = 0; i < onnx.graph.nodes.Count; i++)
            {
                var node = onnx.graph.nodes[i];
                if (string.Equals(node?.opType, "Conv", StringComparison.Ordinal))
                    convNodes.Add(node);
            }

            var ncnnConvLayers = new List<NcnnParamModel.Layer>(ExpectedConvCount);
            var extractPatchesLayers = 0;
            for (var i = 0; i < paramModel.layers.Count; i++)
            {
                var layer = paramModel.layers[i];
                if (layer == null)
                    continue;
                if (layer.type == NcnnLayerTypes.Convolution || layer.type == NcnnLayerTypes.ConvolutionDepthWise)
                    ncnnConvLayers.Add(layer);
                else if (layer.type == NcnnLayerTypes.ExtractPatches)
                    extractPatchesLayers++;
            }

            if (ncnnConvLayers.Count != convNodes.Count)
                throw new InvalidDataException("DeepFillV2 ONNX Conv count does not match NCNN param conv layer count: onnx="
                                               + convNodes.Count.ToString(CultureInfo.InvariantCulture)
                                               + " ncnn=" + ncnnConvLayers.Count.ToString(CultureInfo.InvariantCulture));
            if (extractPatchesLayers != ExpectedExtractPatchesLayerCount)
                throw new InvalidDataException("DeepFillV2 NCNN param must contain "
                                               + ExpectedExtractPatchesLayerCount.ToString(CultureInfo.InvariantCulture)
                                               + " ExtractPatches layers, got "
                                               + extractPatchesLayers.ToString(CultureInfo.InvariantCulture));

            byte[] binBytes;
            using (var ms = new MemoryStream(12 * 1024 * 1024))
            using (var bw = new BinaryWriter(ms))
            {
                for (var i = 0; i < ncnnConvLayers.Count; i++)
                    WriteConvLayerWeights(onnx.graph.initializers, convNodes[i], ncnnConvLayers[i], bw);
                bw.Flush();
                binBytes = ms.ToArray();
            }

            var report = new DeepFillV2OnnxNcnnImportReport
            {
                status = "passed",
                modelVariant = "hifill",
                onnxPath = Path.GetFullPath(onnxPath),
                paramTemplatePath = Path.GetFullPath(paramTemplatePath),
                onnxBytes = new FileInfo(onnxPath).Length,
                opset = onnx.opset,
                nodeCount = onnx.graph.nodes.Count,
                initializerCount = onnx.graph.initializers.Count,
                convNodeCount = convNodes.Count,
                ncnnConvLayerCount = ncnnConvLayers.Count,
                extractImagePatchesNodeCount = onnx.graph.CountNodes("ExtractImagePatches"),
                extractPatchesLayerCount = extractPatchesLayers,
                contextualAttentionNodeCount = 0,
                contextualAttentionLayerCount = 0,
                generatedBinBytes = binBytes.Length,
                generatedBinSha256 = Sha256(binBytes)
            };

            return new DeepFillV2OnnxNcnnImportResult
            {
                paramText = paramText,
                ncnnBinBytes = binBytes,
                report = report
            };
        }

        private static DeepFillV2OnnxNcnnImportResult ImportCase1Official(
            OnnxModel onnx,
            string onnxPath,
            string paramText,
            string paramTemplatePath,
            NcnnParamModel paramModel)
        {
            if (onnx.opset != ExpectedCase1Opset)
                throw new InvalidDataException("DeepFillV2 case1 ONNX opset must be " + ExpectedCase1Opset + ", got " + onnx.opset + ".");
            if (onnx.graph.CountNodes("Conv") != ExpectedCase1ConvCount)
                throw new InvalidDataException("DeepFillV2 case1 ONNX must contain " + ExpectedCase1ConvCount + " Conv nodes, got " + onnx.graph.CountNodes("Conv") + ".");
            if (onnx.graph.CountNodes("DeepFillV2ContextualAttention") != 1)
                throw new InvalidDataException("DeepFillV2 case1 ONNX must contain exactly one DeepFillV2ContextualAttention node.");

            ValidateValueInfo(onnx.graph.inputs, "image", TensorDataType.Float32, new long[] { 1, 3, 512, 400 }, "input");
            ValidateValueInfo(onnx.graph.inputs, "mask", TensorDataType.Float32, new long[] { 1, 1, 512, 400 }, "input");
            ValidateValueInfo(onnx.graph.outputs, "out0", TensorDataType.Float32, new long[] { 1, 3, 512, 400 }, "output");

            var convByName = new Dictionary<string, OnnxNode>(StringComparer.Ordinal);
            for (var i = 0; i < onnx.graph.nodes.Count; i++)
            {
                var node = onnx.graph.nodes[i];
                if (!string.Equals(node?.opType, "Conv", StringComparison.Ordinal))
                    continue;
                if (string.IsNullOrWhiteSpace(node.name) || convByName.ContainsKey(node.name))
                    throw new InvalidDataException("DeepFillV2 case1 ONNX Conv node names must be unique and non-empty.");
                convByName.Add(node.name, node);
            }

            var convLayers = new List<NcnnParamModel.Layer>(ExpectedCase1ConvCount);
            var attentionLayers = 0;
            for (var i = 0; i < paramModel.layers.Count; i++)
            {
                var layer = paramModel.layers[i];
                if (layer == null)
                    continue;
                if (layer.type == NcnnLayerTypes.Convolution)
                    convLayers.Add(layer);
                else if (layer.type == NcnnLayerTypes.DeepFillV2ContextualAttention)
                    attentionLayers++;
            }
            if (convLayers.Count != ExpectedCase1ConvCount)
                throw new InvalidDataException("DeepFillV2 case1 param must contain " + ExpectedCase1ConvCount + " Convolution layers, got " + convLayers.Count + ".");
            if (attentionLayers != 1)
                throw new InvalidDataException("DeepFillV2 case1 param must contain exactly one DeepFillV2ContextualAttention layer.");

            byte[] binBytes;
            using (var ms = new MemoryStream(20 * 1024 * 1024))
            using (var bw = new BinaryWriter(ms))
            {
                for (var i = 0; i < convLayers.Count; i++)
                {
                    var layer = convLayers[i];
                    if (!convByName.TryGetValue(layer.name, out var node))
                        throw new InvalidDataException("DeepFillV2 case1 ONNX Conv node not found for param layer " + layer.name + ".");
                    WriteConvLayerWeights(onnx.graph.initializers, node, layer, bw);
                }
                bw.Flush();
                binBytes = ms.ToArray();
            }

            var report = new DeepFillV2OnnxNcnnImportReport
            {
                status = "passed",
                modelVariant = "pytorch-case1-2021",
                onnxPath = Path.GetFullPath(onnxPath),
                paramTemplatePath = Path.GetFullPath(paramTemplatePath),
                onnxBytes = new FileInfo(onnxPath).Length,
                opset = onnx.opset,
                nodeCount = onnx.graph.nodes.Count,
                initializerCount = onnx.graph.initializers.Count,
                convNodeCount = convByName.Count,
                ncnnConvLayerCount = convLayers.Count,
                extractImagePatchesNodeCount = 0,
                extractPatchesLayerCount = 0,
                contextualAttentionNodeCount = 1,
                contextualAttentionLayerCount = attentionLayers,
                generatedBinBytes = binBytes.Length,
                generatedBinSha256 = Sha256(binBytes)
            };

            return new DeepFillV2OnnxNcnnImportResult
            {
                paramText = paramText,
                ncnnBinBytes = binBytes,
                report = report
            };
        }

        private static void ValidateOnnx(OnnxModel model, string path)
        {
            if (model == null || model.graph == null)
                throw new InvalidDataException("DeepFillV2 ONNX is empty: " + path);
            if (model.opset != ExpectedOpset)
                throw new InvalidDataException("DeepFillV2 ONNX opset must be "
                                               + ExpectedOpset.ToString(CultureInfo.InvariantCulture)
                                               + ", got " + model.opset.ToString(CultureInfo.InvariantCulture));
            if (model.graph.CountNodes("Conv") != ExpectedConvCount)
                throw new InvalidDataException("DeepFillV2 ONNX must contain "
                                               + ExpectedConvCount.ToString(CultureInfo.InvariantCulture)
                                               + " Conv nodes, got "
                                               + model.graph.CountNodes("Conv").ToString(CultureInfo.InvariantCulture));
            if (model.graph.CountNodes("ExtractImagePatches") != ExpectedExtractImagePatchesCount)
                throw new InvalidDataException("DeepFillV2 ONNX must contain "
                                               + ExpectedExtractImagePatchesCount.ToString(CultureInfo.InvariantCulture)
                                               + " ExtractImagePatches nodes, got "
                                               + model.graph.CountNodes("ExtractImagePatches").ToString(CultureInfo.InvariantCulture));

            ValidateValueInfo(model.graph.inputs, "mask:0", TensorDataType.Float32, new long[] { 1, 512, 512, 1 }, "input");
            ValidateValueInfo(model.graph.inputs, "img:0", TensorDataType.Float32, new long[] { 1, 512, 512, 3 }, "input");
            ValidateValueInfo(model.graph.outputs, "inpainted:0", TensorDataType.Float32, new long[] { 1, 512, 512, 3 }, "output");
        }

        private static void ValidateValueInfo(List<OnnxValueInfo> values, string name, TensorDataType dtype, long[] dims, string role)
        {
            OnnxValueInfo value = null;
            if (values != null)
            {
                for (var i = 0; i < values.Count; i++)
                {
                    if (string.Equals(values[i]?.name, name, StringComparison.Ordinal))
                    {
                        value = values[i];
                        break;
                    }
                }
            }

            if (value == null)
                throw new InvalidDataException("DeepFillV2 ONNX missing " + role + " " + name + ".");
            if (value.dataType != dtype)
                throw new InvalidDataException("DeepFillV2 ONNX " + role + " " + name + " dtype mismatch: " + value.dataType);
            if (!SameDims(value.dims, dims))
                throw new InvalidDataException("DeepFillV2 ONNX " + role + " " + name + " shape mismatch: expected "
                                               + FormatDims(dims) + " got " + FormatDims(value.dims));
        }

        private static bool SameDims(long[] actual, long[] expected)
        {
            if (actual == null || expected == null || actual.Length != expected.Length)
                return false;
            for (var i = 0; i < expected.Length; i++)
            {
                if (actual[i] != expected[i])
                    return false;
            }
            return true;
        }

        private static string FormatDims(long[] dims)
        {
            if (dims == null || dims.Length == 0)
                return "[]";
            return "[" + string.Join(",", Array.ConvertAll(dims, d => d.ToString(CultureInfo.InvariantCulture))) + "]";
        }

        private static void WriteConvLayerWeights(
            Dictionary<string, OnnxTensor> initializers,
            OnnxNode convNode,
            NcnnParamModel.Layer layer,
            BinaryWriter writer)
        {
            if (convNode == null)
                throw new InvalidDataException("DeepFillV2 missing ONNX Conv node for layer " + layer?.name);
            if (layer == null)
                throw new InvalidDataException("DeepFillV2 missing NCNN layer for ONNX node " + convNode.name);
            if (convNode.inputs.Count < 2)
                throw new InvalidDataException("ONNX Conv node has no weight input: " + convNode.name);

            var weightName = convNode.inputs[1];
            if (!initializers.TryGetValue(weightName, out var weight))
                throw new InvalidDataException("ONNX Conv weight initializer not found: " + convNode.name + " -> " + weightName);
            var expectedWeightElements = layer.GetInt(6, 0);
            var weightBytes = weight.GetFloat32LittleEndianBytes();
            if (weightBytes.Length != checked(expectedWeightElements * sizeof(float)))
            {
                throw new InvalidDataException("ONNX Conv weight size mismatch for " + convNode.name
                                               + " -> " + layer.name
                                               + ": expectedBytes=" + (expectedWeightElements * sizeof(float)).ToString(CultureInfo.InvariantCulture)
                                               + " got=" + weightBytes.Length.ToString(CultureInfo.InvariantCulture));
            }

            writer.Write(0u);
            writer.Write(weightBytes);

            var biasTerm = layer.GetInt(5, 0) != 0;
            if (!biasTerm)
                return;

            if (convNode.inputs.Count < 3)
                throw new InvalidDataException("NCNN layer expects bias but ONNX Conv has no bias: " + convNode.name + " -> " + layer.name);
            var biasName = convNode.inputs[2];
            if (!initializers.TryGetValue(biasName, out var bias))
                throw new InvalidDataException("ONNX Conv bias initializer not found: " + convNode.name + " -> " + biasName);
            var expectedBiasElements = layer.GetInt(0, 0);
            var biasBytes = bias.GetFloat32LittleEndianBytes();
            if (biasBytes.Length != checked(expectedBiasElements * sizeof(float)))
            {
                throw new InvalidDataException("ONNX Conv bias size mismatch for " + convNode.name
                                               + " -> " + layer.name
                                               + ": expectedBytes=" + (expectedBiasElements * sizeof(float)).ToString(CultureInfo.InvariantCulture)
                                               + " got=" + biasBytes.Length.ToString(CultureInfo.InvariantCulture));
            }
            writer.Write(biasBytes);
        }

        private static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create())
            {
                var hash = sha.ComputeHash(bytes ?? Array.Empty<byte>());
                var chars = new char[hash.Length * 2];
                for (var i = 0; i < hash.Length; i++)
                {
                    var b = hash[i];
                    chars[i * 2] = Hex(b >> 4);
                    chars[i * 2 + 1] = Hex(b & 0xF);
                }
                return new string(chars);
            }
        }

        private static char Hex(int value)
        {
            return (char)(value < 10 ? '0' + value : 'a' + (value - 10));
        }
    }
}
