using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Aexis.Onnx;

namespace Aexis.Execution
{
    public sealed class AexisOnnxCompiledModel
    {
        public string paramText = string.Empty;
        public float[] immutableWeights = Array.Empty<float>();

        public AexisWeightReader CreateWeightReader() => new AexisFloatArrayWeightReader(immutableWeights);

        public void LoadInto(AexisGraphSession session, Action<AexisGraphSession.LoadProgress> onProgress = null)
        {
            if (session == null) throw new ArgumentNullException(nameof(session));
            using var reader = CreateWeightReader();
            session.LoadModel(paramText, reader, onProgress);
        }
    }

    public sealed class AexisFloatArrayWeightReader : AexisWeightReader
    {
        private readonly float[] _values;
        private int _position;
        public AexisFloatArrayWeightReader(float[] values) { _values = values ?? throw new ArgumentNullException(nameof(values)); }
        public override long Position => checked((long)_position * sizeof(float));
        public override float[] ReadFloat32Array(int count) => Read(count);
        public override float[] ReadTensorAsFloat32(int w, int h, int d, int c, int loadType) => Read(Count(w, h, d, c));
        public override void SkipTensor(int w, int h, int d, int c, int loadType) { _position = checked(_position + Count(w, h, d, c)); EnsurePosition(); }
        public override bool TryReadQuantizedTensor(int count, int expectedBlockSize, out AexisQuantizedTensor packed) { packed = null; return false; }
        public override void Dispose() { }

        private float[] Read(int count)
        {
            if (count < 0 || _position > _values.Length - count) throw new EndOfStreamException("ONNX immutable weight stream is truncated.");
            var result = new float[count];
            Array.Copy(_values, _position, result, 0, count);
            _position += count;
            return result;
        }

        private static int Count(int w, int h, int d, int c)
        {
            var count = Math.Max(1, w);
            if (h > 0) count = checked(count * h);
            if (d > 0) count = checked(count * d);
            if (c > 0) count = checked(count * c);
            return count;
        }

        private void EnsurePosition()
        {
            if (_position < 0 || _position > _values.Length) throw new EndOfStreamException("ONNX immutable weight stream is truncated.");
        }
    }

    public static class AexisOnnxGraphCompiler
    {
        public static AexisOnnxCompiledModel Compile(AexisOnnxGraphLoweringResult lowering)
        {
            if (lowering == null) throw new ArgumentNullException(nameof(lowering));
            if (!lowering.IsEligible) throw new InvalidOperationException("Cannot compile an ONNX graph with blocking lowering diagnostics.");
            var tensors = (lowering.initializers ?? Array.Empty<OnnxTensor>()).ToDictionary(tensor => tensor.name, StringComparer.Ordinal);
            var weights = new List<float>();
            foreach (var layer in lowering.graph.layers)
                AppendLayerWeights(layer, tensors, weights);
            return new AexisOnnxCompiledModel { paramText = SerializeParam(lowering.graph), immutableWeights = weights.ToArray() };
        }

        private static void AppendLayerWeights(AexisGraphModel.Layer layer, Dictionary<string, OnnxTensor> tensors, List<float> output)
        {
            if (layer == null) return;
            switch (layer.typeName)
            {
                case "Convolution":
                case "ConvolutionDepthWise":
                case "Convolution1D":
                    AppendNamed(layer, "onnx.weight", tensors, output, true);
                    if (layer.GetInt(5, 0) != 0) AppendNamed(layer, "onnx.bias", tensors, output, true);
                    break;
                case "Deconvolution":
                    AppendConvTransposeWeight(layer, tensors, output);
                    if (layer.GetInt(5, 0) != 0) AppendNamed(layer, "onnx.bias", tensors, output, true);
                    break;
                case "BatchNorm":
                    AppendNamed(layer, "onnx.scale", tensors, output, true);
                    AppendNamed(layer, "onnx.mean", tensors, output, true);
                    AppendNamed(layer, "onnx.variance", tensors, output, true);
                    AppendNamed(layer, "onnx.bias", tensors, output, true);
                    break;
                case "Gemm":
                    if (layer.GetInt(5, 0) != 0) AppendNamed(layer, "onnx.b", tensors, output, true);
                    if (layer.GetInt(6, 0) != 0) AppendNamed(layer, "onnx.c", tensors, output, true);
                    break;
                case "InstanceNorm":
                case "LayerNorm":
                    if (layer.GetInt(2, 1) != 0)
                    {
                        AppendNamed(layer, "onnx.scale", tensors, output, true);
                        AppendNamed(layer, "onnx.bias", tensors, output, true);
                    }
                    break;
                case "PReLU":
                    AppendNamed(layer, "onnx.slope", tensors, output, true);
                    break;
                case "MemoryData":
                    AppendNamedAsFloat(layer, "onnx.tensor", tensors, output, true);
                    break;
            }
        }

        private static void AppendConvTransposeWeight(AexisGraphModel.Layer layer, Dictionary<string, OnnxTensor> tensors, List<float> output)
        {
            if (layer.stringParams == null || !layer.stringParams.TryGetValue("onnx.weight", out var name) || !tensors.TryGetValue(name, out var tensor))
                throw new InvalidDataException("Missing ONNX ConvTranspose weight for layer " + layer.name + ".");
            var values = GetFloatValues(tensor);
            if (tensor.dims == null || tensor.dims.Length != 4)
                throw new InvalidDataException("ONNX ConvTranspose weight must be rank-4 IOHW: " + name + ".");
            var inChannels = checked((int)tensor.dims[0]);
            var outChannelsPerGroup = checked((int)tensor.dims[1]);
            var kernelH = checked((int)tensor.dims[2]);
            var kernelW = checked((int)tensor.dims[3]);
            var group = Math.Max(1, layer.GetInt(7, 1));
            if (group != 1)
                throw new InvalidDataException("Grouped ONNX ConvTranspose immutable packing is not yet supported by the strict Aexis deconvolution kernel: " + layer.name + ".");
            for (var outputChannel = 0; outputChannel < outChannelsPerGroup; outputChannel++)
                for (var inputChannel = 0; inputChannel < inChannels; inputChannel++)
                    for (var y = 0; y < kernelH; y++)
                        for (var x = 0; x < kernelW; x++)
                        {
                            var source = (((inputChannel * outChannelsPerGroup) + outputChannel) * kernelH + y) * kernelW + x;
                            output.Add(values[source]);
                        }
        }

        private static void AppendNamed(AexisGraphModel.Layer layer, string key, Dictionary<string, OnnxTensor> tensors, List<float> output, bool required)
        {
            if (layer.stringParams == null || !layer.stringParams.TryGetValue(key, out var name) || !tensors.TryGetValue(name, out var tensor))
            {
                if (required) throw new InvalidDataException("Missing ONNX initializer binding " + key + " for layer " + layer.name + ".");
                return;
            }
            output.AddRange(GetFloatValues(tensor));
        }

        private static void AppendNamedAsFloat(AexisGraphModel.Layer layer, string key, Dictionary<string, OnnxTensor> tensors, List<float> output, bool required)
        {
            if (layer.stringParams == null || !layer.stringParams.TryGetValue(key, out var name) || !tensors.TryGetValue(name, out var tensor))
            {
                if (required) throw new InvalidDataException("Missing ONNX initializer binding " + key + " for layer " + layer.name + ".");
                return;
            }
            if (tensor.dataType == TensorDataType.Float32)
            {
                output.AddRange(GetFloatValues(tensor));
                return;
            }
            if (tensor.dataType != TensorDataType.Int32)
                throw new InvalidDataException("MemoryData initializer must be FP32 or Int32: " + name + ".");
            var count = checked((int)tensor.ElementCount);
            if (tensor.int32Data != null && tensor.int32Data.Length == count)
            {
                for (var i = 0; i < count; i++) output.Add(tensor.int32Data[i]);
                return;
            }
            if (tensor.int64Data != null && tensor.int64Data.Length == count)
            {
                for (var i = 0; i < count; i++)
                {
                    if (tensor.int64Data[i] < int.MinValue || tensor.int64Data[i] > int.MaxValue) throw new InvalidDataException("INT64 initializer value exceeds Aexis Int32 index range: " + name + ".");
                    output.Add(tensor.int64Data[i]);
                }
                return;
            }
            if (tensor.rawData != null && tensor.onnxDataType == 7 && tensor.rawData.Length == checked(count * sizeof(long)))
            {
                for (var i = 0; i < count; i++)
                {
                    var value = BitConverter.ToInt64(tensor.rawData, i * sizeof(long));
                    if (value < int.MinValue || value > int.MaxValue) throw new InvalidDataException("INT64 initializer value exceeds Aexis Int32 index range: " + name + ".");
                    output.Add(value);
                }
                return;
            }
            if (tensor.rawData != null && tensor.onnxDataType == 6 && tensor.rawData.Length == checked(count * sizeof(int)))
            {
                for (var i = 0; i < count; i++) output.Add(BitConverter.ToInt32(tensor.rawData, i * sizeof(int)));
                return;
            }
            if (tensor.rawData != null && tensor.onnxDataType == 9 && tensor.rawData.Length == count)
            {
                for (var i = 0; i < count; i++) output.Add(tensor.rawData[i] == 0 ? 0f : 1f);
                return;
            }
            throw new InvalidDataException("Int32 initializer has no decoded payload: " + name + ".");
        }

        private static float[] GetFloatValues(OnnxTensor tensor)
        {
            var bytes = tensor.GetFloat32LittleEndianBytes();
            var values = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            if (!BitConverter.IsLittleEndian)
                throw new PlatformNotSupportedException("Aexis ONNX immutable compiler currently requires a little-endian host.");
            return values;
        }

        private static string SerializeParam(AexisGraphModel graph)
        {
            var blobs = new HashSet<string>(StringComparer.Ordinal);
            foreach (var layer in graph.layers)
            {
                foreach (var name in layer.bottomNames ?? Array.Empty<string>()) if (!string.IsNullOrEmpty(name)) blobs.Add(name);
                foreach (var name in layer.topNames ?? Array.Empty<string>()) if (!string.IsNullOrEmpty(name)) blobs.Add(name);
            }
            var builder = new StringBuilder();
            builder.AppendLine("7767517");
            builder.Append(graph.layers.Count.ToString(CultureInfo.InvariantCulture)).Append(' ').Append(blobs.Count.ToString(CultureInfo.InvariantCulture)).AppendLine();
            foreach (var layer in graph.layers)
            {
                builder.Append(layer.typeName).Append(' ').Append(layer.name).Append(' ')
                    .Append(layer.bottoms.ToString(CultureInfo.InvariantCulture)).Append(' ')
                    .Append(layer.tops.ToString(CultureInfo.InvariantCulture));
                foreach (var name in layer.bottomNames ?? Array.Empty<string>()) builder.Append(' ').Append(name);
                foreach (var name in layer.topNames ?? Array.Empty<string>()) builder.Append(' ').Append(name);
                foreach (var pair in layer.intParams.OrderBy(pair => pair.Key)) builder.Append(' ').Append(pair.Key.ToString(CultureInfo.InvariantCulture)).Append('=').Append(pair.Value);
                foreach (var pair in layer.stringParams.OrderBy(pair => pair.Key, StringComparer.Ordinal))
                    if (!pair.Key.StartsWith("onnx.", StringComparison.Ordinal)) builder.Append(' ').Append(pair.Key).Append('=').Append(pair.Value);
                builder.AppendLine();
            }
            return builder.ToString();
        }
    }
}
