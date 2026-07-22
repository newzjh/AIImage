using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Aexis.Execution
{
    [Serializable]
    [StructLayout(LayoutKind.Explicit, Size = 16)]
    public readonly struct AexisLayerTypeKey : IEquatable<AexisLayerTypeKey>
    {
        public const int MaxChars = 16;

        [FieldOffset(0)] public readonly ulong low;
        [FieldOffset(8)] public readonly ulong high;

        public AexisLayerTypeKey(ulong low, ulong high)
        {
            this.low = low;
            this.high = high;
        }

        public static AexisLayerTypeKey FromString(string value)
        {
            if (string.IsNullOrEmpty(value))
                return default;
            var bytes = new byte[16];
            var ascii = Encoding.ASCII.GetBytes(value);
            var copyCount = Math.Min(ascii.Length, MaxChars);
            Buffer.BlockCopy(ascii, 0, bytes, 0, copyCount);
            return new AexisLayerTypeKey(
                BitConverter.ToUInt64(bytes, 0),
                BitConverter.ToUInt64(bytes, 8));
        }

        public bool Equals(AexisLayerTypeKey other)
        {
            return low == other.low && high == other.high;
        }

        public override bool Equals(object obj)
        {
            return obj is AexisLayerTypeKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (low.GetHashCode() * 397) ^ high.GetHashCode();
            }
        }

        public override string ToString()
        {
            var bytes = new byte[16];
            Buffer.BlockCopy(BitConverter.GetBytes(low), 0, bytes, 0, sizeof(ulong));
            Buffer.BlockCopy(BitConverter.GetBytes(high), 0, bytes, sizeof(ulong), sizeof(ulong));
            var length = 0;
            while (length < bytes.Length && bytes[length] != 0)
                length++;
            return Encoding.ASCII.GetString(bytes, 0, length);
        }

        public static bool operator ==(AexisLayerTypeKey left, AexisLayerTypeKey right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(AexisLayerTypeKey left, AexisLayerTypeKey right)
        {
            return !left.Equals(right);
        }
    }

    public static class AexisLayerTypes
    {
        public static readonly AexisLayerTypeKey Input = AexisLayerTypeKey.FromString("Input");
        public static readonly AexisLayerTypeKey PnnxExpression = AexisLayerTypeKey.FromString("pnnx.Expression");
        public static readonly AexisLayerTypeKey AtenTo = AexisLayerTypeKey.FromString("aten::to");
        public static readonly AexisLayerTypeKey AbsVal = AexisLayerTypeKey.FromString("AbsVal");
        public static readonly AexisLayerTypeKey Split = AexisLayerTypeKey.FromString("Split");
        public static readonly AexisLayerTypeKey Concat = AexisLayerTypeKey.FromString("Concat");
        public static readonly AexisLayerTypeKey TanH = AexisLayerTypeKey.FromString("TanH");
        public static readonly AexisLayerTypeKey Reshape = AexisLayerTypeKey.FromString("Reshape");
        public static readonly AexisLayerTypeKey ShuffleChannel = AexisLayerTypeKey.FromString("ShuffleChannel");
        public static readonly AexisLayerTypeKey Permute = AexisLayerTypeKey.FromString("Permute");
        public static readonly AexisLayerTypeKey Slice = AexisLayerTypeKey.FromString("Slice");
        public static readonly AexisLayerTypeKey ExpandDims = AexisLayerTypeKey.FromString("ExpandDims");
        public static readonly AexisLayerTypeKey Squeeze = AexisLayerTypeKey.FromString("Squeeze");
        public static readonly AexisLayerTypeKey Crop = AexisLayerTypeKey.FromString("Crop");
        public static readonly AexisLayerTypeKey Convolution = AexisLayerTypeKey.FromString("Convolution");
        public static readonly AexisLayerTypeKey Convolution3D = AexisLayerTypeKey.FromString("Convolution3D");
        public static readonly AexisLayerTypeKey Convolution1D = AexisLayerTypeKey.FromString("Convolution1D");
        public static readonly AexisLayerTypeKey ConvolutionDepthWise = AexisLayerTypeKey.FromString("ConvolutionDepthWise");
        public static readonly AexisLayerTypeKey Deconvolution = AexisLayerTypeKey.FromString("Deconvolution");
        public static readonly AexisLayerTypeKey Deconvolution3D = AexisLayerTypeKey.FromString("Deconvolution3D");
        public static readonly AexisLayerTypeKey DeconvolutionDepthWise = AexisLayerTypeKey.FromString("DeconvolutionDepthWise");
        public static readonly AexisLayerTypeKey Interp = AexisLayerTypeKey.FromString("Interp");
        public static readonly AexisLayerTypeKey Dropout = AexisLayerTypeKey.FromString("Dropout");
        public static readonly AexisLayerTypeKey Eltwise = AexisLayerTypeKey.FromString("Eltwise");
        public static readonly AexisLayerTypeKey ELU = AexisLayerTypeKey.FromString("ELU");
        public static readonly AexisLayerTypeKey Erf = AexisLayerTypeKey.FromString("Erf");
        public static readonly AexisLayerTypeKey Flatten = AexisLayerTypeKey.FromString("Flatten");
        public static readonly AexisLayerTypeKey BinaryOp = AexisLayerTypeKey.FromString("BinaryOp");
        public static readonly AexisLayerTypeKey UnaryOp = AexisLayerTypeKey.FromString("UnaryOp");
        public static readonly AexisLayerTypeKey HardSigmoid = AexisLayerTypeKey.FromString("HardSigmoid");
        public static readonly AexisLayerTypeKey HardSwish = AexisLayerTypeKey.FromString("HardSwish");
        public static readonly AexisLayerTypeKey InstanceNorm = AexisLayerTypeKey.FromString("InstanceNorm");
        public static readonly AexisLayerTypeKey LRN = AexisLayerTypeKey.FromString("LRN");
        public static readonly AexisLayerTypeKey Mish = AexisLayerTypeKey.FromString("Mish");
        public static readonly AexisLayerTypeKey Swish = AexisLayerTypeKey.FromString("Swish");
        public static readonly AexisLayerTypeKey Noop = AexisLayerTypeKey.FromString("Noop");
        public static readonly AexisLayerTypeKey Normalize = AexisLayerTypeKey.FromString("Normalize");
        public static readonly AexisLayerTypeKey Packing = AexisLayerTypeKey.FromString("Packing");
        public static readonly AexisLayerTypeKey PixelShuffle = AexisLayerTypeKey.FromString("PixelShuffle");
        public static readonly AexisLayerTypeKey PReLU = AexisLayerTypeKey.FromString("PReLU");
        public static readonly AexisLayerTypeKey PriorBox = AexisLayerTypeKey.FromString("PriorBox");
        public static readonly AexisLayerTypeKey Quantize = AexisLayerTypeKey.FromString("Quantize");
        public static readonly AexisLayerTypeKey Dequantize = AexisLayerTypeKey.FromString("Dequantize");
        public static readonly AexisLayerTypeKey Requantize = AexisLayerTypeKey.FromString("Requantize");
        public static readonly AexisLayerTypeKey Reorg = AexisLayerTypeKey.FromString("Reorg");
        public static readonly AexisLayerTypeKey Sigmoid = AexisLayerTypeKey.FromString("Sigmoid");
        public static readonly AexisLayerTypeKey RMSNorm = AexisLayerTypeKey.FromString("RMSNorm");
        public static readonly AexisLayerTypeKey RotaryEmbed = AexisLayerTypeKey.FromString("RotaryEmbed");
        public static readonly AexisLayerTypeKey Scale = AexisLayerTypeKey.FromString("Scale");
        public static readonly AexisLayerTypeKey SDPA = AexisLayerTypeKey.FromString("SDPA");
        public static readonly AexisLayerTypeKey SELU = AexisLayerTypeKey.FromString("SELU");
        public static readonly AexisLayerTypeKey Shrink = AexisLayerTypeKey.FromString("Shrink");
        public static readonly AexisLayerTypeKey Softplus = AexisLayerTypeKey.FromString("Softplus");
        public static readonly AexisLayerTypeKey Softsign = AexisLayerTypeKey.FromString("Softsign");
        public static readonly AexisLayerTypeKey IsInf = AexisLayerTypeKey.FromString("IsInf");
        public static readonly AexisLayerTypeKey IsNaN = AexisLayerTypeKey.FromString("IsNaN");
        public static readonly AexisLayerTypeKey GELU = AexisLayerTypeKey.FromString("GELU");
        public static readonly AexisLayerTypeKey Cast = AexisLayerTypeKey.FromString("Cast");
        public static readonly AexisLayerTypeKey CELU = AexisLayerTypeKey.FromString("CELU");
        public static readonly AexisLayerTypeKey Clip = AexisLayerTypeKey.FromString("Clip");
        public static readonly AexisLayerTypeKey Trilu = AexisLayerTypeKey.FromString("Trilu");
        public static readonly AexisLayerTypeKey Softmax = AexisLayerTypeKey.FromString("Softmax");
        public static readonly AexisLayerTypeKey Padding = AexisLayerTypeKey.FromString("Padding");
        public static readonly AexisLayerTypeKey Pooling = AexisLayerTypeKey.FromString("Pooling");
        public static readonly AexisLayerTypeKey Pooling3D = AexisLayerTypeKey.FromString("Pooling3D");
        public static readonly AexisLayerTypeKey InnerProduct = AexisLayerTypeKey.FromString("InnerProduct");
        public static readonly AexisLayerTypeKey MatMul = AexisLayerTypeKey.FromString("MatMul");
        public static readonly AexisLayerTypeKey Gemm = AexisLayerTypeKey.FromString("Gemm");
        public static readonly AexisLayerTypeKey MultiHeadAttention = AexisLayerTypeKey.FromString("MultiHeadAttention");
        public static readonly AexisLayerTypeKey LayerNorm = AexisLayerTypeKey.FromString("LayerNorm");
        public static readonly AexisLayerTypeKey GroupNorm = AexisLayerTypeKey.FromString("GroupNorm");
        public static readonly AexisLayerTypeKey BatchNorm = AexisLayerTypeKey.FromString("BatchNorm");
        public static readonly AexisLayerTypeKey Embed = AexisLayerTypeKey.FromString("Embed");
        public static readonly AexisLayerTypeKey Reduction = AexisLayerTypeKey.FromString("Reduction");
        public static readonly AexisLayerTypeKey MemoryData = AexisLayerTypeKey.FromString("MemoryData");
        public static readonly AexisLayerTypeKey ReLU = AexisLayerTypeKey.FromString("ReLU");
        public static readonly AexisLayerTypeKey DeepCopy = AexisLayerTypeKey.FromString("DeepCopy");
        public static readonly AexisLayerTypeKey MaxPoolingInd = AexisLayerTypeKey.FromString("MaxPoolingInd");
        public static readonly AexisLayerTypeKey MaxUnPooling = AexisLayerTypeKey.FromString("MaxUnPooling");
        public static readonly AexisLayerTypeKey Tile = AexisLayerTypeKey.FromString("Tile");
        public static readonly AexisLayerTypeKey Unfold = AexisLayerTypeKey.FromString("Unfold");
        public static readonly AexisLayerTypeKey ExtractPatches = AexisLayerTypeKey.FromString("ExtractPatches");
        public static readonly AexisLayerTypeKey DeepFillV2ContextualAttention = AexisLayerTypeKey.FromString("DeepFillV2ContextualAttention");
        public static readonly AexisLayerTypeKey Shape = AexisLayerTypeKey.FromString("Shape");
        public static readonly AexisLayerTypeKey Size = AexisLayerTypeKey.FromString("Size");
        public static readonly AexisLayerTypeKey Range = AexisLayerTypeKey.FromString("Range");
        public static readonly AexisLayerTypeKey ConstantOfShape = AexisLayerTypeKey.FromString("ConstantOfShape");
        public static readonly AexisLayerTypeKey Expand = AexisLayerTypeKey.FromString("Expand");
        public static readonly AexisLayerTypeKey ArgMax = AexisLayerTypeKey.FromString("ArgMax");
        public static readonly AexisLayerTypeKey ArgMin = AexisLayerTypeKey.FromString("ArgMin");
        public static readonly AexisLayerTypeKey Where = AexisLayerTypeKey.FromString("Where");
        public static readonly AexisLayerTypeKey TopK = AexisLayerTypeKey.FromString("TopK");
        public static readonly AexisLayerTypeKey NonZero = AexisLayerTypeKey.FromString("NonZero");
        public static readonly AexisLayerTypeKey OneHot = AexisLayerTypeKey.FromString("OneHot");
        public static readonly AexisLayerTypeKey CumSum = AexisLayerTypeKey.FromString("CumSum");
        public static readonly AexisLayerTypeKey Compress = AexisLayerTypeKey.FromString("Compress");
        public static readonly AexisLayerTypeKey Gather = AexisLayerTypeKey.FromString("Gather");
        public static readonly AexisLayerTypeKey GatherElements = AexisLayerTypeKey.FromString("GatherElements");
        public static readonly AexisLayerTypeKey GatherND = AexisLayerTypeKey.FromString("GatherND");
        public static readonly AexisLayerTypeKey ScatterElements = AexisLayerTypeKey.FromString("ScatterElements");
        public static readonly AexisLayerTypeKey ScatterND = AexisLayerTypeKey.FromString("ScatterND");
        public static readonly AexisLayerTypeKey Scatter = AexisLayerTypeKey.FromString("Scatter");
        public static readonly AexisLayerTypeKey ShortConv = AexisLayerTypeKey.FromString("ShortConv");
        public static readonly AexisLayerTypeKey GatedDeltaRule = AexisLayerTypeKey.FromString("GatedDeltaRule");
        // Keep aliases below the fixed 16-byte layer key limit. Long NCNN names are
        // canonicalized by AexisLayerFactory before a key is looked up.
        public static readonly AexisLayerTypeKey Bias = AexisLayerTypeKey.FromString("Bias");
        public static readonly AexisLayerTypeKey BNLL = AexisLayerTypeKey.FromString("BNLL");
        public static readonly AexisLayerTypeKey CopyTo = AexisLayerTypeKey.FromString("CopyTo");
        public static readonly AexisLayerTypeKey Exp = AexisLayerTypeKey.FromString("Exp");
        public static readonly AexisLayerTypeKey Log = AexisLayerTypeKey.FromString("Log");
        public static readonly AexisLayerTypeKey Power = AexisLayerTypeKey.FromString("Power");
        public static readonly AexisLayerTypeKey Threshold = AexisLayerTypeKey.FromString("Threshold");
        public static readonly AexisLayerTypeKey ThresholdedRelu = AexisLayerTypeKey.FromString("ThresholdedRelu");
        public static readonly AexisLayerTypeKey MVN = AexisLayerTypeKey.FromString("MVN");
        public static readonly AexisLayerTypeKey Pooling1D = AexisLayerTypeKey.FromString("Pooling1D");
    }

    [Serializable]
    public sealed class AexisGraphModel
    {
        [Serializable]
        public sealed class Layer
        {
            public AexisLayerTypeKey type;
            public string typeName;
            public string name;
            public int bottoms;
            public int tops;
            public string[] bottomNames;
            public string[] topNames;
            public Dictionary<int, string> intParams = new Dictionary<int, string>();
            public Dictionary<string, string> stringParams = new Dictionary<string, string>(StringComparer.Ordinal);

            public string GetString(int key, string defaultValue = null)
            {
                if (intParams != null && intParams.TryGetValue(key, out var v))
                    return v;
                return defaultValue;
            }

            public string GetString(string key, string defaultValue = null)
            {
                if (!string.IsNullOrEmpty(key)
                    && stringParams != null
                    && stringParams.TryGetValue(key, out var v))
                {
                    return v;
                }

                return defaultValue;
            }

            public int GetInt(int key, int defaultValue = 0)
            {
                if (intParams != null && intParams.TryGetValue(key, out var v))
                {
                    if (int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i))
                        return i;
                }
                return defaultValue;
            }

            public float GetFloat(int key, float defaultValue = 0f)
            {
                if (intParams != null && intParams.TryGetValue(key, out var v))
                {
                    if (float.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var f))
                        return f;
                }
                return defaultValue;
            }

            public int[] GetInts(int key, int[] defaultValue = null)
            {
                if (intParams != null && intParams.TryGetValue(key, out var v))
                {
                    var parts = v.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 0)
                        return Array.Empty<int>();
                    var start = 0;
                    if (key <= -23300
                        && parts.Length >= 2
                        && int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var count)
                        && count == parts.Length - 1)
                    {
                        start = 1;
                    }

                    var arr = new int[parts.Length - start];
                    for (var i = start; i < parts.Length; i++)
                        arr[i - start] = int.Parse(parts[i], NumberStyles.Integer, CultureInfo.InvariantCulture);
                    return arr;
                }
                return defaultValue;
            }

            public float[] GetFloats(int key, float[] defaultValue = null)
            {
                if (intParams != null && intParams.TryGetValue(key, out var v))
                {
                    var parts = v.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 0)
                        return Array.Empty<float>();
                    var start = 0;
                    if (key <= -23300
                        && parts.Length >= 2
                        && int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var count)
                        && count == parts.Length - 1)
                    {
                        start = 1;
                    }

                    var arr = new float[parts.Length - start];
                    for (var i = start; i < parts.Length; i++)
                        arr[i - start] = float.Parse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture);
                    return arr;
                }
                return defaultValue;
            }

            public void CopyStringParamsFrom(Layer other, bool overwriteExisting = false)
            {
                if (other?.stringParams == null || other.stringParams.Count == 0)
                    return;

                foreach (var kv in other.stringParams)
                {
                    if (string.IsNullOrEmpty(kv.Key))
                        continue;

                    if (!overwriteExisting && stringParams.ContainsKey(kv.Key))
                        continue;

                    stringParams[kv.Key] = kv.Value;
                }
            }
        }

        public string magic;
        public int layerCount;
        public int blobCount;
        public List<Layer> layers = new List<Layer>();

        public Layer FindByName(string layerName)
        {
            if (layers == null || string.IsNullOrEmpty(layerName))
                return null;
            for (var i = 0; i < layers.Count; i++)
            {
                var l = layers[i];
                if (string.Equals(l?.name, layerName, StringComparison.Ordinal))
                    return l;
            }
            return null;
        }
    }

    public static class AexisGraphModelParser
    {
        public static AexisGraphModel Parse(string text)
        {
            if (text == null)
                throw new ArgumentNullException(nameof(text));

            var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .Where(s => s.Length > 0 && !s.StartsWith("#"))
                .ToArray();

            if (lines.Length < 2)
                throw new FormatException("param too short");

            var model = new AexisGraphModel();
            model.magic = lines[0];

            var header = SplitWs(lines[1]);
            if (header.Length < 2)
                throw new FormatException("param header invalid");
            model.layerCount = int.Parse(header[0], CultureInfo.InvariantCulture);
            model.blobCount = int.Parse(header[1], CultureInfo.InvariantCulture);

            for (var i = 2; i < lines.Length; i++)
            {
                var tok = SplitWs(lines[i]);
                if (tok.Length < 4)
                    continue;
                var layer = new AexisGraphModel.Layer
                {
                    type = AexisLayerTypeKey.FromString(tok[0]),
                    typeName = tok[0],
                    name = tok[1],
                    bottoms = int.Parse(tok[2], CultureInfo.InvariantCulture),
                    tops = int.Parse(tok[3], CultureInfo.InvariantCulture)
                };

                var idx = 4;
                layer.bottomNames = new string[layer.bottoms];
                for (var b = 0; b < layer.bottoms && idx < tok.Length; b++, idx++)
                    layer.bottomNames[b] = tok[idx];

                layer.topNames = new string[layer.tops];
                for (var t = 0; t < layer.tops && idx < tok.Length; t++, idx++)
                    layer.topNames[t] = tok[idx];

                for (; idx < tok.Length; idx++)
                {
                    var kv = tok[idx];
                    var eq = kv.IndexOf('=');
                    if (eq <= 0 || eq >= kv.Length - 1)
                        continue;
                    var kStr = kv.Substring(0, eq);
                    var vStr = kv.Substring(eq + 1);
                    if (int.TryParse(kStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var key))
                    {
                        layer.intParams[key] = vStr;
                    }
                    else if (ShouldKeepNamedParam(kStr))
                    {
                        layer.stringParams[kStr] = vStr;
                    }
                }

                model.layers.Add(layer);
            }

            return model;
        }

        public static int MergeStringParamsByLayerName(AexisGraphModel target, AexisGraphModel source, bool overwriteExisting = false)
        {
            if (target?.layers == null || source?.layers == null)
                return 0;

            var sourceByName = new Dictionary<string, AexisGraphModel.Layer>(StringComparer.Ordinal);
            for (var i = 0; i < source.layers.Count; i++)
            {
                var layer = source.layers[i];
                if (layer == null || string.IsNullOrEmpty(layer.name))
                    continue;

                sourceByName[layer.name] = layer;
            }

            var merged = 0;
            for (var i = 0; i < target.layers.Count; i++)
            {
                var layer = target.layers[i];
                if (layer == null || string.IsNullOrEmpty(layer.name))
                    continue;

                if (!sourceByName.TryGetValue(layer.name, out var sourceLayer) || sourceLayer == null)
                    continue;

                var before = layer.stringParams?.Count ?? 0;
                layer.CopyStringParamsFrom(sourceLayer, overwriteExisting);
                var after = layer.stringParams?.Count ?? 0;
                merged += Math.Max(0, after - before);
            }

            return merged;
        }

        private static string[] SplitWs(string s)
        {
            if (string.IsNullOrEmpty(s))
                return Array.Empty<string>();

            var tokens = new List<string>();
            var current = new StringBuilder();
            var inQuotes = false;

            for (var i = 0; i < s.Length; i++)
            {
                var ch = s[i];
                if (ch == '"')
                {
                    inQuotes = !inQuotes;
                    current.Append(ch);
                    continue;
                }

                if (!inQuotes && char.IsWhiteSpace(ch))
                {
                    if (current.Length > 0)
                    {
                        tokens.Add(current.ToString());
                        current.Clear();
                    }
                    continue;
                }

                current.Append(ch);
            }

            if (current.Length > 0)
                tokens.Add(current.ToString());

            return tokens.ToArray();
        }

        private static bool ShouldKeepNamedParam(string key)
        {
            if (string.IsNullOrEmpty(key))
                return false;

            var first = key[0];
            return first != '#' && first != '@';
        }
    }
}
