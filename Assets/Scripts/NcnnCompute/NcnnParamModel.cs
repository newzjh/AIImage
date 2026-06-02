using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace NcnnCompute
{
    [Serializable]
    [StructLayout(LayoutKind.Explicit, Size = 16)]
    public readonly struct NcnnLayerTypeKey : IEquatable<NcnnLayerTypeKey>
    {
        private static readonly Dictionary<NcnnLayerTypeKey, string> NameMap = new Dictionary<NcnnLayerTypeKey, string>();
        private static readonly object NameMapLock = new object();

        [FieldOffset(0)] public readonly ulong low;
        [FieldOffset(8)] public readonly ulong high;

        public NcnnLayerTypeKey(ulong low, ulong high)
        {
            this.low = low;
            this.high = high;
        }

        public static NcnnLayerTypeKey FromString(string value)
        {
            if (string.IsNullOrEmpty(value))
                return default;

            var key = CreateKey(value);
            lock (NameMapLock)
            {
                if (!NameMap.ContainsKey(key))
                    NameMap[key] = value;
            }
            return key;
        }

        public static implicit operator NcnnLayerTypeKey(string value)
        {
            return FromString(value);
        }

        public static implicit operator string(NcnnLayerTypeKey value)
        {
            return value.ToString();
        }

        public bool Equals(NcnnLayerTypeKey other)
        {
            return low == other.low && high == other.high;
        }

        public override bool Equals(object obj)
        {
            return obj is NcnnLayerTypeKey other && Equals(other);
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
            lock (NameMapLock)
            {
                if (NameMap.TryGetValue(this, out var value))
                    return value;
            }

            return low.ToString("X16", CultureInfo.InvariantCulture)
                   + high.ToString("X16", CultureInfo.InvariantCulture);
        }

        public static bool operator ==(NcnnLayerTypeKey left, NcnnLayerTypeKey right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(NcnnLayerTypeKey left, NcnnLayerTypeKey right)
        {
            return !left.Equals(right);
        }

        private static NcnnLayerTypeKey CreateKey(string value)
        {
            var ascii = Encoding.ASCII.GetBytes(value);
            if (ascii.Length <= 16)
            {
                var direct = new byte[16];
                Buffer.BlockCopy(ascii, 0, direct, 0, ascii.Length);
                return new NcnnLayerTypeKey(
                    BitConverter.ToUInt64(direct, 0),
                    BitConverter.ToUInt64(direct, 8));
            }

            using (var md5 = MD5.Create())
            {
                var hash = md5.ComputeHash(Encoding.UTF8.GetBytes(value));
                return new NcnnLayerTypeKey(
                    BitConverter.ToUInt64(hash, 0),
                    BitConverter.ToUInt64(hash, 8));
            }
        }
    }

    public static class NcnnLayerTypes
    {
        public static readonly NcnnLayerTypeKey Input = NcnnLayerTypeKey.FromString("Input");
        public static readonly NcnnLayerTypeKey Split = NcnnLayerTypeKey.FromString("Split");
        public static readonly NcnnLayerTypeKey Concat = NcnnLayerTypeKey.FromString("Concat");
        public static readonly NcnnLayerTypeKey Reshape = NcnnLayerTypeKey.FromString("Reshape");
        public static readonly NcnnLayerTypeKey ShuffleChannel = NcnnLayerTypeKey.FromString("ShuffleChannel");
        public static readonly NcnnLayerTypeKey Permute = NcnnLayerTypeKey.FromString("Permute");
        public static readonly NcnnLayerTypeKey Slice = NcnnLayerTypeKey.FromString("Slice");
        public static readonly NcnnLayerTypeKey ExpandDims = NcnnLayerTypeKey.FromString("ExpandDims");
        public static readonly NcnnLayerTypeKey Squeeze = NcnnLayerTypeKey.FromString("Squeeze");
        public static readonly NcnnLayerTypeKey Crop = NcnnLayerTypeKey.FromString("Crop");
        public static readonly NcnnLayerTypeKey Convolution = NcnnLayerTypeKey.FromString("Convolution");
        public static readonly NcnnLayerTypeKey ConvolutionDepthWise = NcnnLayerTypeKey.FromString("ConvolutionDepthWise");
        public static readonly NcnnLayerTypeKey Deconvolution = NcnnLayerTypeKey.FromString("Deconvolution");
        public static readonly NcnnLayerTypeKey Interp = NcnnLayerTypeKey.FromString("Interp");
        public static readonly NcnnLayerTypeKey Eltwise = NcnnLayerTypeKey.FromString("Eltwise");
        public static readonly NcnnLayerTypeKey BinaryOp = NcnnLayerTypeKey.FromString("BinaryOp");
        public static readonly NcnnLayerTypeKey UnaryOp = NcnnLayerTypeKey.FromString("UnaryOp");
        public static readonly NcnnLayerTypeKey Swish = NcnnLayerTypeKey.FromString("Swish");
        public static readonly NcnnLayerTypeKey Sigmoid = NcnnLayerTypeKey.FromString("Sigmoid");
        public static readonly NcnnLayerTypeKey GELU = NcnnLayerTypeKey.FromString("GELU");
        public static readonly NcnnLayerTypeKey Softmax = NcnnLayerTypeKey.FromString("Softmax");
        public static readonly NcnnLayerTypeKey Padding = NcnnLayerTypeKey.FromString("Padding");
        public static readonly NcnnLayerTypeKey Pooling = NcnnLayerTypeKey.FromString("Pooling");
        public static readonly NcnnLayerTypeKey InnerProduct = NcnnLayerTypeKey.FromString("InnerProduct");
        public static readonly NcnnLayerTypeKey MatMul = NcnnLayerTypeKey.FromString("MatMul");
        public static readonly NcnnLayerTypeKey Gemm = NcnnLayerTypeKey.FromString("Gemm");
        public static readonly NcnnLayerTypeKey MultiHeadAttention = NcnnLayerTypeKey.FromString("MultiHeadAttention");
        public static readonly NcnnLayerTypeKey LayerNorm = NcnnLayerTypeKey.FromString("LayerNorm");
        public static readonly NcnnLayerTypeKey GroupNorm = NcnnLayerTypeKey.FromString("GroupNorm");
        public static readonly NcnnLayerTypeKey BatchNorm = NcnnLayerTypeKey.FromString("BatchNorm");
        public static readonly NcnnLayerTypeKey Embed = NcnnLayerTypeKey.FromString("Embed");
        public static readonly NcnnLayerTypeKey Reduction = NcnnLayerTypeKey.FromString("Reduction");
        public static readonly NcnnLayerTypeKey MemoryData = NcnnLayerTypeKey.FromString("MemoryData");
        public static readonly NcnnLayerTypeKey ReLU = NcnnLayerTypeKey.FromString("ReLU");
        public static readonly NcnnLayerTypeKey MaxPoolingInd = NcnnLayerTypeKey.FromString("MaxPoolingInd");
        public static readonly NcnnLayerTypeKey MaxUnPooling = NcnnLayerTypeKey.FromString("MaxUnPooling");
        public static readonly NcnnLayerTypeKey Tile = NcnnLayerTypeKey.FromString("Tile");
    }

    [Serializable]
    public sealed class NcnnParamModel
    {
        [Serializable]
        public sealed class Layer
        {
            public NcnnLayerTypeKey type;
            public string name;
            public int bottoms;
            public int tops;
            public string[] bottomNames;
            public string[] topNames;
            public Dictionary<int, string> intParams = new Dictionary<int, string>();

            public string GetString(int key, string defaultValue = null)
            {
                if (intParams != null && intParams.TryGetValue(key, out var v))
                    return v;
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

    public static class NcnnParamParser
    {
        public static NcnnParamModel Parse(string text)
        {
            if (text == null)
                throw new ArgumentNullException(nameof(text));

            var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .Where(s => s.Length > 0 && !s.StartsWith("#"))
                .ToArray();

            if (lines.Length < 2)
                throw new FormatException("param too short");

            var model = new NcnnParamModel();
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
                var layer = new NcnnParamModel.Layer
                {
                    type = tok[0],
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
                }

                model.layers.Add(layer);
            }

            return model;
        }

        private static string[] SplitWs(string s)
        {
            return s.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
        }
    }
}
