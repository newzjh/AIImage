using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace NcnnCompute
{
    [Serializable]
    public sealed class NcnnParamModel
    {
        [Serializable]
        public sealed class Layer
        {
            public string type;
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
