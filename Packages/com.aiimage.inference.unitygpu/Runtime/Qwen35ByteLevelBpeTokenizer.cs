using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NcnnCompute
{
    public sealed class Qwen35ByteLevelBpeTokenizer
    {
        private readonly List<string> _idToToken = new List<string>();
        private readonly Dictionary<string, int> _tokenToId = new Dictionary<string, int>(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _mergeRanks = new Dictionary<string, int>(StringComparer.Ordinal);
        private readonly List<string> _specialTokens;
        private readonly HashSet<int> _specialTokenIds = new HashSet<int>();
        private readonly Dictionary<char, byte> _byteDecoder = new Dictionary<char, byte>();
        private readonly Dictionary<byte, char> _byteEncoder = new Dictionary<byte, char>();

        public Qwen35ByteLevelBpeTokenizer(
            string vocabPath,
            string mergesPath,
            IEnumerable<string> specialTokens,
            bool addMissingSpecialTokens = true)
        {
            foreach (var line in File.ReadAllLines(vocabPath, Encoding.UTF8))
            {
                var token = line.TrimEnd('\r', '\n');
                if (token.Length == 0) continue;
                _tokenToId[token] = _idToToken.Count;
                _idToToken.Add(token);
            }
            var rank = 0;
            foreach (var line in File.ReadAllLines(mergesPath, Encoding.UTF8))
            {
                var trimmed = line.Trim();
                if (trimmed.Length == 0 || trimmed.StartsWith("#", StringComparison.Ordinal)) continue;
                var fields = trimmed.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (fields.Length >= 2)
                {
                    var key = fields[0] + "\u0001" + fields[1];
                    if (!_mergeRanks.ContainsKey(key)) _mergeRanks[key] = rank++;
                }
            }
            _specialTokens = new List<string>();
            foreach (var token in specialTokens ?? Array.Empty<string>())
            {
                if (string.IsNullOrEmpty(token) || _specialTokens.Contains(token))
                    continue;
                if (!_tokenToId.TryGetValue(token, out var id))
                {
                    if (!addMissingSpecialTokens)
                        continue;
                    id = _idToToken.Count;
                    _tokenToId[token] = id;
                    _idToToken.Add(token);
                }
                _specialTokens.Add(token);
                _specialTokenIds.Add(id);
            }
            _specialTokens.Sort((a, b) => b.Length.CompareTo(a.Length));
            var extra = 0;
            for (var value = 0; value < 256; value++)
            {
                var printable = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) || value >= 174;
                var codePoint = printable ? value : 256 + extra++;
                var encoded = (char)codePoint;
                _byteEncoder[(byte)value] = encoded;
                _byteDecoder[encoded] = (byte)value;
            }
        }

        public int VocabularySize => _idToToken.Count;
        public int EndOfTurnId => IdOf("<|im_end|>");
        public int IdOf(string token) => token != null && _tokenToId.TryGetValue(token, out var id) ? id : -1;

        public List<int> Encode(string text)
        {
            var result = new List<int>();
            if (string.IsNullOrEmpty(text)) return result;
            var normal = new StringBuilder();
            void Flush()
            {
                if (normal.Length == 0) return;
                result.AddRange(EncodeNormal(normal.ToString()));
                normal.Length = 0;
            }
            for (var i = 0; i < text.Length;)
            {
                string special = null;
                for (var s = 0; s < _specialTokens.Count; s++)
                    if (i + _specialTokens[s].Length <= text.Length && string.CompareOrdinal(text, i, _specialTokens[s], 0, _specialTokens[s].Length) == 0) { special = _specialTokens[s]; break; }
                if (special == null) { normal.Append(text[i++]); continue; }
                Flush();
                result.Add(_tokenToId[special]);
                i += special.Length;
            }
            Flush();
            return result;
        }

        public string Decode(IReadOnlyList<int> ids, bool skipSpecialTokens = true)
        {
            var combined = new StringBuilder();
            for (var i = 0; i < ids.Count; i++)
            {
                var id = ids[i];
                if (id < 0 || id >= _idToToken.Count) continue;
                var token = _idToToken[id];
                if (skipSpecialTokens && _specialTokenIds.Contains(id)) continue;
                combined.Append(token == "\\t" ? "\t" : token == "\\n" ? "\n" : token == "\\r" ? "\r" : token);
            }
            var bytes = new List<byte>();
            for (var i = 0; i < combined.Length; i++) if (_byteDecoder.TryGetValue(combined[i], out var b)) bytes.Add(b);
            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        private List<int> EncodeNormal(string text)
        {
            var encoded = new StringBuilder();
            var utf8 = Encoding.UTF8.GetBytes(text);
            for (var i = 0; i < utf8.Length; i++) encoded.Append(_byteEncoder[utf8[i]]);
            var symbols = new List<string>();
            for (var i = 0; i < encoded.Length; i++) symbols.Add(encoded[i].ToString());
            while (symbols.Count >= 2)
            {
                var best = -1; var bestRank = int.MaxValue;
                for (var i = 0; i + 1 < symbols.Count; i++)
                {
                    if (_mergeRanks.TryGetValue(symbols[i] + "\u0001" + symbols[i + 1], out var rank) && rank < bestRank) { best = i; bestRank = rank; }
                }
                if (best < 0) break;
                symbols[best] = symbols[best] + symbols[best + 1];
                symbols.RemoveAt(best + 1);
            }
            var result = new List<int>();
            var unknown = IdOf("<unk>");
            for (var i = 0; i < symbols.Count; i++)
            {
                if (_tokenToId.TryGetValue(symbols[i], out var id)) { result.Add(id); continue; }
                for (var j = 0; j < symbols[i].Length; j++)
                {
                    if (_tokenToId.TryGetValue(symbols[i][j].ToString(), out id))
                        result.Add(id);
                    else if (unknown >= 0)
                        result.Add(unknown);
                }
            }
            return result;
        }
    }
}
