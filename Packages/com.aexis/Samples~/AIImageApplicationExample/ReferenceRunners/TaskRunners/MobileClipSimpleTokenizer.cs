using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;

namespace Aexis.Samples.Runners
{

public sealed class MobileClipSimpleTokenizer
{
    public const int DefaultContextLength = 77;
    public const int StartTokenId = 49406;
    public const int EndTokenId = 49407;

    private readonly int _contextLength;
    private readonly string[] _byteEncoder;
    private readonly Dictionary<string, int> _encoder;
    private readonly Dictionary<(string first, string second), int> _bpeRanks;
    private readonly Dictionary<string, string> _cache = new Dictionary<string, string>(StringComparer.Ordinal);
    private readonly Regex _pattern;

    public MobileClipSimpleTokenizer(string vocabPath, string bpePath, int contextLength = DefaultContextLength)
    {
        if (string.IsNullOrWhiteSpace(vocabPath))
            throw new ArgumentNullException(nameof(vocabPath));
        if (string.IsNullOrWhiteSpace(bpePath))
            throw new ArgumentNullException(nameof(bpePath));

        _contextLength = Math.Max(1, contextLength);
        _byteEncoder = File.ReadAllLines(vocabPath);
        if (_byteEncoder.Length < 256)
            throw new InvalidOperationException("Tokenizer vocab.txt is incomplete: " + vocabPath);

        var merges = new List<(string first, string second)>();
        using (var sr = new StreamReader(bpePath))
        {
            sr.ReadLine();
            while (!sr.EndOfStream)
            {
                var line = sr.ReadLine();
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2)
                    merges.Add((parts[0], parts[1]));
            }
        }

        var vocab = new List<string>(_byteEncoder.Length * 2 + merges.Count + 2);
        foreach (var s in _byteEncoder)
            vocab.Add(s);
        foreach (var s in _byteEncoder)
            vocab.Add(s + "</w>");
        foreach (var merge in merges)
            vocab.Add(merge.first + merge.second);
        vocab.Add("<start_of_text>");
        vocab.Add("<end_of_text>");

        _encoder = new Dictionary<string, int>(vocab.Count, StringComparer.Ordinal);
        for (var i = 0; i < vocab.Count; i++)
            _encoder[vocab[i]] = i;

        _bpeRanks = new Dictionary<(string first, string second), int>(merges.Count);
        for (var i = 0; i < merges.Count; i++)
            _bpeRanks[merges[i]] = i;

        _pattern = new Regex(@"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\w\d]+",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);
    }

    public int[] Tokenize(string text)
    {
        var cleaned = Clean(text);
        var tokens = new List<int>(_contextLength);
        tokens.Add(StartTokenId);

        foreach (Match match in _pattern.Matches(cleaned))
        {
            var token = match.Value;
            if (string.IsNullOrEmpty(token))
                continue;

            var encodedToken = EncodeBytes(token);
            var bpe = ApplyBpe(encodedToken);
            var parts = bpe.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            for (var i = 0; i < parts.Length; i++)
            {
                if (_encoder.TryGetValue(parts[i], out var id))
                    tokens.Add(id);
            }
        }

        tokens.Add(EndTokenId);
        if (tokens.Count > _contextLength)
        {
            tokens.RemoveRange(_contextLength, tokens.Count - _contextLength);
            tokens[_contextLength - 1] = EndTokenId;
        }

        var padded = new int[_contextLength];
        for (var i = 0; i < tokens.Count; i++)
            padded[i] = tokens[i];
        return padded;
    }

    private static string Clean(string text)
    {
        var decoded = WebUtility.HtmlDecode(text ?? string.Empty);
        decoded = decoded.Trim().ToLowerInvariant();
        decoded = Regex.Replace(decoded, "\\s+", " ");
        return decoded;
    }

    private string EncodeBytes(string token)
    {
        var bytes = Encoding.UTF8.GetBytes(token);
        var sb = new StringBuilder(bytes.Length * 2);
        for (var i = 0; i < bytes.Length; i++)
        {
            var b = bytes[i];
            var index = b - 33;
            if (index < 0 || index >= _byteEncoder.Length)
                throw new InvalidOperationException("Tokenizer currently only supports prompt bytes >= 33: " + b);
            sb.Append(_byteEncoder[index]);
        }
        return sb.ToString();
    }

    private string ApplyBpe(string token)
    {
        if (_cache.TryGetValue(token, out var cached))
            return cached;

        var word = new List<string>(token.Length);
        for (var i = 0; i < token.Length; i++)
            word.Add(token[i].ToString());
        if (word.Count == 0)
            return token;

        word[word.Count - 1] += "</w>";
        while (word.Count > 1)
        {
            var bestRank = int.MaxValue;
            var bestPair = default((string first, string second));
            var found = false;
            for (var i = 0; i < word.Count - 1; i++)
            {
                var pair = (word[i], word[i + 1]);
                if (_bpeRanks.TryGetValue(pair, out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestPair = pair;
                    found = true;
                }
            }

            if (!found)
                break;

            var merged = new List<string>(word.Count);
            for (var i = 0; i < word.Count;)
            {
                if (i < word.Count - 1 && word[i] == bestPair.first && word[i + 1] == bestPair.second)
                {
                    merged.Add(bestPair.first + bestPair.second);
                    i += 2;
                }
                else
                {
                    merged.Add(word[i]);
                    i++;
                }
            }
            word = merged;
        }

        var result = string.Join(" ", word);
        _cache[token] = result;
        return result;
    }
}

}