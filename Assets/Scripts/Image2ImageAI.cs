using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
 
public sealed class Image2ImageAI : MonoBehaviour
{
    public enum Provider
    {
        GoogleAIStudio,
        Replicate,
        AliTongyiWanxiang,
        Doubao
    }
 

    [SerializeField] private Provider provider = Provider.Replicate;
 
    [Header("Google AI Studio (Gemini)")]
    [SerializeField] private string googleApiKey;
    [SerializeField] private string googleModel = "gemini-2.0-flash-exp";
    [SerializeField] private string googleBaseUrl = "https://generativelanguage.googleapis.com/v1beta";
 
    [Header("Replicate")]
    [SerializeField] private string replicateApiToken;
    [SerializeField] private string replicateVersion;
    [SerializeField] private string replicateInputImageField = "image";
    [SerializeField] private float replicateStrength = 0.65f;
 
    [Header("Ali Tongyi Wanxiang (DashScope)")]
    [SerializeField] private string dashScopeApiKey;
    [SerializeField] private string dashScopeModel = "wanx-v1";
    [SerializeField] private string dashScopeBaseUrl = "https://dashscope.aliyuncs.com";
 
    [Header("Doubao")]
    [SerializeField] private string doubaoApiKey = "ba0e3461-60b9-49d2-87e6-d6e6d95e10bd";
    [SerializeField] private string doubaoBaseUrl = "https://ark.cn-beijing.volces.com/api/v3";
 
    [Header("Runtime")]
    [SerializeField] private int timeoutSeconds = 180;
    [SerializeField] private int pollIntervalMs = 900;
 
    public async UniTask<Texture2D> ImageToImageAsync(
        Texture2D input,
        string prompt,
        int targetWidth,
        int targetHeight,
        CancellationToken cancellationToken)
    {
        if (input == null) return null;
        if (string.IsNullOrWhiteSpace(prompt)) return null;
 
        Texture2D raw;
        switch (provider)
        {
            case Provider.GoogleAIStudio:
                raw = await GoogleGeminiImageEditAsync(input, prompt, cancellationToken);
                break;
            case Provider.Replicate:
                raw = await ReplicateImageToImageAsync(input, prompt, cancellationToken);
                break;
            case Provider.AliTongyiWanxiang:
                raw = await DashScopeImageToImageAsync(input, prompt, cancellationToken);
                break;
            case Provider.Doubao:
                raw = await DoubaoImageToImageAsync(input, prompt, cancellationToken);
                break;
            default:
                return null;
        }
 
        if (raw == null) return null;
 
        if (targetWidth > 0 && targetHeight > 0 && (raw.width != targetWidth || raw.height != targetHeight))
        {
            var resized = ResizeTo(raw, targetWidth, targetHeight);
            Destroy(raw);
            raw = resized;
        }
 
        raw.name = "AI_" + provider;
        return raw;
    }
 
    private async UniTask<Texture2D> ReplicateImageToImageAsync(Texture2D input, string prompt, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(replicateApiToken)) return null;
        if (string.IsNullOrWhiteSpace(replicateVersion)) return null;
 
        var png = EncodePng(input);
        if (png == null || png.Length == 0) return null;
 
        var imageDataUrl = "data:image/png;base64," + Convert.ToBase64String(png);
 
        var requestJson = BuildJsonObject(new Dictionary<string, object>
        {
            ["version"] = replicateVersion,
            ["input"] = new Dictionary<string, object>
            {
                ["prompt"] = prompt,
                [replicateInputImageField] = imageDataUrl,
                ["strength"] = replicateStrength
            }
        });
 
        var createUrl = "https://api.replicate.com/v1/predictions";
        var created = await SendJsonAsync(createUrl, "POST", requestJson, new Dictionary<string, string>
        {
            ["Authorization"] = "Token " + replicateApiToken
        }, ct);
 
        if (string.IsNullOrWhiteSpace(created))
            return null;
 
        var id = ExtractJsonString(created, "id");
        if (string.IsNullOrWhiteSpace(id))
            return null;
 
        var deadline = DateTime.UtcNow.AddSeconds(Mathf.Max(5, timeoutSeconds));
        while (DateTime.UtcNow < deadline)
        {
            ct.ThrowIfCancellationRequested();
 
            var pollUrl = "https://api.replicate.com/v1/predictions/" + id;
            var polled = await SendJsonAsync(pollUrl, "GET", null, new Dictionary<string, string>
            {
                ["Authorization"] = "Token " + replicateApiToken
            }, ct);
 
            if (string.IsNullOrWhiteSpace(polled))
                return null;
 
            var status = ExtractJsonString(polled, "status");
            if (string.Equals(status, "succeeded", StringComparison.OrdinalIgnoreCase))
            {
                var outputUrl = ExtractFirstUrlFromJson(polled);
                if (string.IsNullOrWhiteSpace(outputUrl))
                    return null;
                return await DownloadTextureAsync(outputUrl, ct);
            }
 
            if (string.Equals(status, "failed", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(status, "canceled", StringComparison.OrdinalIgnoreCase))
                return null;
 
            await UniTask.Delay(pollIntervalMs, cancellationToken: ct);
        }
 
        return null;
    }
 
    private async UniTask<Texture2D> DashScopeImageToImageAsync(Texture2D input, string prompt, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(dashScopeApiKey)) return null;
 
        var png = EncodePng(input);
        if (png == null || png.Length == 0) return null;
 
        var imageBase64 = Convert.ToBase64String(png);
 
        var url = dashScopeBaseUrl.TrimEnd('/') + "/api/v1/services/aigc/image2image/image-synthesis";
 
        var requestJson = BuildJsonObject(new Dictionary<string, object>
        {
            ["model"] = dashScopeModel,
            ["input"] = new Dictionary<string, object>
            {
                ["prompt"] = prompt,
                ["img"] = "data:image/png;base64," + imageBase64
            },
            ["parameters"] = new Dictionary<string, object>
            {
                ["n"] = 1
            }
        });
 
        var resp = await SendJsonAsync(url, "POST", requestJson, new Dictionary<string, string>
        {
            ["Authorization"] = "Bearer " + dashScopeApiKey
        }, ct);
 
        if (string.IsNullOrWhiteSpace(resp))
            return null;
 
        var outputUrl = ExtractFirstUrlFromJson(resp);
        if (!string.IsNullOrWhiteSpace(outputUrl))
            return await DownloadTextureAsync(outputUrl, ct);
 
        var b64 = ExtractJsonImageBase64(resp);
        if (!string.IsNullOrWhiteSpace(b64))
            return DecodeTextureFromBase64(b64);
 
        var taskId = ExtractJsonString(resp, "task_id");
        if (string.IsNullOrWhiteSpace(taskId))
            taskId = ExtractJsonString(resp, "taskId");
 
        if (string.IsNullOrWhiteSpace(taskId))
            return null;
 
        var pollUrl = dashScopeBaseUrl.TrimEnd('/') + "/api/v1/tasks/" + taskId;
        var deadline = DateTime.UtcNow.AddSeconds(Mathf.Max(5, timeoutSeconds));
        while (DateTime.UtcNow < deadline)
        {
            ct.ThrowIfCancellationRequested();
            var polled = await SendJsonAsync(pollUrl, "GET", null, new Dictionary<string, string>
            {
                ["Authorization"] = "Bearer " + dashScopeApiKey
            }, ct);
 
            if (string.IsNullOrWhiteSpace(polled))
                return null;
 
            var status = ExtractJsonString(polled, "task_status");
            if (string.IsNullOrWhiteSpace(status))
                status = ExtractJsonString(polled, "status");
 
            if (string.Equals(status, "SUCCEEDED", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(status, "succeeded", StringComparison.OrdinalIgnoreCase))
            {
                outputUrl = ExtractFirstUrlFromJson(polled);
                if (!string.IsNullOrWhiteSpace(outputUrl))
                    return await DownloadTextureAsync(outputUrl, ct);
                b64 = ExtractJsonImageBase64(polled);
                if (!string.IsNullOrWhiteSpace(b64))
                    return DecodeTextureFromBase64(b64);
                return null;
            }
 
            if (string.Equals(status, "FAILED", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(status, "failed", StringComparison.OrdinalIgnoreCase))
                return null;
 
            await UniTask.Delay(pollIntervalMs, cancellationToken: ct);
        }
 
        return null;
    }
 
    private async UniTask<Texture2D> GoogleGeminiImageEditAsync(Texture2D input, string prompt, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(googleApiKey)) return null;
        if (string.IsNullOrWhiteSpace(googleModel)) return null;
 
        var png = EncodePng(input);
        if (png == null || png.Length == 0) return null;
 
        var base64 = Convert.ToBase64String(png);
        var url = googleBaseUrl.TrimEnd('/') + "/models/" + googleModel + ":generateContent?key=" + UnityWebRequest.EscapeURL(googleApiKey);
 
        var requestJson =
            "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":" + Quote(prompt) + "},{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":" + Quote(base64) + "}}]}]}";
 
        var resp = await SendJsonAsync(url, "POST", requestJson, null, ct);
        if (string.IsNullOrWhiteSpace(resp))
            return null;
 
        var outB64 = ExtractGeminiInlineImageBase64(resp);
        if (!string.IsNullOrWhiteSpace(outB64))
            return DecodeTextureFromBase64(outB64);
 
        var outUrl = ExtractFirstUrlFromJson(resp);
        if (!string.IsNullOrWhiteSpace(outUrl))
            return await DownloadTextureAsync(outUrl, ct);
 
        return null;
    }
 
    private async UniTask<Texture2D> DoubaoImageToImageAsync(Texture2D input, string prompt, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(doubaoApiKey)) return null;
 
        var png = EncodePng(input);
        if (png == null || png.Length == 0) return null;
 
        var base64 = Convert.ToBase64String(png);
        var dataUrl = "data:image/png;base64," + base64;

        //var url = doubaoBaseUrl.TrimEnd('/') + "/images/edits";
        var url = doubaoBaseUrl.TrimEnd('/') + "/images/generations";
        //string url = "https://ark.cn-beijing.volces.com/api/v3/images/generations";
        var requestJson = BuildJsonObject(new Dictionary<string, object>
        {
            ["model"] = "doubao-seedream-4-5-251128",
            ["prompt"] = prompt,
            ["image"] = dataUrl,
            ["n"] = 1,
            ["response_format"] = "b64_json"
        });
 
        var resp = await SendJsonAsync(url, "POST", requestJson, new Dictionary<string, string>
        {
            ["Authorization"] = "Bearer " + doubaoApiKey
        }, ct);
 
        if (string.IsNullOrWhiteSpace(resp))
            return null;
 
        var b64 = ExtractJsonImageBase64(resp);
        if (!string.IsNullOrWhiteSpace(b64))
            return DecodeTextureFromBase64(b64);
 
        var outUrl = ExtractFirstUrlFromJson(resp);
        if (!string.IsNullOrWhiteSpace(outUrl))
            return await DownloadTextureAsync(outUrl, ct);
 
        return null;
    }
 
    private static byte[] EncodePng(Texture2D tex)
    {
        try
        {
            return tex.EncodeToPNG();
        }
        catch
        {
            return null;
        }
    }
 
    private static Texture2D DecodeTextureFromBase64(string base64)
    {
        if (string.IsNullOrWhiteSpace(base64)) return null;
        byte[] data;
        try
        {
            data = Convert.FromBase64String(base64);
        }
        catch
        {
            return null;
        }
 
        var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        if (!tex.LoadImage(data, false))
        {
            Destroy(tex);
            return null;
        }
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
    }
 
    private static Texture2D ResizeTo(Texture2D src, int targetWidth, int targetHeight)
    {
        var rt = RenderTexture.GetTemporary(targetWidth, targetHeight, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        var prev = RenderTexture.active;
        try
        {
            Graphics.Blit(src, rt);
            RenderTexture.active = rt;
            var tex = new Texture2D(targetWidth, targetHeight, TextureFormat.RGBA32, false);
            tex.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
            tex.Apply(false, false);
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            return tex;
        }
        finally
        {
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);
        }
    }
 
    private static async UniTask<Texture2D> DownloadTextureAsync(string url, CancellationToken ct)
    {
        using (var req = UnityWebRequestTexture.GetTexture(url, true))
        {
            req.timeout = 180;
            await req.SendWebRequest().ToUniTask(cancellationToken: ct);
            if (req.result != UnityWebRequest.Result.Success)
                return null;
            var tex = DownloadHandlerTexture.GetContent(req);
            if (tex == null) return null;
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            return tex;
        }
    }
 
    private async UniTask<string> SendJsonAsync(
        string url,
        string method,
        string jsonBody,
        Dictionary<string, string> headers,
        CancellationToken ct)
    {
        using (var req = new UnityWebRequest(url, method))
        {
            req.downloadHandler = new DownloadHandlerBuffer();
            if (!string.IsNullOrEmpty(jsonBody))
            {
                var bytes = Encoding.UTF8.GetBytes(jsonBody);
                req.uploadHandler = new UploadHandlerRaw(bytes);
                req.SetRequestHeader("Content-Type", "application/json");
            }
 
            if (headers != null)
            {
                foreach (var kv in headers)
                {
                    if (!string.IsNullOrWhiteSpace(kv.Key) && kv.Value != null)
                        req.SetRequestHeader(kv.Key, kv.Value);
                }
            }
 
            req.timeout = Mathf.Max(5, timeoutSeconds);
            await req.SendWebRequest().ToUniTask(cancellationToken: ct);
            if (req.result != UnityWebRequest.Result.Success)
                return req.downloadHandler?.text;
            return req.downloadHandler?.text;
        }
    }
 
    private static string ExtractFirstUrlFromJson(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        var m = Regex.Match(json, "\"url\"\\s*:\\s*\"(?<u>https?:\\\\/\\\\/[^\\\"]+)\"", RegexOptions.IgnoreCase);
        if (m.Success)
            return UnescapeJsonString(m.Groups["u"].Value);
        m = Regex.Match(json, "\"output\"\\s*:\\s*\\[\\s*\"(?<u>https?:\\\\/\\\\/[^\\\"]+)\"", RegexOptions.IgnoreCase);
        if (m.Success)
            return UnescapeJsonString(m.Groups["u"].Value);
        m = Regex.Match(json, "\"output\"\\s*:\\s*\"(?<u>https?:\\\\/\\\\/[^\\\"]+)\"", RegexOptions.IgnoreCase);
        if (m.Success)
            return UnescapeJsonString(m.Groups["u"].Value);
        return null;
    }
 
    private static string ExtractJsonString(string json, string key)
    {
        if (string.IsNullOrWhiteSpace(json) || string.IsNullOrWhiteSpace(key)) return null;
        var m = Regex.Match(json, "\"" + Regex.Escape(key) + "\"\\s*:\\s*\"(?<v>(\\\\.|[^\"])*)\"", RegexOptions.IgnoreCase);
        if (!m.Success) return null;
        return UnescapeJsonString(m.Groups["v"].Value);
    }
 
    private static string ExtractJsonImageBase64(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        var m = Regex.Match(json, "\"b64_json\"\\s*:\\s*\"(?<v>(\\\\.|[^\"])*)\"", RegexOptions.IgnoreCase);
        if (m.Success) return UnescapeJsonString(m.Groups["v"].Value);
 
        m = Regex.Match(json, "\"data\"\\s*:\\s*\"(?<v>(\\\\.|[^\"])*)\"", RegexOptions.IgnoreCase);
        if (m.Success)
        {
            var s = UnescapeJsonString(m.Groups["v"].Value);
            if (LooksLikeBase64(s))
                return s;
        }
 
        return null;
    }
 
    private static string ExtractGeminiInlineImageBase64(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
 
        var m = Regex.Match(json, "\"inlineData\"\\s*:\\s*\\{[^\\}]*\"mimeType\"\\s*:\\s*\"image\\/[^\\\"]+\"[^\\}]*\"data\"\\s*:\\s*\"(?<v>(\\\\.|[^\"])*)\"", RegexOptions.IgnoreCase);
        if (m.Success) return UnescapeJsonString(m.Groups["v"].Value);
 
        m = Regex.Match(json, "\"inline_data\"\\s*:\\s*\\{[^\\}]*\"mime_type\"\\s*:\\s*\"image\\/[^\\\"]+\"[^\\}]*\"data\"\\s*:\\s*\"(?<v>(\\\\.|[^\"])*)\"", RegexOptions.IgnoreCase);
        if (m.Success) return UnescapeJsonString(m.Groups["v"].Value);
 
        return null;
    }
 
    private static bool LooksLikeBase64(string s)
    {
        if (string.IsNullOrWhiteSpace(s)) return false;
        if (s.Length < 64) return false;
        return Regex.IsMatch(s, "^[A-Za-z0-9+/=\\r\\n]+$");
    }
 
    private static string Quote(string s)
    {
        if (s == null) return "null";
        var sb = new StringBuilder(s.Length + 16);
        sb.Append('"');
        for (int i = 0; i < s.Length; i++)
        {
            var c = s[i];
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < 32)
                    {
                        sb.Append("\\u");
                        sb.Append(((int)c).ToString("x4"));
                    }
                    else
                    {
                        sb.Append(c);
                    }
                    break;
            }
        }
        sb.Append('"');
        return sb.ToString();
    }
 
    private static string UnescapeJsonString(string s)
    {
        if (s == null) return null;
        return s.Replace("\\/", "/")
            .Replace("\\\"", "\"")
            .Replace("\\\\", "\\")
            .Replace("\\n", "\n")
            .Replace("\\r", "\r")
            .Replace("\\t", "\t");
    }
 
    private static string BuildJsonObject(Dictionary<string, object> dict)
    {
        var sb = new StringBuilder(1024);
        AppendValue(sb, dict);
        return sb.ToString();
    }
 
    private static void AppendValue(StringBuilder sb, object value)
    {
        switch (value)
        {
            case null:
                sb.Append("null");
                return;
            case string s:
                sb.Append(Quote(s));
                return;
            case bool b:
                sb.Append(b ? "true" : "false");
                return;
            case int or long or float or double or decimal:
                sb.Append(Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture));
                return;
            case Dictionary<string, object> obj:
                sb.Append('{');
                {
                    var first = true;
                    foreach (var kv in obj)
                    {
                        if (!first) sb.Append(',');
                        first = false;
                        sb.Append(Quote(kv.Key));
                        sb.Append(':');
                        AppendValue(sb, kv.Value);
                    }
                }
                sb.Append('}');
                return;
            case IEnumerable<object> arr:
                sb.Append('[');
                {
                    var first = true;
                    foreach (var v in arr)
                    {
                        if (!first) sb.Append(',');
                        first = false;
                        AppendValue(sb, v);
                    }
                }
                sb.Append(']');
                return;
            default:
                sb.Append(Quote(value.ToString()));
                return;
        }
    }
}

