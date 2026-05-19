using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Rendering;
 
public sealed class Image2ImageAI : MonoBehaviour
{
    public enum Provider
    {
        GoogleAIStudio,
        Replicate,
        AliTongyiWanxiang,
        Doubao
    }
 
    public event Func<IReadOnlyList<Texture2D>, UniTask<int>> SelectResultIndex;
    public event Action<string> RequestError;

    [SerializeField]
    public ComputeShader imageProcessingCS;
    private static int s_resizeKernel = -1;
    private static int s_removeWatermarkKernel = -1;


    [SerializeField] private Provider provider = Provider.Replicate;

    public Provider CurrentProvider
    {
        get => provider;
        set => provider = value;
    }

    public string GetApiKeyForProvider(Provider p)
    {
        return p switch
        {
            Provider.GoogleAIStudio => googleApiKey,
            Provider.Replicate => replicateApiToken,
            Provider.AliTongyiWanxiang => dashScopeApiKey,
            Provider.Doubao => doubaoApiKey,
            _ => ""
        };
    }

    public void SetApiKeyForProvider(Provider p, string key)
    {
        key ??= "";
        switch (p)
        {
            case Provider.GoogleAIStudio:
                googleApiKey = key;
                break;
            case Provider.Replicate:
                replicateApiToken = key;
                break;
            case Provider.AliTongyiWanxiang:
                dashScopeApiKey = key;
                break;
            case Provider.Doubao:
                doubaoApiKey = key;
                break;
        }
    }
 
    [Header("Google AI Studio (Gemini)")]
    [SerializeField] private string googleApiKey;
    public const string googleModel = "gemini-2.5-flash-image";
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
    [SerializeField] private string doubaoApiKey;
    [SerializeField] private string doubaoBaseUrl = "https://ark.cn-beijing.volces.com/api/v3";
    public const string doubaoModel = "doubao-seedream-4-5-251128";

    [Header("Runtime")]
    [SerializeField] private int timeoutSeconds = 180;
    [SerializeField] private int pollIntervalMs = 900;

    [Header("Post Process")]
    [SerializeField] private bool enableRemoveWatermark = false;
    [SerializeField] private Vector4 watermarkRect = new Vector4(0.72f, 0.86f, 0.26f, 0.12f);
    [SerializeField] private int watermarkRadius = 6;
 
    public UniTask<Texture2D> ImageToImageAsync(
        Texture2D input,
        string prompt,
        int targetWidth,
        int targetHeight,
        CancellationToken cancellationToken)
    {
        if (input == null) return UniTask.FromResult<Texture2D>(null);
        return ImageToImageAsync(new List<Texture2D> { input }, prompt, targetWidth, targetHeight, cancellationToken);
    }

    public async UniTask<Texture2D> ImageToImageAsync(
        IReadOnlyList<Texture2D> referenceImages,
        string prompt,
        int targetWidth,
        int targetHeight,
        CancellationToken cancellationToken)
    {
        if (referenceImages == null || referenceImages.Count == 0) return null;
        if (string.IsNullOrWhiteSpace(prompt)) return null;

        var refs = new List<Texture2D>(referenceImages.Count);
        for (var i = 0; i < referenceImages.Count; i++)
        {
            if (referenceImages[i] != null)
                refs.Add(referenceImages[i]);
        }
        if (refs.Count == 0) return null;

        List<Texture2D> raws;
        switch (provider)
        {
            case Provider.GoogleAIStudio:
                raws = await GoogleGeminiImageEditAsync(refs, prompt, cancellationToken);
                break;
            case Provider.Replicate:
                raws = await ReplicateImageToImageAsync(refs, prompt, cancellationToken);
                break;
            case Provider.AliTongyiWanxiang:
                raws = await DashScopeImageToImageAsync(refs, prompt, cancellationToken);
                break;
            case Provider.Doubao:
                raws = await DoubaoImageToImageAsync(refs, prompt, cancellationToken);
                break;
            default:
                return null;
        }

        if (raws == null || raws.Count == 0) return null;

        var selectedIndex = 0;
        if (raws.Count > 1 && SelectResultIndex != null)
        {
            try
            {
                selectedIndex = await SelectResultIndex.Invoke(raws);
            }
            catch
            {
                selectedIndex = 0;
            }
        }
        selectedIndex = Mathf.Clamp(selectedIndex, 0, raws.Count - 1);

        var selected = raws[selectedIndex];
        for (var i = 0; i < raws.Count; i++)
        {
            if (i == selectedIndex) continue;
            if (raws[i] != null) Destroy(raws[i]);
        }

        if (selected == null) return null;

        var needResize = targetWidth > 0 && targetHeight > 0 && (selected.width != targetWidth || selected.height != targetHeight);
        if (needResize || enableRemoveWatermark)
        {
            var resized = await ResizeAndPostProcessAsync(selected, targetWidth, targetHeight, cancellationToken);
            Destroy(selected);
            selected = resized;
        }

        if (selected != null)
            selected.name = "AI_" + provider;
        return selected;
    }

    private async UniTask<List<Texture2D>> ReplicateImageToImageAsync(IReadOnlyList<Texture2D> referenceImages, string prompt, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(replicateApiToken)) return null;
        //if (string.IsNullOrWhiteSpace(replicateVersion)) return null;
 
        var png = EncodePng(referenceImages[0]);
        if (png == null || png.Length == 0) return null;
 
        var refs = new List<object>();
        for (var i = 0; i < referenceImages.Count; i++)
        {
            var b = EncodePng(referenceImages[i]);
            if (b == null || b.Length == 0) continue;
            refs.Add("data:image/png;base64," + Convert.ToBase64String(b));
        }
 
        var inputObj = new Dictionary<string, object>
        {
            ["prompt"] = prompt,
            ["input_images"] = refs,
            ["resolution"] = "2 MP",
            ["output_format"] = "png",
            ["safety_tolerance"] = 2
        };

        //var requestJson = BuildJsonObject(new Dictionary<string, object>
        //{
        //    ["version"] = replicateVersion,
        //    ["input"] = inputObj
        //});

        var requestJson = BuildJsonObject(new Dictionary<string, object>
        {
            ["input"] = inputObj
        });

        //var createUrl = "https://api.replicate.com/v1/predictions";
        var createUrl = "https://api.replicate.com/v1/models/black-forest-labs/flux-2-pro/predictions";
        
        var created = await SendJsonAsync(createUrl, "POST", requestJson, new Dictionary<string, string>
        {
            ["Authorization"] = "Bearer " + replicateApiToken
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
                ["Authorization"] = "Bearer " + replicateApiToken
            }, ct);
 
            if (string.IsNullOrWhiteSpace(polled))
                return null;
 
            var status = ExtractJsonString(polled, "status");
            if (string.Equals(status, "succeeded", StringComparison.OrdinalIgnoreCase))
            {
                var urls = ExtractAllUrlsFromJson(polled);
                if (urls.Count == 0)
                    return ExtractAllImagesFromJson(polled);

                var results = new List<Texture2D>(urls.Count);
                for (var i = 0; i < urls.Count; i++)
                {
                    var tex = await DownloadTextureAsync(urls[i], ct);
                    if (tex != null) results.Add(tex);
                }
                return results;
            }
 
            if (string.Equals(status, "failed", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(status, "canceled", StringComparison.OrdinalIgnoreCase))
                return null;
 
            await UniTask.Delay(pollIntervalMs, cancellationToken: ct);
        }
 
        return null;
    }
 
    private async UniTask<List<Texture2D>> DashScopeImageToImageAsync(IReadOnlyList<Texture2D> referenceImages, string prompt, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(dashScopeApiKey)) return null;
 
        var png = EncodePng(referenceImages[0]);
        if (png == null || png.Length == 0) return null;
 
        var imageBase64 = Convert.ToBase64String(png);
        var extraRefs = new List<object>();
        for (var i = 1; i < referenceImages.Count; i++)
        {
            var b = EncodePng(referenceImages[i]);
            if (b == null || b.Length == 0) continue;
            extraRefs.Add("data:image/png;base64," + Convert.ToBase64String(b));
        }
 
        var url = dashScopeBaseUrl.TrimEnd('/') + "/api/v1/services/aigc/image2image/image-synthesis";
 
        var inputObj = new Dictionary<string, object>
        {
            ["prompt"] = prompt,
            ["img"] = "data:image/png;base64," + imageBase64
        };
        if (extraRefs.Count > 0)
            inputObj["ref_images"] = extraRefs;

        var requestJson = BuildJsonObject(new Dictionary<string, object>
        {
            ["model"] = dashScopeModel,
            ["input"] = inputObj,
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
 
        var urls0 = ExtractAllUrlsFromJson(resp);
        if (urls0.Count > 0)
            return await DownloadAllTexturesAsync(urls0, ct);
        var imgs0 = ExtractAllImagesFromJson(resp);
        if (imgs0.Count > 0)
            return imgs0;
 
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
                var urls = ExtractAllUrlsFromJson(polled);
                if (urls.Count > 0)
                    return await DownloadAllTexturesAsync(urls, ct);

                var imgs = ExtractAllImagesFromJson(polled);
                if (imgs.Count > 0)
                    return imgs;

                return null;
            }
 
            if (string.Equals(status, "FAILED", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(status, "failed", StringComparison.OrdinalIgnoreCase))
                return null;
 
            await UniTask.Delay(pollIntervalMs, cancellationToken: ct);
        }
 
        return null;
    }
 
    private async UniTask<List<Texture2D>> GoogleGeminiImageEditAsync(IReadOnlyList<Texture2D> referenceImages, string prompt, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(googleApiKey)) return null;
        if (string.IsNullOrWhiteSpace(googleModel)) return null;
 
        var parts = new StringBuilder(1024);
        parts.Append("{\"text\":");
        parts.Append(Quote(prompt));
        parts.Append('}');

        for (var i = 0; i < referenceImages.Count; i++)
        {
            var png = EncodePng(referenceImages[i]);
            if (png == null || png.Length == 0) continue;
            var base64 = Convert.ToBase64String(png);
            parts.Append(",{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":");
            parts.Append(Quote(base64));
            parts.Append("}}");
        }

        var url = googleBaseUrl.TrimEnd('/') + "/models/" + googleModel + ":generateContent?key=" + UnityWebRequest.EscapeURL(googleApiKey);
 
        var requestJson = "{\"contents\":[{\"role\":\"user\",\"parts\":[" + parts + "]}]}";
 
        var resp = await SendJsonAsync(url, "POST", requestJson, null, ct);
        if (string.IsNullOrWhiteSpace(resp))
            return null;
 
        var b64s = ExtractAllGeminiInlineImageBase64(resp);
        if (b64s.Count > 0)
        {
            var imgs = new List<Texture2D>(b64s.Count);
            for (var i = 0; i < b64s.Count; i++)
            {
                var tex = DecodeTextureFromBase64(b64s[i]);
                if (tex != null) imgs.Add(tex);
            }
            return imgs;
        }

        var urls = ExtractAllUrlsFromJson(resp);
        if (urls.Count > 0)
            return await DownloadAllTexturesAsync(urls, ct);

        return null;
    }
 
    private async UniTask<List<Texture2D>> DoubaoImageToImageAsync(IReadOnlyList<Texture2D> referenceImages, string prompt, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(doubaoApiKey)) return null;
 
        var png = EncodePng(referenceImages[0]);
        if (png == null || png.Length == 0) return null;
 
        var base64 = Convert.ToBase64String(png);
        var dataUrl = "data:image/png;base64," + base64;
        var extraRefs = new List<object>();
        for (var i = 1; i < referenceImages.Count; i++)
        {
            var b = EncodePng(referenceImages[i]);
            if (b == null || b.Length == 0) continue;
            extraRefs.Add("data:image/png;base64," + Convert.ToBase64String(b));
        }

        var url = doubaoBaseUrl.TrimEnd('/') + "/images/generations";
        var reqObj = new Dictionary<string, object>
        {
            ["model"] = doubaoModel,
            ["prompt"] = prompt,
            ["image"] = dataUrl,
            ["size"] = "2k",
            ["sequential_image_generation"] = "disabled",
            ["watermark"] = false,
            ["response_format"] = "b64_json",
        };
        if (extraRefs.Count > 0)
            reqObj["reference_images"] = extraRefs;

        var requestJson = BuildJsonObject(reqObj);
 
        var resp = await SendJsonAsync(url, "POST", requestJson, new Dictionary<string, string>
        {
            ["Authorization"] = "Bearer " + doubaoApiKey
        }, ct);
 
        if (string.IsNullOrWhiteSpace(resp))
            return null;
 
        var imgs = ExtractAllImagesFromJson(resp);
        if (imgs.Count > 0)
            return imgs;

        var urls = ExtractAllUrlsFromJson(resp);
        if (urls.Count > 0)
            return await DownloadAllTexturesAsync(urls, ct);

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
 
    private async UniTask<Texture2D> ResizeAndPostProcessAsync(Texture2D src, int targetWidth, int targetHeight, CancellationToken ct)
    {
        if (src == null) return null;

        var dstW = targetWidth > 0 ? targetWidth : src.width;
        var dstH = targetHeight > 0 ? targetHeight : src.height;

        var compute = GetImageProcessingCompute();

        if (compute != null && s_resizeKernel < 0)
        {
            try { s_resizeKernel = compute.FindKernel("ResizeBilinear"); }
            catch { s_resizeKernel = -1; }
        }

        if (compute != null && enableRemoveWatermark && s_removeWatermarkKernel < 0)
        {
            try { s_removeWatermarkKernel = compute.FindKernel("RemoveWatermark"); }
            catch { s_removeWatermarkKernel = -1; }
        }

        var rt0 = new RenderTexture(dstW, dstH, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        rt0.enableRandomWrite = true;
        rt0.wrapMode = TextureWrapMode.Clamp;
        rt0.filterMode = FilterMode.Bilinear;
        rt0.Create();

        RenderTexture rt1 = null;
        RenderTexture finalRt = rt0;

        try
        {
            ct.ThrowIfCancellationRequested();

            if (compute != null && s_resizeKernel >= 0)
            {
                compute.SetTexture(s_resizeKernel, "_Source", src);
                compute.SetTexture(s_resizeKernel, "_Result", rt0);
                compute.SetVector("_SrcSize", new Vector4(src.width, src.height, 0f, 0f));
                compute.SetVector("_DstSize", new Vector4(dstW, dstH, 0f, 0f));

                var gx = Mathf.CeilToInt(dstW / 8f);
                var gy = Mathf.CeilToInt(dstH / 8f);
                compute.Dispatch(s_resizeKernel, Mathf.Max(1, gx), Mathf.Max(1, gy), 1);
            }
            else
            {
                Graphics.Blit(src, rt0);
            }

            if (compute != null && enableRemoveWatermark && s_removeWatermarkKernel >= 0)
            {
                rt1 = new RenderTexture(dstW, dstH, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
                rt1.enableRandomWrite = true;
                rt1.wrapMode = TextureWrapMode.Clamp;
                rt1.filterMode = FilterMode.Bilinear;
                rt1.Create();

                compute.SetTexture(s_removeWatermarkKernel, "_Source", rt0);
                compute.SetTexture(s_removeWatermarkKernel, "_Result", rt1);
                compute.SetVector("_WatermarkRect", watermarkRect);
                compute.SetInt("_Radius", Mathf.Max(2, watermarkRadius));

                var gx = Mathf.CeilToInt(dstW / 8f);
                var gy = Mathf.CeilToInt(dstH / 8f);
                compute.Dispatch(s_removeWatermarkKernel, Mathf.Max(1, gx), Mathf.Max(1, gy), 1);

                finalRt = rt1;
            }

            var r = await RequestReadback(finalRt);
            if (r.hasError)
                return null;
            if (ct.IsCancellationRequested)
                return null;

            var data = r.GetData<byte>();
            var tex = new Texture2D(dstW, dstH, TextureFormat.RGBA32, false);
            tex.LoadRawTextureData(data);
            tex.Apply(false, false);
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            return tex;
        }
        finally
        {
            if (rt1 != null)
            {
                rt1.Release();
                Destroy(rt1);
            }
            rt0.Release();
            Destroy(rt0);
        }
    }

    private static UniTask<AsyncGPUReadbackRequest> RequestReadback(RenderTexture rt)
    {
        var tcs = new UniTaskCompletionSource<AsyncGPUReadbackRequest>();
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, r => tcs.TrySetResult(r));
        return tcs.Task;
    }

    private ComputeShader GetImageProcessingCompute()
    {
        if (imageProcessingCS != null) return imageProcessingCS;
        try
        {
            imageProcessingCS = Resources.Load<ComputeShader>("ImageProcessing");
        }
        catch
        {
            imageProcessingCS = null;
        }
        return imageProcessingCS;
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
            try
            {
                await req.SendWebRequest();
            }
            catch (Exception ex)
            {
                Debug.LogError(ex);
                var body = req.downloadHandler?.text;
                var msg = BuildRequestErrorMessage(method, url, req.responseCode, ex.ToString(), body);
                try { RequestError?.Invoke(msg); } catch { }
                return body;
            }
            if (req.result != UnityWebRequest.Result.Success)
            {
                var body = req.downloadHandler?.text;
                var msg = BuildRequestErrorMessage(method, url, req.responseCode, req.error, body);
                try { RequestError?.Invoke(msg); } catch { }
                return body;
            }
            if (req.responseCode >= 400)
            {
                var body = req.downloadHandler?.text;
                var msg = BuildRequestErrorMessage(method, url, req.responseCode, req.error, body);
                try { RequestError?.Invoke(msg); } catch { }
                return body;
            }
            return req.downloadHandler?.text;
        }
    }

    private string BuildRequestErrorMessage(string method, string url, long code, string error, string body)
    {
        var p = provider.ToString();
        var sb = new StringBuilder(256);
        sb.Append(p);
        sb.Append(" request failed: ");
        sb.Append(method);
        sb.Append(' ');
        sb.Append(ShortenUrl(url));
        if (code > 0)
        {
            sb.Append(" (");
            sb.Append(code);
            sb.Append(')');
        }
        if (!string.IsNullOrWhiteSpace(error))
        {
            sb.Append(' ');
            sb.Append(error);
        }
        var snippet = ExtractErrorSnippet(body);
        if (!string.IsNullOrWhiteSpace(snippet))
        {
            sb.Append(" | ");
            sb.Append(snippet);
        }
        return sb.ToString();
    }

    private static string ShortenUrl(string url)
    {
        if (string.IsNullOrWhiteSpace(url))
            return "";
        if (url.Length <= 70)
            return url;
        return url.Substring(0, 62) + "…";
    }

    private static string ExtractErrorSnippet(string body)
    {
        if (string.IsNullOrWhiteSpace(body))
            return "";
        var s = body.Replace("\r", " ").Replace("\n", " ").Trim();
        if (s.Length <= 120)
            return s;
        return s.Substring(0, 112) + "…";
    }
 
    private static List<string> ExtractAllUrlsFromJson(string json)
    {
        var list = new List<string>();
        if (string.IsNullOrWhiteSpace(json)) return list;

        foreach (Match m in Regex.Matches(json, "\"url\"\\s*:\\s*\"(?<u>https?:\\\\/\\\\/[^\\\"]+)\"", RegexOptions.IgnoreCase))
        {
            var u = UnescapeJsonString(m.Groups["u"].Value);
            if (!string.IsNullOrWhiteSpace(u) && !list.Contains(u))
                list.Add(u);
        }

        foreach (Match m in Regex.Matches(json, "(?<u>https?:\\\\/\\\\/[^\\\"\\s]+)", RegexOptions.IgnoreCase))
        {
            var u = UnescapeJsonString(m.Groups["u"].Value);
            if (string.IsNullOrWhiteSpace(u)) continue;
            if (u.Contains("schema=")) continue;
            if (!list.Contains(u))
                list.Add(u);
        }

        return list;
    }

    private static async UniTask<List<Texture2D>> DownloadAllTexturesAsync(List<string> urls, CancellationToken ct)
    {
        var results = new List<Texture2D>(urls.Count);
        for (var i = 0; i < urls.Count; i++)
        {
            var tex = await DownloadTextureAsync(urls[i], ct);
            if (tex != null) results.Add(tex);
        }
        return results;
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

    private static List<Texture2D> ExtractAllImagesFromJson(string json)
    {
        var results = new List<Texture2D>();
        if (string.IsNullOrWhiteSpace(json)) return results;

        foreach (Match m in Regex.Matches(json, "\"b64_json\"\\s*:\\s*\"(?<v>(\\\\.|[^\"])*)\"", RegexOptions.IgnoreCase))
        {
            var s = UnescapeJsonString(m.Groups["v"].Value);
            var tex = DecodeTextureFromBase64(s);
            if (tex != null) results.Add(tex);
        }

        foreach (Match m in Regex.Matches(json, "\"data\"\\s*:\\s*\"(?<v>(\\\\.|[^\"])*)\"", RegexOptions.IgnoreCase))
        {
            var s = UnescapeJsonString(m.Groups["v"].Value);
            if (!LooksLikeBase64(s)) continue;
            var tex = DecodeTextureFromBase64(s);
            if (tex != null) results.Add(tex);
        }

        return results;
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

    private static List<string> ExtractAllGeminiInlineImageBase64(string json)
    {
        var list = new List<string>();
        if (string.IsNullOrWhiteSpace(json)) return list;

        foreach (Match m in Regex.Matches(json, "\"inlineData\"\\s*:\\s*\\{[^\\}]*\"mimeType\"\\s*:\\s*\"image\\/[^\\\"]+\"[^\\}]*\"data\"\\s*:\\s*\"(?<v>(\\\\.|[^\"])*)\"", RegexOptions.IgnoreCase))
        {
            var s = UnescapeJsonString(m.Groups["v"].Value);
            if (!string.IsNullOrWhiteSpace(s))
                list.Add(s);
        }

        foreach (Match m in Regex.Matches(json, "\"inline_data\"\\s*:\\s*\\{[^\\}]*\"mime_type\"\\s*:\\s*\"image\\/[^\\\"]+\"[^\\}]*\"data\"\\s*:\\s*\"(?<v>(\\\\.|[^\"])*)\"", RegexOptions.IgnoreCase))
        {
            var s = UnescapeJsonString(m.Groups["v"].Value);
            if (!string.IsNullOrWhiteSpace(s))
                list.Add(s);
        }

        return list;
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
