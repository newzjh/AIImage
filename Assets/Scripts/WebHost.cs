// Assets/Scripts/blockly/UwbBlocklyHost.cs
using UnityEngine;
using System.IO;
using Cysharp.Threading.Tasks;
using System.Text;
using System.Collections.Generic;
using System;
using System.Linq;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
using ZenFulcrum.EmbeddedBrowser;
#else
#endif

[DisallowMultipleComponent]
public class WebHost : MonoBehaviour
{
    public UnityEngine.UI.Image blank;

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
    public Browser browser;
#else
    public UniWebView browser2;
    private bool _uniWebViewHooksInstalled;
#endif



    // ��δ ready / connected ʱҪ���ص� URL��������ַ��
    private string _pendingUrl;


    public bool IsWebViewVisible { get; private set; }

    private void Awake()
    {
        IsWebViewVisible = false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        TrySetupZFBrowserProfile();
        browser = GetComponentInChildren<Browser>(true);
        if (browser == null)
        {
            Debug.LogError("[WebBrowser] δ��ȡ�� browser��");
            return;
        }
        browser.enabled = true;
        var pointerugui = GetComponentInChildren<PointerUIGUI>(true);
        if (pointerugui)
            pointerugui.enabled = true;

        _pendingUrl = BuildInitialUrl();
        browser.Url = _pendingUrl;
#else
        browser2 = GetComponentInChildren<UniWebView>(true);
        if (browser2 == null)
        {
            Debug.LogError("[WebBrowser] δ��ȡ�� browser��");
            return;
        }

        float barHeight = Screen.height*0.15f;
        browser2.Frame = new Rect(0,barHeight,Screen.width, Screen.height-barHeight);
        browser2.enabled=true;

        _pendingUrl = BuildInitialUrl();
        RestoreUniWebViewCookies(_pendingUrl);
        browser2.Load(_pendingUrl);
#endif
    }

    private async void Start()
    {

        await RegisterFunctions();

    }

    private async UniTask RegisterFunctions()
    {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        browser = GetComponentInChildren<Browser>(true);
        if (browser == null)
        {
            Debug.LogError("[UWB] δ��ȡ�� browser��");
            return;
        }

        try
        {
            await UniTask.WaitUntil(() => browser.IsReady);
            Debug.Log("browser.IsReady");

            //browser.RegisterFunction("sendBatchedDataX", (args) =>
            //{
            //    string arg0 = args[0].Value.ToString();
            //    string arg1 = args[1].Value.ToString();
            //    if (arg0 == "OnProgramJson")
            //        OnProgramJsonFinish(DecodeBase64FromJs(arg1));
            //    else if (arg0 == "OnProgramXml")
            //        OnProgramXmlFinish(DecodeBase64FromJs(arg1));
            //});
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning("[UWB] RegisterJsMethod ʧ�ܣ�" + ex);
        }
#else
        browser2 = GetComponentInChildren<UniWebView>(true);
        if (browser2 == null)
        {
            Debug.LogError("[WebBrowser] δ��ȡ�� browser��");
            return;
        }
        // ע��ǰ�˻ص�
        try
        {
            await UniTask.WaitUntil(() => browser2.isActiveAndEnabled);
            Debug.Log("browser2.isActiveAndEnabled");

            if (!_uniWebViewHooksInstalled)
            {
                _uniWebViewHooksInstalled = true;
                browser2.OnPageFinished += OnUniWebViewPageFinished;
            }

            //browser2.OnMessageReceived += (UniWebView webView, UniWebViewMessage message) =>
            //{ 
            //    if (message.Args.ContainsKey("OnProgramJson"))
            //        OnProgramJsonFinish(DecodeBase64FromJs(message.Args["OnProgramJson"]));
            //    if (message.Args.ContainsKey("OnProgramXml"))
            //        OnProgramXmlFinish(DecodeBase64FromJs(message.Args["OnProgramXml"]));
            //};
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning("[UWB] RegisterJsMethod ʧ�ܣ�" + ex);
        }
#endif
    }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
    private void TrySetupZFBrowserProfile()
    {
        try
        {
            if (!string.IsNullOrEmpty(BrowserNative.ProfilePath))
                return;

            var profilePath = Path.Combine(Application.persistentDataPath, "ZFBrowserProfile");
            Directory.CreateDirectory(profilePath);
            BrowserNative.ProfilePath = profilePath;
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[ZFBrowser] ProfilePath ����ʧ�ܣ�" + ex);
        }
    }
#else
    private static string GetCookieStoreKey(string url)
    {
        try
        {
            var u = new Uri(url);
            return "WebHost.UniWebView.Cookies." + u.Host.ToLowerInvariant();
        }
        catch
        {
            return "WebHost.UniWebView.Cookies";
        }
    }

    private static void RestoreUniWebViewCookies(string url)
    {
        var key = GetCookieStoreKey(url);
        var cookieString = PlayerPrefs.GetString(key, "");
        if (string.IsNullOrWhiteSpace(cookieString))
            return;

        string baseUrl;
        try
        {
            var u = new Uri(url);
            baseUrl = u.Scheme + "://" + u.Host + "/";
        }
        catch
        {
            baseUrl = url;
        }

        var parts = cookieString.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var p in parts)
        {
            var s = p.Trim();
            if (string.IsNullOrEmpty(s)) continue;
            if (!s.Contains("=")) continue;
            UniWebView.SetCookie(baseUrl, s + "; path=/", true);
        }
    }

    private void OnUniWebViewPageFinished(UniWebView webView, int statusCode, string url)
    {
        if (webView == null) return;

        webView.EvaluateJavaScript("document.cookie", payload =>
        {
            if (payload == null) return;
            if (payload.resultCode != "0") return;

            var key = GetCookieStoreKey(url);
            var cookieString = payload.data ?? "";
            if (string.IsNullOrWhiteSpace(cookieString)) return;

            PlayerPrefs.SetString(key, cookieString);
            PlayerPrefs.Save();
        });
    }
#endif

    // ת��JavaScript�ַ���
    private string EscapeJavaScriptString(string str)
    {
        return str.Replace("\\", "\\\\")
                 .Replace("'", "\\'")
                 .Replace("\"", "\\\"")
                 .Replace("\r", "\\r")
                 .Replace("\n", "\\n")
                 .Replace("\t", "\\t");
    }

    // ������� Ready & Connected ���ټ��� _pendingUrl�������ԣ�
    private async UniTask Co_WaitReadyAndLoadInitial()
    {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        if (browser == null)
            return;

        //if (string.IsNullOrEmpty(_pendingUrl))
        //    return;

        //if (browser.IsReady)
        //    browser.LoadURL(_pendingUrl,true);

        //var xmls = "<xml xmlns=\"https://developers.google.com/blockly/xml\">\n  <block type=\"when_start\" id=\"w%-,~@Xj=xxSj=ur`UEG\" deletable=\"false\" x=\"1216\" y=\"311\">\n    <statement name=\"DO\">\n      <block type=\"move_forward_time\" id=\"SKyeC=cuPh,.2kBGu_w+\">\n        <field name=\"DIR\">fwd</field>\n        <value name=\"SPEED\">\n          <block type=\"math_number\" id=\"F!To[IE#{$Nk[N0,3Pm.\">\n            <field name=\"NUM\">50</field>\n          </block>\n        </value>\n        <value name=\"SEC\">\n          <block type=\"math_number\" id=\"fR*RHa;^*OVYcyg{PmB=\">\n            <field name=\"NUM\">1</field>\n          </block>\n        </value>\n        <next>\n          <block type=\"stop_motion\" id=\"t:VJmlD)evit!)/OGJSH\"></block>\n        </next>\n      </block>\n    </statement>\n  </block>\n</xml>";
        //string JSXml = EscapeJavaScriptString(xmls);
        //string loadXmlScript = $"loadBlocklyXml('{JSXml}')";
        //browser.EvalJS(loadXmlScript);

        await UniTask.WaitUntil(() => browser.IsLoaded);
        Debug.Log("browser.IsLoaded");
#endif
    }

    public bool IsLoaded
    {
        get
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (browser)
                return browser.IsLoaded;
#else
            if (browser2)
                return browser2.isActiveAndEnabled;
#endif
            else
                return false;
        }
    }

    // �������ù�����ʼ URL
    private string BuildInitialUrl()
    {
        //string rel = "python/index.html";
        //return BuildLocalUrl(rel);
        return "http://www.doubao.com/";
    }

    // �� StreamingAssets ���·��ת�� file:// URL
    private string BuildLocalUrl(string rel)
    {
        string path = Path.Combine(Application.streamingAssetsPath, rel).Replace("\\", "/");
        if (!path.StartsWith("file://"))
            path = "file://" + path;
        return path;
    }

    public async UniTask WaitUntilPageReady()
    {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        if (browser == null) return;
        await UniTask.WaitUntil(() => browser.IsReady);
        await UniTask.WaitUntil(() => browser.IsLoaded);
#else
        if (browser2 == null) return;
        await UniTask.WaitUntil(() => browser2.isActiveAndEnabled);
        await UniTask.NextFrame();
#endif
    }

    /// <summary>
    /// �л��� StreamingAssets �µ�ĳ������ҳ��
    /// </summary>
    private async void LoadLocalPage(string rel)
    {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        if (browser == null)
        {
            Debug.LogError("[UWB] browser ��û׼���ã������л�ҳ�档");
            return;
        }

        string url = BuildLocalUrl(rel);

        // �����û ReadySignalReceived�����ȼ�¼ pending����Э��ȥ���� LoadUrl
        if (!browser.IsReady)
        {
            _pendingUrl = url;
            Debug.Log("[UWB] �����δ��������¼ pending URL: " + url);
            return;
        }
#else
        if (browser2 == null)
        {
            Debug.LogError("[UWB] browser ��û׼���ã������л�ҳ�档");
            return;
        }

        string url = BuildLocalUrl(rel);

        // �����û ReadySignalReceived�����ȼ�¼ pending����Э��ȥ���� LoadUrl
        if (!browser2.isActiveAndEnabled)
        {
            _pendingUrl = url;
            Debug.Log("[UWB] �����δ��������¼ pending URL: " + url);
            return;
        }
#endif

        try
        {
            Debug.Log("[UWB] �л�ҳ��: " + url);
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            browser.LoadURL(url, true);
#else
            browser2.Load(url);
#endif
            await RegisterFunctions();
        }
        catch (System.Exception ex)
        {
            Debug.LogError("[UWB] �л�ҳ��ʧ��(���ɻָ�): " + ex);
        }
    }

    public void Show()
    {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        IsWebViewVisible = true;
#else
    if (browser2 == null)
    {
        Debug.LogError("[UWB] browser ��û׼���ã������л�ҳ�档");
        return;
    }
    browser2.Show();
    IsWebViewVisible = true;
#endif
    }

    public void Hide()
    {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        IsWebViewVisible = false;
#else
    if (browser2 == null)
    {
        Debug.LogError("[UWB] browser ��û׼���ã������л�ҳ�档");
        return;
    }
    browser2.Hide();
    IsWebViewVisible = false;
#endif
    }

    private void OnDisable()
    {
        IsWebViewVisible = false;
        var raw = GetComponent<UnityEngine.UI.RawImage>();
        if (raw != null)
            raw.enabled = false;

#if !UNITY_EDITOR_WIN && !UNITY_STANDALONE_WIN
        if (browser2 != null)
            browser2.Hide();
#endif
    }

    private void Update()
    {
 #if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        if (browser == null) return;
        if (!browser.focusState.hasKeyboardFocus && !browser.focusState.focusedNodeEditable) return;

        var ctrl = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);
        if (!ctrl) return;

        if (Input.GetKeyDown(KeyCode.V))
        {
            browser.SendFrameCommand(BrowserNative.FrameCommand.Paste);
        }
        else if (Input.GetKeyDown(KeyCode.C))
        {
            browser.SendFrameCommand(BrowserNative.FrameCommand.Copy);
            browser.EvalJS(@"
(function(){
  try{
    var el = document.activeElement;
    if (el && (el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && (el.type === 'text' || el.type === 'search' || el.type === 'url' || el.type === 'email' || el.type === 'tel' || el.type === 'password')))) {
      var start = el.selectionStart || 0;
      var end = el.selectionEnd || 0;
      return (el.value || '').substring(start, end);
    }
  } catch(e) {}
  try{
    var s = window.getSelection ? window.getSelection().toString() : '';
    return s || '';
  } catch(e) {}
  return '';
})()
").Then(ret =>
            {
                try
                {
                    var txt = ret != null ? ret.Value as string : "";
                    if (!string.IsNullOrEmpty(txt))
                        GUIUtility.systemCopyBuffer = txt;
                }
                catch
                {
                }
            });
        }
 #endif
    }

    /// <summary>
    /// Unity����JS��Base64�ַ��� �� ����ΪUTF-8��ʽ����ͨ�ַ���
    /// </summary>
    /// <param name="base64Str">JS���ݵ�Base64�����ַ���</param>
    /// <returns>������UTF-8�ַ�������Blockly��XML�������ı��ȣ�</returns>
    public static string DecodeBase64FromJs(string base64Str)
    {
        // ��ֵ/���ַ���У��
        if (string.IsNullOrWhiteSpace(base64Str))
        {
            Debug.LogError("JS���ݵ�Base64�ַ���Ϊ�գ�");
            return string.Empty;
        }

        try
        {
            // ����1��Base64�ַ��� �� UTF-8�ֽ�����
            byte[] utf8Bytes = Convert.FromBase64String(base64Str);

            // ����2��UTF-8�ֽ����� �� ��ͨ�ַ���
            string result = Encoding.UTF8.GetString(utf8Bytes);

            Debug.Log($"Base64����ɹ���ԭʼ�ַ�����{result}");
            return result;
        }
        catch (FormatException ex)
        {
            Debug.LogError($"Base64��ʽ���󣨷ǺϷ�Base64������{ex.Message}");
            return string.Empty;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Base64����ʧ�ܣ�{ex.Message}");
            return string.Empty;
        }
    }




    private void OnProgramJsonFinish(string json)
    {

    }



    private void OnProgramXmlFinish(string xml)
    {
        string preview = (xml != null && xml.Length > 120) ? xml.Substring(0, 120) + "..." : xml;
        Debug.Log("[UWB] �յ� Blockly XML��Ԥ��120����" + preview);

    }

}
