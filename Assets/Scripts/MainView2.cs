using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.UIElements;

public sealed class MainView2 : BasePageView
{
    private const string PrefKeyMaleFacePath = "MainView.Ref.MaleFacePath";
    private const string PrefKeyFemaleFacePath = "MainView.Ref.FemaleFacePath";
    private const string PrefKeyBackgroundPath = "MainView.Ref.BackgroundPath";
    private const string PrefKeyAIProvider = "MainView.AI.Provider";
    private const string PrefKeyGoogleApiKey = "MainView.AI.GoogleApiKey";
    private const string PrefKeyReplicateApiKey = "MainView.AI.ReplicateApiToken";
    private const string PrefKeyDashScopeApiKey = "MainView.AI.DashScopeApiKey";
    private const string PrefKeyDoubaoApiKey = "MainView.AI.DoubaoApiKey";
    private const string PrefKeyHuggingFaceToken = "MainView.AI.HuggingFaceToken";
    private const string PrefKeyRunwareApiKey = "MainView.AI.RunwareApiKey";
    private const string PrefKeyLumenfallApiKey = "MainView.AI.LumenfallApiKey";

    private static readonly string[] OriginalNameMarkersZh = { "原图", "原始", "原片", "未编辑", "未处理", "直出", "原版" };
    private static readonly string[] OriginalNameMarkersEn = { "original", "originals", "orig", "unedited", "unprocessed", "raw", "source", "camera" };

    private enum ImageOp
    {
        FaceSwap,
        Sharpen,
        Whiten,
        SharpenWhiten,
        ChangeBackground,
        DehazeColorGrade,
        ColorGrade,
        Dehaze
    }

    public override AppPageId PageId => AppPageId.MainView2;
    public string CurrentSourcePathForSync => CurrentImagePath;
    public Texture2D CurrentEditedTextureForSync => GetCurrentHistoryTexture();
    public Texture2D CurrentOriginalTextureForSync => GetOriginalHistoryTexture();
    public string CurrentDisplayLabelForSync => GetCurrentHistoryLabel();

    private Texture2D _maleFaceTexture;
    private Texture2D _femaleFaceTexture;
    private Texture2D _backgroundTexture;
    private string _maleFacePath;
    private string _femaleFacePath;
    private string _backgroundPath;

    private DropdownField _providerDropdown;
    private TextField _apiKeyField;
    private Button _maleFaceButton;
    private Button _femaleFaceButton;
    private Button _backgroundButton;
    private Button _panelToggleButton;
    private VisualElement _adjustPanel;
    private VisualElement _adjustBody;
    private VisualElement _adjustHost;
    private ScrollView _toolbarScroll;
    private ScrollView _presetScroll;
    private bool _adjustPanelCollapsed;
    private bool _appendDeGlarePrompt;
    private bool _appendRemoveBgPeoplePrompt = true;
    private bool _gpuSharpenDumpStages;
    private bool _aiRunning;
    private bool _adjustRunning;
    private static readonly List<string> YoloInpaintResourceSnapshotLines = new List<string>(256);

    private System.Threading.CancellationTokenSource _lifetimeCts;
    private System.Threading.CancellationTokenSource _faceMaskCts;
    private System.Threading.CancellationTokenSource _maleFaceMaskCts;
    private System.Threading.CancellationTokenSource _femaleFaceMaskCts;

    protected override void OnInitialized()
    {
        RestoreReferencePickersFromPrefs();
        RestoreAISettingsFromPrefs();
    }

    protected override void OnShown()
    {
        _lifetimeCts?.Cancel();
        _lifetimeCts?.Dispose();
        _lifetimeCts = new System.Threading.CancellationTokenSource();

        if (Host?.FileDialog != null)
            Host.FileDialog.EnsureInitialized();

        BindAiEvents();
        SyncReferenceButtonState();
        UpdateProviderUi();
        if (GetCurrentHistoryTexture() == null)
        {
            var lastPath = Host?.GetLastImagePath();
            if (!string.IsNullOrWhiteSpace(lastPath) && File.Exists(lastPath))
                LoadImageFromPath(lastPath, true);
        }
        else
        {
            var current = GetCurrentHistoryTexture();
            var original = GetOriginalHistoryTexture();
            CompareView?.SetSources(current, original, GetCurrentHistoryLabel());
            CompareView?.FitToView();
        }
    }

    protected override void OnBeforeDetach()
    {
        UnbindAiEvents();
        CancelAndDisposeCts(ref _lifetimeCts);
        CancelAndDisposeCts(ref _faceMaskCts);
        CancelAndDisposeCts(ref _maleFaceMaskCts);
        CancelAndDisposeCts(ref _femaleFaceMaskCts);
    }

    protected override AppPageId? ResolveSwipeTarget(SwipeDirection direction)
    {
        return direction switch
        {
            SwipeDirection.Left => AppPageId.LibraryView,
            SwipeDirection.Right => AppPageId.DesignView,
            _ => null
        };
    }

    protected override bool UseOverlaySwitchZone => true;

    protected override float GetSwitchPillAlignment01() => 0.5f;

    protected override void OnLayoutChanged(bool isPortrait, Rect layoutRect)
    {
        if (_adjustPanel == null || _adjustHost == null)
            return;

        if (isPortrait)
        {
            _adjustPanel.style.width = Length.Percent(90);
            _adjustPanel.style.maxWidth = 420;
            _adjustPanel.style.right = 5;
            _adjustPanel.style.left = 5;
            _adjustPanel.style.bottom = 18;
            _adjustPanel.style.top = new StyleLength(StyleKeyword.Auto);
            _adjustHost.style.paddingRight = 12;
            SetAdjustPanelCollapsed(_adjustPanelCollapsed || _adjustBody.style.display == DisplayStyle.None, false);
        }
        else
        {
            _adjustPanel.style.width = 346;
            _adjustPanel.style.left = new StyleLength(StyleKeyword.Auto);
            _adjustPanel.style.right = 16;
            _adjustPanel.style.top = 18;
            _adjustPanel.style.bottom = 18;
            _adjustHost.style.paddingRight = 12;
            if (_adjustBody.style.display == DisplayStyle.None)
                SetAdjustPanelCollapsed(false, false);
        }

        _toolbarScroll?.schedule.Execute(() => _toolbarScroll.horizontalScroller.value = 0f);
        _presetScroll?.schedule.Execute(() => _presetScroll.horizontalScroller.value = 0f);
    }

    protected override void BuildPage(VisualElement contentRoot)
    {
        contentRoot.style.flexDirection = FlexDirection.Column;
        contentRoot.style.flexGrow = 1;
        contentRoot.style.minHeight = 0;

        var topBar = BuildTopToolbar();
        contentRoot.Add(topBar);

        _adjustHost = new VisualElement();
        _adjustHost.style.flexGrow = 1;
        _adjustHost.style.minHeight = 0;
        _adjustHost.style.position = Position.Relative;
        _adjustHost.style.paddingLeft = 12;
        _adjustHost.style.paddingRight = 12;
        _adjustHost.style.paddingTop = 8;
        _adjustHost.style.paddingBottom = 0;
        contentRoot.Add(_adjustHost);

        var canvasHost = new VisualElement();
        canvasHost.style.flexGrow = 1;
        canvasHost.style.minHeight = 0;
        canvasHost.style.position = Position.Relative;
        canvasHost.style.backgroundColor = new StyleColor(new Color(0.08f, 0.09f, 0.11f, 1f));
        canvasHost.style.borderTopLeftRadius = 24;
        canvasHost.style.borderTopRightRadius = 24;
        canvasHost.style.borderBottomLeftRadius = 24;
        canvasHost.style.borderBottomRightRadius = 24;
        canvasHost.style.overflow = Overflow.Hidden;
        _adjustHost.Add(canvasHost);

        CreateCompareView(canvasHost, true);
        canvasHost.Add(CreateFloatingHistoryPanel(230f, "主编辑历史"));

        _panelToggleButton = new Button(() => SetAdjustPanelCollapsed(!_adjustPanelCollapsed, true))
        {
            text = "调节"
        };
        _panelToggleButton.style.position = Position.Absolute;
        _panelToggleButton.style.right = 24;
        _panelToggleButton.style.top = 24;
        _panelToggleButton.style.height = 34;
        _panelToggleButton.style.paddingLeft = 16;
        _panelToggleButton.style.paddingRight = 16;
        _panelToggleButton.style.backgroundColor = new StyleColor(new Color(0.07f, 0.41f, 0.78f, 0.92f));
        _panelToggleButton.style.color = Color.white;
        _panelToggleButton.style.borderTopLeftRadius = 17;
        _panelToggleButton.style.borderTopRightRadius = 17;
        _panelToggleButton.style.borderBottomLeftRadius = 17;
        _panelToggleButton.style.borderBottomRightRadius = 17;
        canvasHost.Add(_panelToggleButton);

        _adjustPanel = BuildAdjustPanel();
        canvasHost.Add(_adjustPanel);

        var presetBar = BuildPresetBar();
        presetBar.style.position = Position.Absolute;
        presetBar.style.left = 12;
        presetBar.style.right = 12;
        presetBar.style.bottom = 28;
        _adjustHost.Add(presetBar);
        BuildStandardOverlays();

        SetAdjustPanelCollapsed(IsPortraitLayout, false);
    }

    public bool LoadImageFromPath(string filePath, bool bypassOriginalNameGuard = false)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return false;
        if (!File.Exists(filePath))
            return false;

        var fileName = Path.GetFileName(filePath);
        if (!bypassOriginalNameGuard && (IsOriginalDefinedName(fileName) || IsOriginalDefinedPath(filePath)))
        {
            ShowToast("该文件名被识别为原图标记，请先另存为新文件再编辑", 2800);
            return false;
        }

        var texture = Host?.LoadTexture(filePath, true);
        if (texture == null)
        {
            ShowToast("图片加载失败", 2200);
            return false;
        }

        ResetHistoryWithOriginal(texture, fileName, filePath);
        CompareView?.SetSources(texture, texture, fileName);
        CompareView?.FitToView();
        PrepareFaceMaskForSelectedImageAsync(filePath, texture, _gpuSharpenDumpStages).Forget();
        return true;
    }

    private VisualElement BuildTopToolbar()
    {
        var shell = new VisualElement();
        shell.style.flexShrink = 0;
        shell.style.paddingLeft = 12;
        shell.style.paddingRight = 12;
        shell.style.paddingTop = 10;
        shell.style.paddingBottom = 8;

        _toolbarScroll = new ScrollView(ScrollViewMode.Horizontal);
        _toolbarScroll.style.flexShrink = 0;
        _toolbarScroll.horizontalScrollerVisibility = ScrollerVisibility.Hidden;
        _toolbarScroll.verticalScrollerVisibility = ScrollerVisibility.Hidden;
        shell.Add(_toolbarScroll);

        var row = new VisualElement();
        row.style.flexDirection = FlexDirection.Row;
        row.style.alignItems = Align.Center;
        row.style.height = 52;
        _toolbarScroll.Add(row);

        void AddTool(string title, string icon, Action onClick, Color? tint = null)
        {
            var button = new Button(onClick);
            button.style.width = 54;
            button.style.height = 54;
            button.style.marginRight = 8;
            button.style.paddingLeft = 0;
            button.style.paddingRight = 0;
            button.style.paddingTop = 4;
            button.style.paddingBottom = 4;
            button.style.flexDirection = FlexDirection.Column;
            button.style.alignItems = Align.Center;
            button.style.justifyContent = Justify.Center;
            button.style.backgroundColor = new StyleColor(new Color(0.13f, 0.14f, 0.18f, 1f));
            button.style.borderTopLeftRadius = 16;
            button.style.borderTopRightRadius = 16;
            button.style.borderBottomLeftRadius = 16;
            button.style.borderBottomRightRadius = 16;

            var iconLabel = new Label(icon);
            iconLabel.style.fontSize = 16;
            iconLabel.style.color = tint ?? Color.white;
            iconLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            button.Add(iconLabel);

            var textLabel = new Label(title);
            textLabel.style.fontSize = 10;
            textLabel.style.color = new Color(0.83f, 0.86f, 0.92f, 1f);
            textLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            button.Add(textLabel);
            row.Add(button);
        }

        AddTool("Fit", "⌖", () => CompareView?.FitToView());
        AddTool("重置", "↺", () => CompareView?.ResetView());
        AddTool("保存", "⇩", OnSaveCurrentImage, new Color(0.37f, 0.78f, 1f));
        AddTool("浏览", "▣", OnBrowseOriginalImage);
        AddTool("CLIP", "◎", OnClipClassify, new Color(0.73f, 0.56f, 1f));
        AddTool("换脸", "☺", OnFaceSwap, new Color(0.99f, 0.74f, 0.35f));
        AddTool("清晰", "✦", OnSharpen);
        AddTool("美白", "◌", OnWhiten);
        AddTool("清白", "✧", OnSharpenWhiten);
        AddTool("背景", "▥", OnChangeBackground);
        AddTool("去雾", "≋", OnDehaze);
        AddTool("调色", "◐", OnColorGrade);
        AddTool("去雾调", "◑", OnDehazeColorGrade);
        AddTool("GPU", "⚙", OnGpuSharpen);
        AddTool("ESR", "⤢", OnRealEsrganRepro);
        AddTool("YOLO", "▤", OnYoloAndInpaintingRepro);
        AddTool("抠图", "⌗", OnMattingRepro);
        AddTool("GFP", "◍", OnGfpganRepro);
        AddTool("CF", "◎", OnCodeFormerRepro);
        return shell;
    }

    private VisualElement BuildPresetBar()
    {
        var shell = new VisualElement();
        shell.style.height = 100;

        var frame = new VisualElement();
        frame.style.height = 100;
        frame.style.backgroundColor = new StyleColor(new Color(0.07f, 0.08f, 0.10f, 0.92f));
        frame.style.borderTopLeftRadius = 24;
        frame.style.borderTopRightRadius = 24;
        frame.style.paddingLeft = 10;
        frame.style.paddingRight = 10;
        frame.style.paddingTop = 8;
        frame.style.paddingBottom = 8;
        shell.Add(frame);

        _presetScroll = new ScrollView(ScrollViewMode.Horizontal);
        _presetScroll.horizontalScrollerVisibility = ScrollerVisibility.Hidden;
        _presetScroll.verticalScrollerVisibility = ScrollerVisibility.Hidden;
        frame.Add(_presetScroll);

        var row = new VisualElement();
        row.style.flexDirection = FlexDirection.Row;
        row.style.alignItems = Align.Center;
        row.style.height = 80;
        _presetScroll.Add(row);

        void AddPreset(string id, string title, Color swatch, bool accent = false)
        {
            var button = new Button(() => ApplyPresetAsync(id, title).Forget());
            button.style.width = 78;
            button.style.height = 68;
            button.style.marginRight = 10;
            button.style.paddingLeft = 6;
            button.style.paddingRight = 6;
            button.style.paddingTop = 6;
            button.style.paddingBottom = 6;
            button.style.backgroundColor = accent
                ? new StyleColor(new Color(0.12f, 0.60f, 0.92f, 0.96f))
                : new StyleColor(new Color(0.12f, 0.13f, 0.17f, 0.96f));
            button.style.borderTopLeftRadius = 16;
            button.style.borderTopRightRadius = 16;
            button.style.borderBottomLeftRadius = 16;
            button.style.borderBottomRightRadius = 16;
            button.style.flexDirection = FlexDirection.Column;
            button.style.alignItems = Align.Center;
            button.style.justifyContent = Justify.Center;

            var swatchDot = new VisualElement();
            swatchDot.style.width = 26;
            swatchDot.style.height = 26;
            swatchDot.style.backgroundColor = new StyleColor(swatch);
            swatchDot.style.borderTopLeftRadius = 13;
            swatchDot.style.borderTopRightRadius = 13;
            swatchDot.style.borderBottomLeftRadius = 13;
            swatchDot.style.borderBottomRightRadius = 13;
            button.Add(swatchDot);

            var label = new Label(title);
            label.style.marginTop = 6;
            label.style.fontSize = 11;
            label.style.color = Color.white;
            label.style.unityTextAlign = TextAnchor.MiddleCenter;
            button.Add(label);
            row.Add(button);
        }

        AddPreset("cn", "CN", new Color(0.97f, 0.49f, 0.16f), true);
        AddPreset("clean", "清透", new Color(0.31f, 0.80f, 0.94f));
        AddPreset("warm", "暖调", new Color(0.94f, 0.58f, 0.30f));
        AddPreset("film", "胶片", new Color(0.35f, 0.33f, 0.46f));
        AddPreset("bw", "BW", new Color(0.86f, 0.86f, 0.86f));
        AddPreset("fresh", "清新", new Color(0.32f, 0.80f, 0.63f));
        AddPreset("portrait", "人像", new Color(0.93f, 0.56f, 0.67f));
        AddPreset("night", "夜景", new Color(0.42f, 0.50f, 0.93f));
        return shell;
    }

    private VisualElement BuildAdjustPanel()
    {
        var panel = new VisualElement();
        panel.style.position = Position.Absolute;
        panel.style.width = 346;
        panel.style.right = 16;
        panel.style.top = 18;
        panel.style.bottom = 18;
        panel.style.backgroundColor = new StyleColor(new Color(0.12f, 0.13f, 0.17f, 0.96f));
        panel.style.borderTopLeftRadius = 24;
        panel.style.borderTopRightRadius = 24;
        panel.style.borderBottomLeftRadius = 24;
        panel.style.borderBottomRightRadius = 24;
        panel.style.borderLeftWidth = 1;
        panel.style.borderTopWidth = 1;
        panel.style.borderRightWidth = 1;
        panel.style.borderBottomWidth = 1;
        panel.style.borderLeftColor = new StyleColor(new Color(1f, 1f, 1f, 0.08f));
        panel.style.borderTopColor = new StyleColor(new Color(1f, 1f, 1f, 0.08f));
        panel.style.borderRightColor = new StyleColor(new Color(1f, 1f, 1f, 0.08f));
        panel.style.borderBottomColor = new StyleColor(new Color(1f, 1f, 1f, 0.08f));
        panel.style.paddingLeft = 12;
        panel.style.paddingRight = 12;
        panel.style.paddingTop = 12;
        panel.style.paddingBottom = 12;
        panel.style.flexDirection = FlexDirection.Column;
        panel.style.minHeight = 0;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.marginBottom = 8;
        panel.Add(header);

        var title = new Label("调节与参考");
        title.style.flexGrow = 1;
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        title.style.fontSize = 15;
        header.Add(title);

        var collapseButton = new Button(() => SetAdjustPanelCollapsed(!_adjustPanelCollapsed, true))
        {
            text = "—"
        };
        collapseButton.style.width = 32;
        collapseButton.style.height = 32;
        collapseButton.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.08f));
        collapseButton.style.color = Color.white;
        collapseButton.style.borderTopLeftRadius = 16;
        collapseButton.style.borderTopRightRadius = 16;
        collapseButton.style.borderBottomLeftRadius = 16;
        collapseButton.style.borderBottomRightRadius = 16;
        header.Add(collapseButton);
        EnableFloatingPanelDrag(panel, header);

        var scroll = new ScrollView(ScrollViewMode.Vertical);
        scroll.style.flexGrow = 1;
        scroll.style.minHeight = 0;
        panel.Add(scroll);
        _adjustBody = scroll;

        scroll.Add(BuildReferenceButtons());
        scroll.Add(BuildProviderEditor());
        scroll.Add(BuildToggleRow("去反光", value => _appendDeGlarePrompt = value, false));
        scroll.Add(BuildToggleRow("去背景人物", value => _appendRemoveBgPeoplePrompt = value, true));
        scroll.Add(BuildToggleRow("调试输出", value => _gpuSharpenDumpStages = value, false));

        scroll.Add(CreateAdjustRow("对比度", -0.5f, 0.5f, 0f, "AdjustContrast", (cs, v) => cs.SetFloat("_Contrast", v), v => $"对比度 {v:0.00}"));
        scroll.Add(CreateAdjustRow("亮度", -0.5f, 0.5f, 0f, "AdjustBrightness", (cs, v) => cs.SetFloat("_Brightness", v), v => $"亮度 {v:0.00}"));
        scroll.Add(CreateAdjustRow("自然饱和度", -1f, 1f, 0f, "AdjustVibrance", (cs, v) => cs.SetFloat("_Vibrance", v), v => $"自然饱和度 {v:0.00}"));
        scroll.Add(CreateAdjustRow("去阴影", 0f, 0.5f, 0f, "AdjustShadows", (cs, v) => cs.SetFloat("_Shadows", v), v => $"去阴影 {v:0.00}"));
        scroll.Add(CreateAdjustRow("去高光", 0f, 0.5f, 0f, "AdjustHighlights", (cs, v) => cs.SetFloat("_Highlights", v), v => $"去高光 {v:0.00}"));
        scroll.Add(CreateAdjustRow("暖色滤镜", 0f, 1f, 0f, "WarmFilter", (cs, v) => cs.SetFloat("_Warm", v), v => $"暖色 {v:0.00}"));
        scroll.Add(CreateAdjustRow("冷色滤镜", 0f, 1f, 0f, "CoolFilter", (cs, v) => cs.SetFloat("_Cool", v), v => $"冷色 {v:0.00}"));
        scroll.Add(CreateAdjustRow("锐化", 0f, 4f, 0f, "Sharpen", (cs, v) => cs.SetFloat("_Sharpen", v), v => $"锐化 {v:0.00}"));
        scroll.Add(CreateAdjustRow("模糊", 0f, 4f, 0f, "Blur", (cs, v) => cs.SetFloat("_Blur", v), v => $"模糊 {v:0.00}"));
        return panel;
    }

    private VisualElement BuildReferenceButtons()
    {
        var shell = new VisualElement();
        shell.style.marginBottom = 10;

        var title = new Label("参考图");
        title.style.color = new Color(0.84f, 0.88f, 0.94f, 1f);
        title.style.marginBottom = 6;
        shell.Add(title);

        var row = new VisualElement();
        row.style.flexDirection = FlexDirection.Row;
        row.style.justifyContent = Justify.SpaceBetween;
        shell.Add(row);

        row.Add(BuildReferencePickerButton("男脸", OnPickMaleFace, out _maleFaceButton));
        row.Add(BuildReferencePickerButton("女脸", OnPickFemaleFace, out _femaleFaceButton));
        row.Add(BuildReferencePickerButton("背景", OnPickBackground, out _backgroundButton));
        return shell;
    }

    private VisualElement BuildReferencePickerButton(string label, Action onClick, out Button button)
    {
        var shell = new VisualElement();
        shell.style.flexDirection = FlexDirection.Column;
        shell.style.alignItems = Align.Center;

        button = new Button(onClick);
        button.style.width = 92;
        button.style.height = 92;
        button.style.backgroundColor = new StyleColor(new Color(0.18f, 0.19f, 0.24f, 1f));
        button.style.borderTopLeftRadius = 18;
        button.style.borderTopRightRadius = 18;
        button.style.borderBottomLeftRadius = 18;
        button.style.borderBottomRightRadius = 18;
        button.style.color = Color.white;
        button.text = label;
        shell.Add(button);

        var text = new Label(label);
        text.style.marginTop = 6;
        text.style.color = Color.white;
        shell.Add(text);
        return shell;
    }

    private VisualElement BuildProviderEditor()
    {
        var card = new VisualElement();
        card.style.backgroundColor = new StyleColor(new Color(0.17f, 0.18f, 0.22f, 0.96f));
        card.style.borderTopLeftRadius = 16;
        card.style.borderTopRightRadius = 16;
        card.style.borderBottomLeftRadius = 16;
        card.style.borderBottomRightRadius = 16;
        card.style.paddingLeft = 12;
        card.style.paddingRight = 12;
        card.style.paddingTop = 10;
        card.style.paddingBottom = 12;
        card.style.marginBottom = 10;

        var providerLabel = new Label("模型提供方");
        providerLabel.style.color = Color.white;
        providerLabel.style.marginBottom = 6;
        card.Add(providerLabel);

        _providerDropdown = new DropdownField();
        _providerDropdown.choices = Enum.GetNames(typeof(Image2ImageAI.Provider)).ToList();
        _providerDropdown.RegisterValueChangedCallback(evt =>
        {
            if (Host?.Image2ImageAI == null)
                return;
            if (Enum.TryParse<Image2ImageAI.Provider>(evt.newValue, out var provider))
            {
                Host.Image2ImageAI.CurrentProvider = provider;
                PlayerPrefs.SetString(PrefKeyAIProvider, evt.newValue);
                PlayerPrefs.Save();
                UpdateProviderUi();
            }
        });
        card.Add(_providerDropdown);

        var apiLabel = new Label("API Key");
        apiLabel.style.color = Color.white;
        apiLabel.style.marginTop = 8;
        apiLabel.style.marginBottom = 6;
        card.Add(apiLabel);

        _apiKeyField = new TextField();
        _apiKeyField.isPasswordField = true;
        _apiKeyField.RegisterValueChangedCallback(evt =>
        {
            if (Host?.Image2ImageAI == null)
                return;
            var provider = Host.Image2ImageAI.CurrentProvider;
            Host.Image2ImageAI.SetApiKeyForProvider(provider, evt.newValue);
            SaveApiKey(provider, evt.newValue);
        });
        card.Add(_apiKeyField);
        return card;
    }

    private VisualElement BuildToggleRow(string text, Action<bool> onChanged, bool defaultValue)
    {
        var row = new VisualElement();
        row.style.flexDirection = FlexDirection.Row;
        row.style.alignItems = Align.Center;
        row.style.justifyContent = Justify.SpaceBetween;
        row.style.backgroundColor = new StyleColor(new Color(0.17f, 0.18f, 0.22f, 0.96f));
        row.style.borderTopLeftRadius = 14;
        row.style.borderTopRightRadius = 14;
        row.style.borderBottomLeftRadius = 14;
        row.style.borderBottomRightRadius = 14;
        row.style.paddingLeft = 12;
        row.style.paddingRight = 12;
        row.style.paddingTop = 10;
        row.style.paddingBottom = 10;
        row.style.marginBottom = 8;

        var label = new Label(text);
        label.style.color = Color.white;
        row.Add(label);

        var toggle = new Toggle();
        toggle.value = defaultValue;
        toggle.RegisterValueChangedCallback(evt => onChanged?.Invoke(evt.newValue));
        row.Add(toggle);
        return row;
    }

    private void SetAdjustPanelCollapsed(bool collapsed, bool showToast)
    {
        _adjustPanelCollapsed = collapsed;
        if (_adjustBody != null)
            _adjustBody.style.display = collapsed ? DisplayStyle.None : DisplayStyle.Flex;
        if (_adjustPanel != null)
        {
            _adjustPanel.style.height = collapsed ? 58 : new StyleLength(StyleKeyword.Auto);
            if (!IsPortraitLayout)
                _adjustPanel.style.bottom = collapsed ? new StyleLength(StyleKeyword.Auto) : 18;
        }
        if (_panelToggleButton != null)
            _panelToggleButton.text = collapsed ? "展开调节" : "收起调节";
        if (showToast)
            ShowToast(collapsed ? "调节面板已折叠" : "调节面板已展开", 1400);
    }

    private async UniTaskVoid ApplyPresetAsync(string presetId, string title)
    {
        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null)
            return;

        ShowBusy("应用预设 " + title);
        try
        {
            await UniTask.Yield();
            var tex = GeneratePresetTexture(src, presetId, title);
            if (tex != null)
                AddHistory(tex, "预设: " + title);
        }
        finally
        {
            HideBusy();
        }
    }

    private Texture2D GeneratePresetTexture(Texture2D src, string presetId, string title)
    {
        Color32[] pixels;
        try
        {
            pixels = src.GetPixels32();
        }
        catch
        {
            return null;
        }

        void Adjust(Func<Color32, Color32> map)
        {
            for (var i = 0; i < pixels.Length; i++)
                pixels[i] = map(pixels[i]);
        }

        switch (presetId)
        {
            case "bw":
                Adjust(p =>
                {
                    var gray = (byte)Mathf.Clamp((p.r * 0.3f) + (p.g * 0.59f) + (p.b * 0.11f), 0f, 255f);
                    return new Color32(gray, gray, gray, p.a);
                });
                break;
            case "warm":
                Adjust(p => new Color32((byte)Mathf.Clamp(p.r + 18, 0, 255), (byte)Mathf.Clamp(p.g + 8, 0, 255), (byte)Mathf.Clamp(p.b - 10, 0, 255), p.a));
                break;
            case "clean":
                Adjust(p => new Color32((byte)Mathf.Clamp(p.r + 6, 0, 255), (byte)Mathf.Clamp(p.g + 10, 0, 255), (byte)Mathf.Clamp(p.b + 16, 0, 255), p.a));
                break;
            case "film":
                Adjust(p => new Color32((byte)Mathf.Clamp(p.r * 1.03f + 10f, 0, 255), (byte)Mathf.Clamp(p.g * 0.97f, 0, 255), (byte)Mathf.Clamp(p.b * 0.92f + 12f, 0, 255), p.a));
                break;
            case "portrait":
                Adjust(p => new Color32((byte)Mathf.Clamp(p.r + 12, 0, 255), (byte)Mathf.Clamp(p.g + 4, 0, 255), (byte)Mathf.Clamp(p.b + 2, 0, 255), p.a));
                break;
            case "night":
                Adjust(p => new Color32((byte)Mathf.Clamp(p.r * 0.92f, 0, 255), (byte)Mathf.Clamp(p.g * 0.96f, 0, 255), (byte)Mathf.Clamp(p.b * 1.12f + 6f, 0, 255), p.a));
                break;
            case "fresh":
                Adjust(p => new Color32((byte)Mathf.Clamp(p.r * 0.96f, 0, 255), (byte)Mathf.Clamp(p.g * 1.05f + 8f, 0, 255), (byte)Mathf.Clamp(p.b * 1.03f + 4f, 0, 255), p.a));
                break;
            case "cn":
            default:
                Adjust(p => new Color32((byte)Mathf.Clamp(p.r * 1.08f + 6f, 0, 255), (byte)Mathf.Clamp(p.g * 1.02f, 0, 255), (byte)Mathf.Clamp(p.b * 0.95f, 0, 255), p.a));
                break;
        }

        var result = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false);
        result.SetPixels32(pixels);
        result.Apply(false, false);
        result.wrapMode = TextureWrapMode.Clamp;
        result.filterMode = FilterMode.Bilinear;
        result.name = title;
        return result;
    }

    private void OnFaceSwap() => ApplyOperation(ImageOp.FaceSwap);
    private void OnSharpen() => ApplyOperation(ImageOp.Sharpen);
    private void OnWhiten() => ApplyOperation(ImageOp.Whiten);
    private void OnSharpenWhiten() => ApplyOperation(ImageOp.SharpenWhiten);
    private void OnChangeBackground() => ApplyOperation(ImageOp.ChangeBackground);
    private void OnDehazeColorGrade() => ApplyOperation(ImageOp.DehazeColorGrade);
    private void OnColorGrade() => ApplyOperation(ImageOp.ColorGrade);
    private void OnDehaze() => ApplyOperation(ImageOp.Dehaze);

    private void ApplyOperation(ImageOp op)
    {
        RunAIForOperation(op).Forget();
    }

    private async UniTaskVoid RunAIForOperation(ImageOp op)
    {
        if (_aiRunning || Host?.Image2ImageAI == null || _lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
            return;

        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        var original = GetOriginalHistoryTexture();
        if (src == null || original == null)
            return;

        _aiRunning = true;
        ShowBusy(OpLabel(op) + "处理中");
        try
        {
            var useChinesePrompt = Host.Image2ImageAI.CurrentProvider == Image2ImageAI.Provider.Doubao ||
                                   Host.Image2ImageAI.CurrentProvider == Image2ImageAI.Provider.AliTongyiWanxiang;
            var prompt = BuildPromptForOp(op, useChinesePrompt);

            if ((op == ImageOp.Sharpen || op == ImageOp.SharpenWhiten) && Host.Image2ImageAI.CurrentProvider != Image2ImageAI.Provider.Doubao)
            {
                prompt += useChinesePrompt
                    ? " 调整构图，聚焦前景人物，适当放大主体。"
                    : " Adjust the framing slightly to emphasize the main subject.";
            }

            var refs = new List<Texture2D> { src };
            if (op == ImageOp.FaceSwap)
            {
                if (_maleFaceTexture != null)
                    refs.Add(_maleFaceTexture);
                if (_femaleFaceTexture != null && !ReferenceEquals(_femaleFaceTexture, _maleFaceTexture))
                    refs.Add(_femaleFaceTexture);
            }
            else if (op == ImageOp.ChangeBackground && _backgroundTexture != null)
            {
                refs.Add(_backgroundTexture);
            }

            if (ShouldAppendPromptToggles(op))
                prompt = AppendPromptToggles(prompt, useChinesePrompt);

            var result = await Host.Image2ImageAI.ImageToImageAsync(refs, prompt, original.width, original.height, _lifetimeCts.Token);
            if (result != null)
                AddHistory(result, OpLabel(op));
        }
        finally
        {
            _aiRunning = false;
            HideBusy();
        }
    }

    private bool ShouldAppendPromptToggles(ImageOp op)
    {
        return op == ImageOp.Sharpen ||
               op == ImageOp.Whiten ||
               op == ImageOp.SharpenWhiten ||
               op == ImageOp.DehazeColorGrade ||
               op == ImageOp.ColorGrade ||
               op == ImageOp.Dehaze;
    }

    private string AppendPromptToggles(string prompt, bool useChinesePrompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
            prompt = string.Empty;

        if (_appendDeGlarePrompt)
            prompt += useChinesePrompt
                ? " 同时减少反光与镜面高光，保留真实纹理。"
                : " Reduce glare and specular highlights while preserving fine texture.";
        if (_appendRemoveBgPeoplePrompt)
            prompt += useChinesePrompt
                ? " 移除背景人物并自然补全背景纹理。"
                : " Remove background people and inpaint the scene naturally.";
        return prompt;
    }

    private async UniTask<int> OnSelectAIResultIndex(IReadOnlyList<Texture2D> options)
    {
        HideBusy();
        return await ShowChoiceAsync(options);
    }

    private void OnAiRequestError(string message)
    {
        if (!string.IsNullOrWhiteSpace(message))
            ShowToast(message, 4200);
    }

    private void OnClipClassify()
    {
        ApplyClipClassificationAsync().Forget();
    }

    private async UniTaskVoid ApplyClipClassificationAsync()
    {
        if (_adjustRunning || _aiRunning || _lifetimeCts == null || Host?.ClipRunner == null)
            return;

        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null)
            return;

        _adjustRunning = true;
        ShowProgress("CLIP 分类");
        try
        {
            void OnProgress(float p, string t) => SetProgress(p, t);
            Host.ClipRunner.ProgressChanged -= OnProgress;
            Host.ClipRunner.ProgressChanged += OnProgress;
            var result = await Host.ClipRunner.ProcessAsync(src, _lifetimeCts.Token);
            Host.ClipRunner.ProgressChanged -= OnProgress;
            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3200);
                return;
            }
            ShowToast("CLIP: " + result.bestLabel + "  " + FormatClipTopScores(result.scores, 3), 3600);
        }
        finally
        {
            _adjustRunning = false;
            HideProgress();
        }
    }

    private void OnGpuSharpen() => ApplyGpuSharpenAsync().Forget();
    private void OnRealEsrganRepro() => ApplyRealEsrganReproAsync().Forget();
    private void OnYoloAndInpaintingRepro() => ApplyYoloAndInpaintingReproAsync().Forget();
    private void OnMattingRepro() => ApplyMattingReproAsync().Forget();
    private void OnGfpganRepro() => ApplyGfpganReproAsync().Forget();
    private void OnCodeFormerRepro() => ApplyCodeFormerReproAsync().Forget();

    private async UniTaskVoid ApplyGpuSharpenAsync()
    {
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || Host?.GpuSharpenRunner == null)
            return;

        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null)
            return;

        _adjustRunning = true;
        ShowBusy("GPU 清晰化");
        try
        {
            Texture2D faceMask = Host.FaceMaskGenerator != null ? Host.FaceMaskGenerator.currentImageFaceMask : null;
            if ((faceMask == null || faceMask.width != src.width || faceMask.height != src.height) && Host.FaceMaskGenerator != null)
            {
                CancelAndDisposeCts(ref _faceMaskCts);
                _faceMaskCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
                var generated = await Host.FaceMaskGenerator.GenerateForCurrentAsync(src, false, _faceMaskCts.Token);
                faceMask = string.IsNullOrWhiteSpace(generated.error) ? generated.mask : null;
            }

            var result = await Host.GpuSharpenRunner.ProcessAsync(src, faceMask, _gpuSharpenDumpStages, _lifetimeCts.Token);
            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3200);
                return;
            }
            if (result.texture != null)
                AddHistory(result.texture, "GPU 清晰化");
        }
        finally
        {
            _adjustRunning = false;
            HideBusy();
        }
    }

    private async UniTaskVoid ApplyRealEsrganReproAsync()
    {
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || Host?.RealEsrganReproRunner == null)
            return;
        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null)
            return;

        _adjustRunning = true;
        ShowProgress("ESRGAN");
        try
        {
            void OnProgress(float p, string t) => SetProgress(p, t);
            Host.RealEsrganReproRunner.ProgressChanged -= OnProgress;
            Host.RealEsrganReproRunner.ProgressChanged += OnProgress;
            var result = await Host.RealEsrganReproRunner.ProcessAsync(src, _lifetimeCts.Token);
            Host.RealEsrganReproRunner.ProgressChanged -= OnProgress;
            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3400);
                return;
            }
            if (result.texture != null)
                AddHistory(result.texture, "ESRGAN");
        }
        finally
        {
            _adjustRunning = false;
            HideProgress();
        }
    }

    private async UniTaskVoid ApplyYoloAndInpaintingReproAsync()
    {
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || Host?.YoloSegRunner == null || Host?.SDInpaintingRunner == null)
            return;
        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null)
            return;

        _adjustRunning = true;
        ShowProgress("YOLO + Inpainting");
        try
        {
            NcnnGpuResourceTracker.Enabled = true;
            NcnnGpuResourceTracker.Reset("MainView2.YoloInpaint");
            YoloInpaintResourceSnapshotLines.Clear();
            LogYoloInpaintResourceSnapshot("begin");
            var oldTargetPersonOnly = Host.YoloSegRunner.targetPersonOnly;
            Host.YoloSegRunner.targetPersonOnly = true;
            void OnYoloProgress(float p, string t) => SetProgress(p * 0.35f, string.IsNullOrWhiteSpace(t) ? "YOLO Seg" : t);
            void OnInpaintProgress(float p, string t) => SetProgress(0.35f + p * 0.65f, string.IsNullOrWhiteSpace(t) ? "SD Inpainting" : t);
            Host.YoloSegRunner.ProgressChanged -= OnYoloProgress;
            Host.YoloSegRunner.ProgressChanged += OnYoloProgress;
            YoloSegResult result;
            try
            {
                result = await Host.YoloSegRunner.ProcessAsync(src, _lifetimeCts.Token);
                LogYoloInpaintResourceSnapshot("after_yolo_process");
            }
            finally
            {
                Host.YoloSegRunner.ProgressChanged -= OnYoloProgress;
                Host.YoloSegRunner.targetPersonOnly = oldTargetPersonOnly;
            }
            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3400);
                return;
            }
            if (result.personCount <= 0 || result.mask == null)
            {
                ShowToast("未检测到可修复的人物区域", 2600);
                if (result.overlay != null)
                    AddHistory(result.overlay, "YOLO 未检测到人物");
                return;
            }

            if (result.texture != null)
            {
                Destroy(result.texture);
                result.texture = null;
            }

            if (result.overlay != null)
            {
                Destroy(result.overlay);
                result.overlay = null;
            }

            Host.YoloSegRunner.ReleaseRuntimeResources();
            await ReleaseGpuPressureBeforeInpaintAsync(_lifetimeCts.Token);
            LogYoloInpaintResourceSnapshot("after_yolo_release");

            Host.SDInpaintingRunner.useOfficialUnetCache = false;
            Host.SDInpaintingRunner.enableTempPool = false;
            Host.SDInpaintingRunner.maxPooledPerShape = 0;
            Host.SDInpaintingRunner.keepRawConvWeightsForTexturePath = false;
            Host.SDInpaintingRunner.tensorTextureFormat = RenderTextureFormat.ARGBHalf;
            Host.SDInpaintingRunner.encoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
            Host.SDInpaintingRunner.decoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
            Host.SDInpaintingRunner.ApplyPeopleRemovalPreset();
            Host.SDInpaintingRunner.ReleaseRuntimeResources();
            await ReleaseGpuPressureBeforeInpaintAsync(_lifetimeCts.Token);
            LogYoloInpaintResourceSnapshot("before_inpaint_process");

            Host.SDInpaintingRunner.ProgressChanged -= OnInpaintProgress;
            Host.SDInpaintingRunner.ProgressChanged += OnInpaintProgress;
            SDInpaintingNcnnReproResult inpaintResult;
            try
            {
                inpaintResult = await Host.SDInpaintingRunner.ProcessAsync(src, result.mask, _lifetimeCts.Token);
                LogYoloInpaintResourceSnapshot("after_inpaint_process");
            }
            finally
            {
                Host.SDInpaintingRunner.ProgressChanged -= OnInpaintProgress;
            }
            if (!string.IsNullOrWhiteSpace(inpaintResult.error))
            {
                ShowToast(inpaintResult.error, 3600);
                if (result.overlay != null)
                    AddHistory(result.overlay, $"YOLO 识别 {result.personCount}");
                return;
            }

            if (inpaintResult.texture != null)
            {
                AddHistory(inpaintResult.texture, $"YOLO修复 {result.personCount}");
                LogYoloInpaintResourceSnapshot("after_add_history");
            }
        }
        finally
        {
            LogYoloInpaintResourceSnapshot("finally");
            TryWriteYoloInpaintResourceReport();
            NcnnGpuResourceTracker.Enabled = false;
            _adjustRunning = false;
            HideProgress();
        }
    }

    private static async UniTask ReleaseGpuPressureBeforeInpaintAsync(CancellationToken ct)
    {
        await UniTask.Yield(PlayerLoopTiming.Update, ct);
        var unloadOp = Resources.UnloadUnusedAssets();
        if (unloadOp != null)
            await unloadOp.ToUniTask(cancellationToken: ct);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        await UniTask.Yield(PlayerLoopTiming.Update, ct);
    }

    private static void LogYoloInpaintResourceSnapshot(string stage)
    {
        try
        {
            var process = Process.GetCurrentProcess();
            var privateMb = process.PrivateMemorySize64 / (1024.0 * 1024.0);
            var workingSetMb = process.WorkingSet64 / (1024.0 * 1024.0);
            var managedMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
            var gfxMb = GetGraphicsDriverMemoryMb();
            var rtCount = Resources.FindObjectsOfTypeAll<RenderTexture>().Length;
            var line =
                "[MainView2][YoloInpaint][Resources] stage=" + (stage ?? "")
                + " | private_mb=" + privateMb.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)
                + " | working_set_mb=" + workingSetMb.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)
                + " | managed_mb=" + managedMb.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)
                + " | gfx_mb=" + gfxMb.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)
                + " | rt_objects=" + rtCount.ToString(System.Globalization.CultureInfo.InvariantCulture)
                + " | " + NcnnGpuResourceTracker.BuildSummary();
            UnityEngine.Debug.Log(line);
            YoloInpaintResourceSnapshotLines.Add(line);
        }
        catch (Exception e)
        {
            try
            {
                var line = "[MainView2][YoloInpaint][Resources] stage=" + (stage ?? "") + " | snapshot_failed=" + e.Message;
                UnityEngine.Debug.Log(line);
                YoloInpaintResourceSnapshotLines.Add(line);
            }
            catch { }
        }
    }

    private static void TryWriteYoloInpaintResourceReport()
    {
        try
        {
            var root = Path.Combine(Path.GetTempPath(), "YanQi", "AIImage");
            Directory.CreateDirectory(root);
            var dir = Path.Combine(root, "AIImage_MainView2_YoloInpaint_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            Directory.CreateDirectory(dir);
            if (YoloInpaintResourceSnapshotLines.Count > 0)
                File.WriteAllText(Path.Combine(dir, "resource_snapshots.txt"), string.Join(Environment.NewLine, YoloInpaintResourceSnapshotLines));
            NcnnGpuResourceTracker.WriteReport(dir, "gpu_resource_stats.txt");
        }
        catch
        {
        }
    }

    private static float GetGraphicsDriverMemoryMb()
    {
        try
        {
            return UnityEngine.Profiling.Profiler.GetAllocatedMemoryForGraphicsDriver() / (1024f * 1024f);
        }
        catch
        {
            return 0f;
        }
    }

    private async UniTaskVoid ApplyMattingReproAsync()
    {
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || Host?.MattingReproRunner == null)
            return;
        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null)
            return;

        _adjustRunning = true;
        ShowProgress("Matting");
        try
        {
            void OnProgress(float p, string t) => SetProgress(p, t);
            Host.MattingReproRunner.ProgressChanged -= OnProgress;
            Host.MattingReproRunner.ProgressChanged += OnProgress;
            var result = await Host.MattingReproRunner.ProcessAsync(src, _lifetimeCts.Token);
            Host.MattingReproRunner.ProgressChanged -= OnProgress;
            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3400);
                return;
            }
            if (result.texture != null)
                AddHistory(result.texture, "Matting");
        }
        finally
        {
            _adjustRunning = false;
            HideProgress();
        }
    }

    private async UniTaskVoid ApplyGfpganReproAsync()
    {
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || Host?.GfpganReproRunner == null)
            return;
        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null)
            return;

        _adjustRunning = true;
        ShowProgress("GFPGAN");
        try
        {
            void OnProgress(float p, string t) => SetProgress(p, t);
            Host.GfpganReproRunner.ProgressChanged -= OnProgress;
            Host.GfpganReproRunner.ProgressChanged += OnProgress;
            var result = await Host.GfpganReproRunner.ProcessAsync(src, _lifetimeCts.Token);
            Host.GfpganReproRunner.ProgressChanged -= OnProgress;
            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3400);
                return;
            }
            if (result.texture != null)
                AddHistory(result.texture, "GFPGAN");
        }
        finally
        {
            _adjustRunning = false;
            HideProgress();
        }
    }

    private async UniTaskVoid ApplyCodeFormerReproAsync()
    {
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || Host?.CodeFormerReproRunner == null)
            return;
        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null)
            return;

        _adjustRunning = true;
        ShowProgress("CodeFormer");
        try
        {
            void OnProgress(float p, string t) => SetProgress(p, t);
            Host.CodeFormerReproRunner.ProgressChanged -= OnProgress;
            Host.CodeFormerReproRunner.ProgressChanged += OnProgress;
            var result = await Host.CodeFormerReproRunner.ProcessAsync(src, _lifetimeCts.Token);
            Host.CodeFormerReproRunner.ProgressChanged -= OnProgress;
            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3400);
                return;
            }
            if (result.texture != null)
                AddHistory(result.texture, "CodeFormer");
        }
        finally
        {
            _adjustRunning = false;
            HideProgress();
        }
    }

    private void OnSaveCurrentImage()
    {
        SaveCurrentImageAsync().Forget();
    }

    private async UniTaskVoid SaveCurrentImageAsync()
    {
        var path = CurrentImagePath;
        if (string.IsNullOrWhiteSpace(path))
            return;
        var tex = GetCurrentHistoryTexture();
        if (tex == null)
            return;

        byte[] bytes = null;
        var ext = (Path.GetExtension(path) ?? string.Empty).ToLowerInvariant();
        try
        {
            if (ext == ".png") bytes = tex.EncodeToPNG();
            else if (ext == ".jpg" || ext == ".jpeg") bytes = tex.EncodeToJPG(95);
            else if (ext == ".tga") bytes = tex.EncodeToTGA();
            else if (ext == ".exr") bytes = tex.EncodeToEXR(Texture2D.EXRFlags.CompressZIP);
        }
        catch
        {
            bytes = null;
        }

        if (bytes == null || bytes.Length == 0)
        {
            ShowToast("当前格式暂不支持覆盖保存", 2400);
            return;
        }

        try
        {
            await File.WriteAllBytesAsync(path, bytes);
            Host?.InvalidateTextureCacheForPath(path);
            if (Host != null && Host.ReloadMainImageFromDisk(path))
                ShowToast("已保存，并按原路径重新载入", 1800);
            else
                ShowToast("已保存到原路径", 1800);
        }
        catch
        {
            ShowToast("保存失败", 2200);
        }
    }

    private void OnBrowseOriginalImage()
    {
        var path = CurrentImagePath;
        if (string.IsNullOrWhiteSpace(path))
            return;
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
            OpenFolderInShell(directory);
    }

    private void OnPickMaleFace() => PickReferenceImageAsync(_maleFaceButton, tex => _maleFaceTexture = tex, () => _maleFaceTexture, path => _maleFacePath = path, PrefKeyMaleFacePath, "男脸").Forget();
    private void OnPickFemaleFace() => PickReferenceImageAsync(_femaleFaceButton, tex => _femaleFaceTexture = tex, () => _femaleFaceTexture, path => _femaleFacePath = path, PrefKeyFemaleFacePath, "女脸").Forget();
    private void OnPickBackground() => PickReferenceImageAsync(_backgroundButton, tex => _backgroundTexture = tex, () => _backgroundTexture, path => _backgroundPath = path, PrefKeyBackgroundPath, "背景").Forget();

    private async UniTaskVoid PickReferenceImageAsync(Button button, Action<Texture2D> setTexture, Func<Texture2D> getTexture, Action<string> setPath, string prefKey, string fallbackLabel)
    {
        if (_lifetimeCts == null || Host?.FileDialog == null)
            return;

        var path = await Host.FileDialog.ShowOpenImageAsync();
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
            return;

        if (string.IsNullOrWhiteSpace(path))
        {
            var old = getTexture();
            if (old != null)
                Destroy(old);
            setTexture(null);
            setPath(null);
            PlayerPrefs.DeleteKey(prefKey);
            PlayerPrefs.Save();
            if (button != null)
            {
                button.style.backgroundImage = StyleKeyword.None;
                button.text = fallbackLabel;
            }
            return;
        }

        var texture = LoadTextureFromFile(path);
        if (texture == null)
            return;

        var oldTexture = getTexture();
        if (oldTexture != null)
            Destroy(oldTexture);

        setTexture(texture);
        setPath(path);
        PlayerPrefs.SetString(prefKey, path);
        PlayerPrefs.Save();

        if (button != null)
        {
            button.style.backgroundImage = new StyleBackground(texture);
            button.style.unityBackgroundScaleMode = ScaleMode.ScaleToFit;
            button.text = string.Empty;
        }

        if (ReferenceEquals(button, _maleFaceButton))
            PrepareFaceMaskForReferenceAsync(texture, true).Forget();
        else if (ReferenceEquals(button, _femaleFaceButton))
            PrepareFaceMaskForReferenceAsync(texture, false).Forget();
    }

    private async UniTaskVoid PrepareFaceMaskForSelectedImageAsync(string _, Texture2D src, bool dumpDebug)
    {
        if (_lifetimeCts == null || Host?.FaceMaskGenerator == null || src == null)
            return;

        CancelAndDisposeCts(ref _faceMaskCts);
        _faceMaskCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
        var result = await Host.FaceMaskGenerator.GenerateForCurrentAsync(src, dumpDebug, _faceMaskCts.Token);
        if (!string.IsNullOrWhiteSpace(result.error))
            UnityEngine.Debug.Log(result.error);
    }

    private async UniTaskVoid PrepareFaceMaskForReferenceAsync(Texture2D src, bool isMale)
    {
        if (_lifetimeCts == null || Host?.FaceMaskGenerator == null || src == null)
            return;

        if (isMale)
        {
            CancelAndDisposeCts(ref _maleFaceMaskCts);
            _maleFaceMaskCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
            await Host.FaceMaskGenerator.GenerateForMaleAsync(src, false, _maleFaceMaskCts.Token);
        }
        else
        {
            CancelAndDisposeCts(ref _femaleFaceMaskCts);
            _femaleFaceMaskCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
            await Host.FaceMaskGenerator.GenerateForFemaleAsync(src, false, _femaleFaceMaskCts.Token);
        }
    }

    private void RestoreReferencePickersFromPrefs()
    {
        _maleFacePath = PlayerPrefs.GetString(PrefKeyMaleFacePath, string.Empty);
        _femaleFacePath = PlayerPrefs.GetString(PrefKeyFemaleFacePath, string.Empty);
        _backgroundPath = PlayerPrefs.GetString(PrefKeyBackgroundPath, string.Empty);

        if (File.Exists(_maleFacePath))
            _maleFaceTexture = LoadTextureFromFile(_maleFacePath);
        if (File.Exists(_femaleFacePath))
            _femaleFaceTexture = LoadTextureFromFile(_femaleFacePath);
        if (File.Exists(_backgroundPath))
            _backgroundTexture = LoadTextureFromFile(_backgroundPath);
    }

    private void SyncReferenceButtonState()
    {
        ApplyReferenceButtonTexture(_maleFaceButton, _maleFaceTexture, "男脸");
        ApplyReferenceButtonTexture(_femaleFaceButton, _femaleFaceTexture, "女脸");
        ApplyReferenceButtonTexture(_backgroundButton, _backgroundTexture, "背景");
    }

    private static void ApplyReferenceButtonTexture(Button button, Texture2D texture, string fallbackLabel)
    {
        if (button == null)
            return;
        if (texture == null)
        {
            button.style.backgroundImage = StyleKeyword.None;
            button.text = fallbackLabel;
            return;
        }
        button.style.backgroundImage = new StyleBackground(texture);
        button.style.unityBackgroundScaleMode = ScaleMode.ScaleToFit;
        button.text = string.Empty;
    }

    private void RestoreAISettingsFromPrefs()
    {
        if (Host?.Image2ImageAI == null)
            return;

        var providerName = PlayerPrefs.GetString(PrefKeyAIProvider, Host.Image2ImageAI.CurrentProvider.ToString());
        if (Enum.TryParse<Image2ImageAI.Provider>(providerName, out var provider))
            Host.Image2ImageAI.CurrentProvider = provider;

        TryRestoreApiKey(Image2ImageAI.Provider.GoogleAIStudio, PrefKeyGoogleApiKey);
        TryRestoreApiKey(Image2ImageAI.Provider.Replicate, PrefKeyReplicateApiKey);
        TryRestoreApiKey(Image2ImageAI.Provider.AliTongyiWanxiang, PrefKeyDashScopeApiKey);
        TryRestoreApiKey(Image2ImageAI.Provider.Doubao, PrefKeyDoubaoApiKey);
        TryRestoreApiKey(Image2ImageAI.Provider.HuggingFaceInferenceProviders, PrefKeyHuggingFaceToken);
        TryRestoreApiKey(Image2ImageAI.Provider.RunwareAI, PrefKeyRunwareApiKey);
        TryRestoreApiKey(Image2ImageAI.Provider.Lumenfall, PrefKeyLumenfallApiKey);
    }

    private void UpdateProviderUi()
    {
        if (_providerDropdown == null || _apiKeyField == null || Host?.Image2ImageAI == null)
            return;

        var provider = Host.Image2ImageAI.CurrentProvider;
        _providerDropdown.SetValueWithoutNotify(provider.ToString());
        _apiKeyField.SetValueWithoutNotify(Host.Image2ImageAI.GetApiKeyForProvider(provider));
    }

    private void SaveApiKey(Image2ImageAI.Provider provider, string value)
    {
        var key = provider switch
        {
            Image2ImageAI.Provider.GoogleAIStudio => PrefKeyGoogleApiKey,
            Image2ImageAI.Provider.Replicate => PrefKeyReplicateApiKey,
            Image2ImageAI.Provider.AliTongyiWanxiang => PrefKeyDashScopeApiKey,
            Image2ImageAI.Provider.Doubao => PrefKeyDoubaoApiKey,
            Image2ImageAI.Provider.HuggingFaceInferenceProviders => PrefKeyHuggingFaceToken,
            Image2ImageAI.Provider.RunwareAI => PrefKeyRunwareApiKey,
            Image2ImageAI.Provider.Lumenfall => PrefKeyLumenfallApiKey,
            _ => null
        };

        if (!string.IsNullOrWhiteSpace(key))
        {
            PlayerPrefs.SetString(key, value ?? string.Empty);
            PlayerPrefs.Save();
        }
    }

    private void TryRestoreApiKey(Image2ImageAI.Provider provider, string prefKey)
    {
        if (Host?.Image2ImageAI == null)
            return;
        var value = PlayerPrefs.GetString(prefKey, null);
        if (!string.IsNullOrWhiteSpace(value))
            Host.Image2ImageAI.SetApiKeyForProvider(provider, value);
    }

    private void BindAiEvents()
    {
        if (Host?.Image2ImageAI == null)
            return;
        Host.Image2ImageAI.SelectResultIndex -= OnSelectAIResultIndex;
        Host.Image2ImageAI.SelectResultIndex += OnSelectAIResultIndex;
        Host.Image2ImageAI.RequestError -= OnAiRequestError;
        Host.Image2ImageAI.RequestError += OnAiRequestError;
    }

    private void UnbindAiEvents()
    {
        if (Host?.Image2ImageAI == null)
            return;
        Host.Image2ImageAI.SelectResultIndex -= OnSelectAIResultIndex;
        Host.Image2ImageAI.RequestError -= OnAiRequestError;
    }

    private static void CancelAndDisposeCts(ref System.Threading.CancellationTokenSource cts)
    {
        if (cts == null)
            return;
        try { cts.Cancel(); } catch { }
        try { cts.Dispose(); } catch { }
        cts = null;
    }

    private static bool IsOriginalDefinedName(string fileName)
    {
        var name = (fileName ?? string.Empty).ToLowerInvariant();
        return OriginalNameMarkersZh.Any(marker => name.Contains(marker)) ||
               OriginalNameMarkersEn.Any(marker => name.Contains(marker));
    }

    private static bool IsOriginalDefinedPath(string path)
    {
        var text = (path ?? string.Empty).ToLowerInvariant();
        return OriginalNameMarkersZh.Any(marker => text.Contains(marker)) ||
               OriginalNameMarkersEn.Any(marker => text.Contains(marker));
    }

    private static string FormatClipTopScores(ClipLabelScore[] scores, int count)
    {
        if (scores == null || scores.Length == 0)
            return string.Empty;
        var take = Mathf.Clamp(count, 1, scores.Length);
        return string.Join("  ", scores.Take(take).Select(s => $"{s.label}:{(s.probability * 100f):0}%"));
    }

    private static string BuildPromptForOp(ImageOp op, bool useChinesePrompt)
    {
        if (useChinesePrompt)
        {
            return op switch
            {
                ImageOp.FaceSwap => "以当前图片为主体，进行自然换脸，保持光照、肤色和细节真实。",
                ImageOp.Sharpen => "严格保持人物五官与发型不变，提升主体清晰度。",
                ImageOp.Whiten => "严格保持人物五官与发型不变，进行轻微美白和肤色优化，避免过度磨皮。",
                ImageOp.SharpenWhiten => "严格保持人物五官与发型不变，提升主体清晰度并进行轻微美白和肤色优化。",
                ImageOp.ChangeBackground => "保持主体完整自然，替换背景并使边缘干净、融合自然。",
                ImageOp.DehazeColorGrade => "对图片去雾、提对比并进行自然调色，避免过饱和。",
                ImageOp.ColorGrade => "对图片进行自然调色，提升整体观感与层次。",
                ImageOp.Dehaze => "对图片去雾并增强通透感，保留细节。",
                _ => op.ToString()
            };
        }

        return op switch
        {
            ImageOp.FaceSwap => "Perform a natural face replacement while preserving realistic skin tone, lighting and details.",
            ImageOp.Sharpen => "Keep facial features unchanged and enhance the subject's clarity.",
            ImageOp.Whiten => "Keep facial features unchanged and apply subtle whitening with realistic skin texture.",
            ImageOp.SharpenWhiten => "Keep facial features unchanged, increase clarity and apply subtle whitening.",
            ImageOp.ChangeBackground => "Replace the background while keeping the subject intact with clean edges.",
            ImageOp.DehazeColorGrade => "Remove haze, boost clarity and apply natural color grading.",
            ImageOp.ColorGrade => "Apply natural color grading and improve depth without oversaturation.",
            ImageOp.Dehaze => "Remove haze and improve clarity while preserving detail.",
            _ => op.ToString()
        };
    }

    private static string OpLabel(ImageOp op)
    {
        return op switch
        {
            ImageOp.FaceSwap => "换脸",
            ImageOp.Sharpen => "清晰",
            ImageOp.Whiten => "美白",
            ImageOp.SharpenWhiten => "清晰+美白",
            ImageOp.ChangeBackground => "换背景",
            ImageOp.DehazeColorGrade => "去雾+调色",
            ImageOp.ColorGrade => "调色",
            ImageOp.Dehaze => "去雾",
            _ => op.ToString()
        };
    }
}
