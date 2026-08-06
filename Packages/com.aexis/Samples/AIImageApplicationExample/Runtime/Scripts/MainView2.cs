using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Aexis.Samples.Async;
using AIImage.Qwen35;
using Aexis.Ncnn;
using UnityEngine;
using UnityEngine.UIElements;
using Aexis.Execution;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

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
    private const string InpaintBackendEnvVar = "AIIMAGE_INPAINT_BACKEND";
    private const int Qwen35AnalysisMaxNewTokens = 256;
    private const string Qwen35DirectResponseInstruction =
        "请直接用中文回答，不要复述任务或展示推理过程。第一行必须写出图中人物总数，"
        + "然后按从左到右的顺序列出每个独立人物；单独可见的手、手臂、倒影或局部肢体不算额外人物，"
        + "不要在描述第一个人物后结束。\n";
    private const string Qwen35AnalysisPrompt =
        "请对当前图像进行详细、客观的中文分析，涵盖主体与人物、场景环境、构图、色彩与光线、可见文字、关键细节和可能用途。"
        + "对无法确认的内容明确说明，不要编造。";

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
    public bool HasCurrentImage => GetCurrentHistoryTexture() != null || GetOriginalHistoryTexture() != null;

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
    private Button _qwenAnalysisButton;
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private Button _developmentRunnerTestButton;
    private bool _developmentRunnerTestRunning;
#endif
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
    private bool _qwenAnalysisRunning;
    private bool _saveCurrentImageRunning;
#if UNITY_ANDROID
    private bool _saveStoragePermissionRequestInFlight;
#endif
    private VisualElement _qwenAnalysisOverlay;
    private Label _qwenAnalysisStatus;
    private Label _qwenAnalysisOutput;
    private ProgressBar _qwenAnalysisProgress;
    private Button _qwenAnalysisCancelButton;
    private Button _qwenAnalysisCloseButton;
    private Button _qwenAnalysisCopyButton;
    private Texture3D _identityLut;
    private Texture3D _selectedLut;
    private int _selectedLutSize;
    private Vector3 _selectedLutDomainMin = Vector3.zero;
    private Vector3 _selectedLutDomainMax = Vector3.one;
    private Button _lutPickerButton;
    private Texture2D _localAdjustmentMask;
    private Color32[] _localAdjustmentMaskPixels;
    private bool _localMaskPainting;
    private Vector2 _lastLocalMaskPoint;
    private bool _hasLastLocalMaskPoint;
    private Button _localMaskPaintButton;
    private static readonly List<string> YoloInpaintResourceSnapshotLines = new List<string>(256);

    private struct AutoToneAnalysis
    {
        public float exposure;
        public float shadows;
        public float highlights;
        public float contrast;
        public float vibrance;
        public float temperature;
        public float tint;
    }

    private sealed class SaveFileResult
    {
        public bool success;
        public bool usedFallback;
        public string path;
        public string error;
    }

    private sealed class OriginalBackupResult
    {
        public bool success;
        public bool moved;
        public string path;
        public string error;
    }

    private enum AutoToneProfile
    {
        General,
        Portrait,
        Food,
        Landscape,
        Night,
        Architecture,
        Document
    }

    private System.Threading.CancellationTokenSource _lifetimeCts;
    private System.Threading.CancellationTokenSource _faceMaskCts;
    private System.Threading.CancellationTokenSource _maleFaceMaskCts;
    private System.Threading.CancellationTokenSource _femaleFaceMaskCts;
    private System.Threading.CancellationTokenSource _qwenAnalysisCts;
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private System.Threading.CancellationTokenSource _developmentRunnerTestCts;
#endif

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
        if (CompareView != null)
        {
            CompareView.LocalMaskStroke -= OnLocalMaskStroke;
            CompareView.LocalMaskPaintingEnabled = false;
            CompareView.LocalMaskOverlay = null;
        }
        DestroyTexture(ref _identityLut);
        DestroyTexture(ref _selectedLut);
        DestroyTexture(ref _localAdjustmentMask);
        _localAdjustmentMaskPixels = null;
#if UNITY_EDITOR || DEVELOPMENT_BUILD
        CancelAndDisposeCts(ref _developmentRunnerTestCts);
#endif
        CancelAndDisposeCts(ref _qwenAnalysisCts);
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

    protected override bool ShowDirectionalSwitchButtons => true;

    protected override bool SwitchZoneControlsImageNavigation => true;

    protected override float GetSwitchPillAlignment01() => 0.5f;

    protected override bool HandleDirectionalImageNavigation(int direction)
    {
        return Host != null && Host.TryOpenAdjacentMainImage(direction);
    }

    protected override void BuildPage(VisualElement contentRoot)
    {
        PageRoot.RegisterCallback<KeyDownEvent>(OnPageKeyDown, TrickleDown.TrickleDown);
        BuildPageContent(contentRoot);
    }

    protected override void BuildPageNavigationControls(VisualElement root)
    {
        root.Add(CreateCornerNavigationButton(
            L("Library", "图库"),
            () => Host?.RequestPageSwitch(this, AppPageId.LibraryView, SwipeDirection.Left),
            true,
            L("Open the library", "打开图库")));
        root.Add(CreateCornerNavigationButton(
            L("Design", "设计"),
            () => Host?.RequestPageSwitch(this, AppPageId.DesignView, SwipeDirection.Right),
            false,
            L("Open design tools", "打开设计工具")));
    }

    private void OnPageKeyDown(KeyDownEvent evt)
    {
        if ((evt.keyCode != KeyCode.Delete && evt.keyCode != KeyCode.Backspace) || IsTextFieldFocused())
            return;

        DeleteSelectedHistoryEntry();
        evt.StopPropagation();
        evt.PreventDefault();
    }

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
            _adjustPanel.style.bottom = 158;
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
            _adjustPanel.style.bottom = 158;
            _adjustHost.style.paddingRight = 12;
            if (_adjustBody.style.display == DisplayStyle.None)
                SetAdjustPanelCollapsed(false, false);
        }

        _toolbarScroll?.schedule.Execute(() => _toolbarScroll.horizontalScroller.value = 0f);
        _presetScroll?.schedule.Execute(() => _presetScroll.horizontalScroller.value = 0f);
    }

    private void BuildPageContent(VisualElement contentRoot)
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
        BuildQwenAnalysisOverlay();

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
        ClearLocalMask();
        CancelAndDisposeCts(ref _faceMaskCts);
        Host?.FaceMaskGenerator?.ClearCurrentMask();
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
        shell.style.flexDirection = FlexDirection.Row;
        shell.style.alignItems = Align.Center;

        _toolbarScroll = new ScrollView(ScrollViewMode.Horizontal);
        _toolbarScroll.style.flexGrow = 1;
        _toolbarScroll.style.flexShrink = 1;
        _toolbarScroll.style.minWidth = 0;
        _toolbarScroll.style.marginRight = 8;
        _toolbarScroll.style.overflow = Overflow.Hidden;
        _toolbarScroll.horizontalScrollerVisibility = ScrollerVisibility.Hidden;
        _toolbarScroll.verticalScrollerVisibility = ScrollerVisibility.Hidden;
        shell.Add(_toolbarScroll);

        var row = new VisualElement();
        row.style.flexDirection = FlexDirection.Row;
        row.style.alignItems = Align.Center;
        row.style.left = -5;
        row.style.height = 56;
        _toolbarScroll.Add(row);

        Button AddTool(
            string title,
            string icon,
            Action onClick,
            Color? tint = null,
            float width = 50f,
            bool visibleInTopBar = false)
        {
            var button = new Button(onClick);
            button.tooltip = title;
            button.style.width = width;
            button.style.height = 50;
            button.style.marginRight = 2;
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
            iconLabel.style.fontSize = 28;
            iconLabel.style.bottom = -7;
            iconLabel.style.color = tint ?? Color.white;
            iconLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            button.Add(iconLabel);

            var textLabel = new Label(title);
            textLabel.style.top = -7;
            textLabel.style.fontSize = title.Length > 10 ? 12 : 16;
            textLabel.style.color = new Color(0.83f, 0.86f, 0.92f, 1f);
            textLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            button.Add(textLabel);
            if (visibleInTopBar)
                row.Add(button);
            return button;
        }

        AddTool(L("Fit", "适应"), "♻️", () => CompareView?.FitToView(), null, 50f, true);
        //AddTool("重置", "↺", () => CompareView?.ResetView());
        AddTool(L("Save", "保存"), "💾", OnSaveCurrentImage, new Color(0.37f, 0.78f, 1f), 50f, true);
        AddTool(L("Enhance", "清晰化"), "✨", OnOneClickSharpen, new Color(0.37f, 0.78f, 1f), 66f, true);
        AddTool(L("Face Repair", "修复人脸"), "🤦‍", OnOneClickFaceRepair, new Color(0.99f, 0.74f, 0.35f),66f, true);
        AddTool(L("Remove People", "去路人"), "🚶", OnOneClickRemovePassers, new Color(0.88f, 0.54f, 0.48f), 66f, true);
        AddTool(L("Background", "优化背景"), "🖼", OnOneClickOptimizeBackground, new Color(0.42f, 0.72f, 1f), 66f, true);
        AddTool(L("Auto Tone", "一键调色"), "🎨", OnOneClickAutoTone, new Color(0.44f, 0.82f, 0.68f), 66f, true);
        _qwenAnalysisButton = AddTool(
            L("Analyze", "识别内容"),
            "◉",
            OnQwenAnalyze,
            new Color(0.33f, 0.86f, 0.72f),
            66f,
            true);
        _qwenAnalysisButton.tooltip = L("Analyze the current history image with Qwen3.5", "使用 Qwen3.5 分析当前历史图像");
        //AddTool("浏览", "▣", OnBrowseOriginalImage);
        AddTool("CLIP", "✨", OnClipClassify, new Color(0.73f, 0.56f, 1f));
        //AddTool("换脸", "☺", OnFaceSwap, new Color(0.99f, 0.74f, 0.35f));
        //AddTool("清晰", "✦", OnSharpen);
        //AddTool("美白", "◌", OnWhiten);
        //AddTool("清白", "✧", OnSharpenWhiten);
        //AddTool("背景", "▥", OnChangeBackground);
        //AddTool("去雾", "≋", OnDehaze);
        //AddTool("调色", "◐", OnColorGrade);
        //AddTool("去雾调", "◑", OnDehazeColorGrade);
        AddTool("清晰", "👀", OnGpuSharpen);
        AddTool("ESR", "⚙", OnRealEsrganRepro);
        AddTool("YOLO", "🎉", OnYoloAndInpaintingRepro);
        AddTool("抠图", "🖼", OnMattingRepro);
        AddTool("CF", "🤦‍", OnCodeFormerRepro);
        AddTool("GFP", "🤦‍", OnGfpganRepro);

#if UNITY_EDITOR || DEVELOPMENT_BUILD
        _developmentRunnerTestButton = new Button(OnDevelopmentRunnerTest)
        {
            text = L("Test", "测试")
        };
        _developmentRunnerTestButton.tooltip = L(
            "Run all local development runner tests with the current image",
            "使用当前图片运行本地开发跑测");
        _developmentRunnerTestButton.style.width = 58;
        _developmentRunnerTestButton.style.height = 36;
        _developmentRunnerTestButton.style.marginLeft = 6;
        _developmentRunnerTestButton.style.paddingLeft = 0;
        _developmentRunnerTestButton.style.paddingRight = 0;
        _developmentRunnerTestButton.style.color = Color.white;
        _developmentRunnerTestButton.style.backgroundColor = new StyleColor(new Color(0.20f, 0.55f, 0.32f, 1f));
        _developmentRunnerTestButton.style.borderTopLeftRadius = 8;
        _developmentRunnerTestButton.style.borderTopRightRadius = 8;
        _developmentRunnerTestButton.style.borderBottomLeftRadius = 8;
        _developmentRunnerTestButton.style.borderBottomRightRadius = 8;
        _developmentRunnerTestButton.style.marginLeft = 8;
        shell.Add(_developmentRunnerTestButton);
#endif
        return shell;
    }

#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private void OnDevelopmentRunnerTest()
    {
        RunDevelopmentRunnerTestAsync().Forget();
    }

    private async UniTaskVoid RunDevelopmentRunnerTestAsync()
    {
        if (_developmentRunnerTestRunning || _aiRunning || _adjustRunning)
        {
            ShowToast(L("Another image operation is already running.", "已有图像任务正在运行。"), 2400);
            return;
        }
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
            return;

        var source = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (source == null)
        {
            ShowToast(L("Open an image before running the development test.", "请先打开图片再运行开发跑测。"), 2800);
            return;
        }

        CancelAndDisposeCts(ref _developmentRunnerTestCts);
        _developmentRunnerTestCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
        var operationCts = _developmentRunnerTestCts;
        _developmentRunnerTestRunning = true;
        _adjustRunning = true;
        _developmentRunnerTestButton?.SetEnabled(false);
        ShowProgress(L("Development runner test", "开发跑测"));

        try
        {
            var report = await AIImageDevelopmentRunnerTest.RunAsync(
                Host,
                source,
                CurrentImagePath,
                operationCts.Token,
                (completed, total, detail) =>
                {
                    SetProgress(total <= 0 ? 0f : completed / (float)total, detail);
                });
            SetProgress(1f, L("Development runner test complete", "开发跑测完成"));
            AIImageDevelopmentRunnerTest.RevealReport(report.reportPath);
            ShowToast(
                L("Runner report saved: ", "跑测报告已保存：") + report.reportPath,
                5200);
        }
        catch (OperationCanceledException)
        {
            ShowToast(L("Development runner test cancelled.", "开发跑测已取消。"), 2600);
        }
        catch (Exception exception)
        {
            UnityEngine.Debug.LogException(exception);
            ShowToast(L("Development runner test failed: ", "开发跑测失败：") + exception.Message, 4600);
        }
        finally
        {
            _developmentRunnerTestRunning = false;
            _adjustRunning = false;
            if (ReferenceEquals(_developmentRunnerTestCts, operationCts))
                _developmentRunnerTestCts = null;
            try { operationCts.Dispose(); } catch { }
            _developmentRunnerTestButton?.SetEnabled(true);
            HideProgress();
        }
    }
#endif

    private void BuildQwenAnalysisOverlay()
    {
        _qwenAnalysisOverlay = new VisualElement();
        _qwenAnalysisOverlay.style.position = Position.Absolute;
        _qwenAnalysisOverlay.style.left = 0;
        _qwenAnalysisOverlay.style.top = 0;
        _qwenAnalysisOverlay.style.right = 0;
        _qwenAnalysisOverlay.style.bottom = 0;
        _qwenAnalysisOverlay.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.68f));
        _qwenAnalysisOverlay.style.alignItems = Align.Center;
        _qwenAnalysisOverlay.style.justifyContent = Justify.Center;
        _qwenAnalysisOverlay.style.display = DisplayStyle.None;

        var panel = new VisualElement();
        panel.style.width = 720;
        panel.style.height = Length.Percent(76);
        panel.style.maxWidth = Length.Percent(92);
        panel.style.maxHeight = 720;
        panel.style.minHeight = 360;
        panel.style.backgroundColor = new StyleColor(new Color(0.10f, 0.11f, 0.14f, 0.99f));
        panel.style.borderTopLeftRadius = 8;
        panel.style.borderTopRightRadius = 8;
        panel.style.borderBottomLeftRadius = 8;
        panel.style.borderBottomRightRadius = 8;
        panel.style.paddingLeft = 18;
        panel.style.paddingRight = 18;
        panel.style.paddingTop = 16;
        panel.style.paddingBottom = 16;
        panel.style.flexDirection = FlexDirection.Column;
        _qwenAnalysisOverlay.Add(panel);

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.marginBottom = 12;
        panel.Add(header);

        var title = new Label("Qwen3.5 图像分析");
        title.style.flexGrow = 1;
        title.style.color = Color.white;
        title.style.fontSize = 17;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        header.Add(title);

        _qwenAnalysisCopyButton = new Button(CopyQwenAnalysisResult) { text = "复制" };
        _qwenAnalysisCopyButton.tooltip = "复制分析结果";
        _qwenAnalysisCopyButton.style.height = 32;
        _qwenAnalysisCopyButton.style.marginRight = 8;
        _qwenAnalysisCopyButton.style.paddingLeft = 12;
        _qwenAnalysisCopyButton.style.paddingRight = 12;
        _qwenAnalysisCopyButton.style.display = DisplayStyle.None;
        header.Add(_qwenAnalysisCopyButton);

        _qwenAnalysisCloseButton = new Button(HideQwenAnalysisOverlay) { text = "×" };
        _qwenAnalysisCloseButton.tooltip = "关闭";
        _qwenAnalysisCloseButton.style.width = 34;
        _qwenAnalysisCloseButton.style.height = 32;
        _qwenAnalysisCloseButton.style.fontSize = 18;
        _qwenAnalysisCloseButton.style.display = DisplayStyle.None;
        header.Add(_qwenAnalysisCloseButton);

        _qwenAnalysisStatus = new Label();
        _qwenAnalysisStatus.style.color = new Color(0.76f, 0.83f, 0.91f, 1f);
        _qwenAnalysisStatus.style.marginBottom = 8;
        panel.Add(_qwenAnalysisStatus);

        _qwenAnalysisProgress = new ProgressBar
        {
            lowValue = 0,
            highValue = 100,
            value = 0,
            title = "0%"
        };
        _qwenAnalysisProgress.style.height = 18;
        _qwenAnalysisProgress.style.marginBottom = 10;
        panel.Add(_qwenAnalysisProgress);

        var outputScroll = new ScrollView(ScrollViewMode.Vertical);
        outputScroll.style.flexGrow = 1;
        outputScroll.style.minHeight = 0;
        outputScroll.style.backgroundColor = new StyleColor(new Color(0.06f, 0.07f, 0.09f, 1f));
        outputScroll.style.borderTopLeftRadius = 6;
        outputScroll.style.borderTopRightRadius = 6;
        outputScroll.style.borderBottomLeftRadius = 6;
        outputScroll.style.borderBottomRightRadius = 6;
        outputScroll.style.paddingLeft = 14;
        outputScroll.style.paddingRight = 14;
        outputScroll.style.paddingTop = 12;
        outputScroll.style.paddingBottom = 12;
        panel.Add(outputScroll);

        _qwenAnalysisOutput = new Label();
        _qwenAnalysisOutput.enableRichText = false;
        _qwenAnalysisOutput.style.whiteSpace = WhiteSpace.Normal;
        _qwenAnalysisOutput.style.color = new Color(0.92f, 0.94f, 0.97f, 1f);
        _qwenAnalysisOutput.style.fontSize = 14;
        outputScroll.Add(_qwenAnalysisOutput);

        var footer = new VisualElement();
        footer.style.flexDirection = FlexDirection.Row;
        footer.style.justifyContent = Justify.FlexEnd;
        footer.style.marginTop = 12;
        panel.Add(footer);

        _qwenAnalysisCancelButton = new Button(CancelQwenAnalysis) { text = "取消" };
        _qwenAnalysisCancelButton.style.height = 34;
        _qwenAnalysisCancelButton.style.paddingLeft = 18;
        _qwenAnalysisCancelButton.style.paddingRight = 18;
        _qwenAnalysisCancelButton.style.backgroundColor = new StyleColor(new Color(0.68f, 0.22f, 0.24f, 1f));
        _qwenAnalysisCancelButton.style.color = Color.white;
        footer.Add(_qwenAnalysisCancelButton);

        PageRoot.Add(_qwenAnalysisOverlay);
    }

    private void OnQwenAnalyze()
    {
        AnalyzeCurrentImageWithQwenAsync().Forget();
    }

    private async UniTaskVoid AnalyzeCurrentImageWithQwenAsync()
    {
        if (_qwenAnalysisRunning || _aiRunning || _adjustRunning)
        {
            ShowToast("已有图像任务正在运行", 2200);
            return;
        }
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
            return;

        var source = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (source == null)
        {
            ShowToast("请先在历史记录中选择图像", 2200);
            return;
        }

        var modelDirectory = ResolveQwen35ModelDirectory();
        if (!HasQwen35ModelPayload(modelDirectory)
            && !await Host.EnsureModelGroupsAvailableAsync(
                "Qwen3.5 model download",
                _lifetimeCts.Token,
                ResolveQwen35ModelGroup()))
            return;

        modelDirectory = ResolveQwen35ModelDirectory();
        if (!HasQwen35ModelPayload(modelDirectory))
        {
            ShowToast(L("Qwen3.5 model files are incomplete: ", "Qwen3.5 模型文件不完整: ") + modelDirectory, 5000);
            return;
        }

        CancelAndDisposeCts(ref _qwenAnalysisCts);
        _qwenAnalysisCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
        var operationCts = _qwenAnalysisCts;
        var token = operationCts.Token;
        var streamedText = new StringBuilder(2048);
        var stopwatch = Stopwatch.StartNew();
        _qwenAnalysisRunning = true;
        _aiRunning = true;
        _qwenAnalysisButton?.SetEnabled(false);
        ShowQwenAnalysisOverlay();

        try
        {
            await UniTask.Yield(PlayerLoopTiming.Update, token);
            var detectedPersonCount = await DetectQwenPersonCountAsync(source, token);
            await UniTask.Yield(PlayerLoopTiming.Update, token);
            using (var runner = await Qwen35Runner.CreateAsync(
                modelDirectory,
                Qwen35AnalysisMaxNewTokens,
                true,
                token,
                progress => SetQwenPipelineProgress(progress, 0f, 10f)))
            {
                var result = await runner.GenerateImageAsync(
                    source,
                    BuildQwen35AnalysisPrompt(detectedPersonCount),
                    Qwen35SamplingConfig.Greedy(),
                    token,
                    (tokenId, piece) =>
                    {
                        streamedText.Append(piece);
                        _qwenAnalysisOutput.text = streamedText.ToString();
                    },
                    (completed, total) =>
                    {
                        var progress = 86f + 14f * completed / Mathf.Max(1, total);
                        SetQwenAnalysisProgress(progress, L("Generating " + completed + " / " + total, "正在生成 " + completed + " / " + total));
                    },
                    null,
                    progress => SetQwenPipelineProgress(progress, 10f, 100f));

                var finalText = string.IsNullOrWhiteSpace(result.Text)
                    ? streamedText.ToString().Trim()
                    : result.Text.Trim();
                _qwenAnalysisOutput.text = finalText;
                SetQwenAnalysisProgress(100f, "分析完成");
                _qwenAnalysisStatus.text = L(
                    "Current history image · " + AppLocalization.Translate(GetCurrentHistoryLabel())
                    + " · " + stopwatch.Elapsed.TotalSeconds.ToString("0.0") + " s",
                    "当前历史图像 · " + GetCurrentHistoryLabel()
                    + " · " + stopwatch.Elapsed.TotalSeconds.ToString("0.0") + " 秒");
            }
        }
        catch (OperationCanceledException)
        {
            _qwenAnalysisStatus.text = AppLocalization.Translate("分析已取消");
        }
        catch (Exception exception)
        {
            _qwenAnalysisStatus.text = AppLocalization.Translate("分析失败");
            var partial = streamedText.ToString().Trim();
            _qwenAnalysisOutput.text = string.IsNullOrEmpty(partial)
                ? exception.Message
                : partial + "\n\n" + exception.Message;
            UnityEngine.Debug.LogException(exception);
        }
        finally
        {
            stopwatch.Stop();
            _qwenAnalysisRunning = false;
            _aiRunning = false;
            if (ReferenceEquals(_qwenAnalysisCts, operationCts))
                _qwenAnalysisCts = null;
            try { operationCts.Dispose(); } catch { }
            _qwenAnalysisButton?.SetEnabled(true);
            _qwenAnalysisCancelButton.style.display = DisplayStyle.None;
            _qwenAnalysisCloseButton.style.display = DisplayStyle.Flex;
            _qwenAnalysisCopyButton.style.display = string.IsNullOrWhiteSpace(_qwenAnalysisOutput.text)
                ? DisplayStyle.None
                : DisplayStyle.Flex;
        }
    }

    private static string BuildQwen35AnalysisPrompt(int detectedPersonCount)
    {
        var prompt = Qwen35DirectResponseInstruction;
        if (detectedPersonCount > 0)
        {
            prompt += "本地人物检测已确认图中有 " + detectedPersonCount
                + " 个独立人物；请以这个人数为准完成逐人描述。\n";
        }
        return prompt + Qwen35AnalysisPrompt;
    }

    private async UniTask<int> DetectQwenPersonCountAsync(Texture2D source, CancellationToken token)
    {
        var runner = Host?.YoloSegRunner;
        if (runner == null || source == null) return 0;

        var result = default(YoloSegResult);
        var originalTargetPersonOnly = runner.targetPersonOnly;
        try
        {
            runner.targetPersonOnly = true;
            result = await runner.ProcessAsync(source, token);
            token.ThrowIfCancellationRequested();
            return string.IsNullOrWhiteSpace(result.error) ? Mathf.Max(0, result.personCount) : 0;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            UnityEngine.Debug.LogWarning("Qwen3.5 person-count assist failed: " + exception.Message);
            return 0;
        }
        finally
        {
            runner.targetPersonOnly = originalTargetPersonOnly;
            DestroyQwenPersonCountTextures(result);
            runner.ReleaseRuntimeResources();
        }
    }

    private static void DestroyQwenPersonCountTextures(YoloSegResult result)
    {
        if (result.texture != null) UnityEngine.Object.Destroy(result.texture);
        if (result.mask != null) UnityEngine.Object.Destroy(result.mask);
        if (result.overlay != null) UnityEngine.Object.Destroy(result.overlay);
    }

    private void ShowQwenAnalysisOverlay()
    {
        _qwenAnalysisOutput.text = string.Empty;
        _qwenAnalysisStatus.text = AppLocalization.Translate("准备分析当前历史图像");
        _qwenAnalysisProgress.value = 0;
        _qwenAnalysisProgress.title = "0%";
        _qwenAnalysisCancelButton?.SetEnabled(true);
        _qwenAnalysisCancelButton.style.display = DisplayStyle.Flex;
        _qwenAnalysisCloseButton.style.display = DisplayStyle.None;
        _qwenAnalysisCopyButton.style.display = DisplayStyle.None;
        _qwenAnalysisOverlay.style.display = DisplayStyle.Flex;
        _qwenAnalysisOverlay.BringToFront();
    }

    private void SetQwenAnalysisStage(string stage)
    {
        switch (stage)
        {
            case "loading_vision": SetQwenAnalysisProgress(2f, "正在加载视觉模型"); break;
            case "encoding_image": SetQwenAnalysisProgress(8f, "正在编码图像"); break;
            case "loading_decoder": SetQwenAnalysisProgress(12f, "正在加载语言模型"); break;
            case "generating": SetQwenAnalysisProgress(20f, "正在生成分析"); break;
        }
    }

    private void SetQwenPipelineProgress(Qwen35Progress progress, float start, float end)
    {
        var value = Mathf.Lerp(start, end, progress.Progress01);
        string status;
        switch (progress.Stage)
        {
            case "validating_assets": status = "正在校验模型文件"; break;
            case "validating_contract": status = "正在校验模型结构"; break;
            case "loading_tokenizer": status = "正在加载分词器"; break;
            case "loading_vision": status = "正在加载视觉模型"; break;
            case "encoding_image": status = "正在编码图像"; break;
            case "loading_decoder": status = "正在加载语言模型"; break;
            case "prefill": status = "正在处理图像与提示词"; break;
            case "generating": status = "正在生成分析"; break;
            default: status = "正在准备 Qwen3.5"; break;
        }
        if (!string.IsNullOrWhiteSpace(progress.Detail))
            status += " · " + progress.Detail;
        SetQwenAnalysisProgress(value, status);
    }

    private void SetQwenAnalysisProgress(float progress, string status)
    {
        var value = Mathf.Clamp(progress, 0f, 100f);
        if (_qwenAnalysisRunning && _qwenAnalysisProgress != null)
            value = Mathf.Max(value, _qwenAnalysisProgress.value);
        _qwenAnalysisProgress.value = value;
        _qwenAnalysisProgress.title = Mathf.RoundToInt(value) + "%";
        _qwenAnalysisStatus.text = AppLocalization.Translate(status ?? string.Empty);
    }

    private void CancelQwenAnalysis()
    {
        if (!_qwenAnalysisRunning)
            return;
        _qwenAnalysisCancelButton?.SetEnabled(false);
        _qwenAnalysisStatus.text = AppLocalization.Translate("正在取消");
        try { _qwenAnalysisCts?.Cancel(); } catch { }
    }

    private void HideQwenAnalysisOverlay()
    {
        if (_qwenAnalysisRunning)
            return;
        if (_qwenAnalysisOverlay != null)
            _qwenAnalysisOverlay.style.display = DisplayStyle.None;
    }

    private void CopyQwenAnalysisResult()
    {
        var text = _qwenAnalysisOutput?.text;
        if (string.IsNullOrWhiteSpace(text))
            return;
        CrossPlatformClipboard.Copy(text);
        ShowToast("分析结果已复制", 1600);
    }

    private static string ResolveQwen35ModelDirectory()
    {
        var group = ResolveQwen35ModelGroup();
        var modelDirectoryName = group == AIImageModelGroupId.Qwen35FullPrecision
            ? "qwen3.5_0.8b"
            : group == AIImageModelGroupId.Qwen35MobileQ8
                ? "qwen3.5_0.8b_mobile_q8"
                : "qwen3.5_0.8b_mobile_q4";
        var configured = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            var configuredDirectory = Qwen35ModelDirectoryResolver.Resolve(configured, modelDirectoryName);
            if (HasQwen35ModelPayload(configuredDirectory))
                return configuredDirectory;
        }

        var persistentDirectory = Path.Combine(Application.persistentDataPath, modelDirectoryName);
        if (HasQwen35ModelPayload(persistentDirectory))
            return persistentDirectory;

        var deliveredDirectory = Path.Combine(
            AIImageModelDelivery.PersistentRoot,
            "QWEN35",
            modelDirectoryName);
        if (HasQwen35ModelPayload(deliveredDirectory))
            return deliveredDirectory;

        var playerDirectory = Path.Combine(Application.streamingAssetsPath, "QWEN35", modelDirectoryName);
        if (HasQwen35ModelPayload(playerDirectory))
            return playerDirectory;

        if (Aexis.Samples.AexisSampleStreamingAssets.TryResolveDirectoryPath("QWEN35", out var streamingAssetsDirectory))
        {
            var deployedDirectory = Qwen35ModelDirectoryResolver.Resolve(streamingAssetsDirectory, modelDirectoryName);
            if (HasQwen35ModelPayload(deployedDirectory))
                return deployedDirectory;
        }

#if UNITY_EDITOR
        var projectDirectory = Path.GetFullPath(Path.Combine(
            Application.dataPath,
            "..",
            "Tools",
            "Qwen35NcnnBaseline",
            "_models",
            modelDirectoryName));
        if (HasQwen35ModelPayload(projectDirectory))
            return projectDirectory;
#endif
        return persistentDirectory;
    }

    private static bool HasQwen35ModelPayload(string modelDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory) || !Directory.Exists(modelDirectory))
            return false;

        // Both mobile precisions are delivered as manifest-backed, sharded
        // assets.  Do not send Q4 through the legacy loose-.bin check: that
        // falsely reports its weights missing even after Android has copied
        // every bundled Q4 shard into persistent storage.
        if (File.Exists(Path.Combine(modelDirectory, Qwen35MobileAssetSet.Q4ManifestFileName))
            || File.Exists(Path.Combine(modelDirectory, Qwen35MobileAssetSet.ManifestFileName)))
        {
            try
            {
                // A stale interrupted download can leave only the manifest in the
                // persistent path. Do not let it shadow the complete Player copy.
                return Qwen35MobileAssetSet.TryLoad(modelDirectory) != null;
            }
            catch
            {
                return false;
            }
        }

        var requiredFiles = new[]
        {
            "model.json",
            "vocab.txt",
            "merges.txt",
            "qwen3.5_decoder.ncnn.param",
            "qwen3.5_decoder.ncnn.bin",
            "qwen3.5_embed_token.ncnn.param",
            "qwen3.5_embed_token.ncnn.bin",
            "qwen3.5_proj_out.ncnn.param",
            "qwen3.5_vision_embed_patch.ncnn.param",
            "qwen3.5_vision_embed_patch.ncnn.bin",
            "qwen3.5_vision_embed_pos.ncnn.param",
            "qwen3.5_vision_embed_pos.ncnn.bin",
            "qwen3.5_vision_encoder.ncnn.param",
            "qwen3.5_vision_encoder.ncnn.bin"
        };

        for (var index = 0; index < requiredFiles.Length; index++)
        {
            if (!File.Exists(Path.Combine(modelDirectory, requiredFiles[index])))
                return false;
        }

        return true;
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

        var title = new Label("调节");
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

        //scroll.Add(BuildReferenceButtons());
        //scroll.Add(BuildProviderEditor());
        //scroll.Add(BuildToggleRow("去反光", value => _appendDeGlarePrompt = value, false));
        //scroll.Add(BuildToggleRow("去背景人物", value => _appendRemoveBgPeoplePrompt = value, true));
        //scroll.Add(BuildToggleRow("调试输出", value => _gpuSharpenDumpStages = value, false));

        scroll.Add(CreateAdjustRow("对比度", -0.5f, 0.5f, 0f, "AdjustContrast", (cs, v) => cs.SetFloat("_Contrast", v), v => $"对比度 {v:0.00}"));
        scroll.Add(CreateAdjustRow("亮度", -0.5f, 0.5f, 0f, "AdjustBrightness", (cs, v) => cs.SetFloat("_Brightness", v), v => $"亮度 {v:0.00}"));
        scroll.Add(CreateAdjustRow("自然饱和度", -1f, 1f, 0f, "AdjustVibrance", (cs, v) => cs.SetFloat("_Vibrance", v), v => $"自然饱和度 {v:0.00}"));
        scroll.Add(CreateAdjustRow("去阴影", 0f, 0.5f, 0f, "AdjustShadows", (cs, v) => cs.SetFloat("_Shadows", v), v => $"去阴影 {v:0.00}"));
        scroll.Add(CreateAdjustRow("去高光", 0f, 0.5f, 0f, "AdjustHighlights", (cs, v) => cs.SetFloat("_Highlights", v), v => $"去高光 {v:0.00}"));
        scroll.Add(CreateAdjustRow("暖色滤镜", 0f, 1f, 0f, "WarmFilter", (cs, v) => cs.SetFloat("_Warm", v), v => $"暖色 {v:0.00}"));
        scroll.Add(CreateAdjustRow("冷色滤镜", 0f, 1f, 0f, "CoolFilter", (cs, v) => cs.SetFloat("_Cool", v), v => $"冷色 {v:0.00}"));
        scroll.Add(CreateAdjustRow("锐化", 0f, 4f, 0f, "Sharpen", (cs, v) => cs.SetFloat("_Sharpen", v), v => $"锐化 {v:0.00}"));
        scroll.Add(CreateAdjustRow("模糊", 0f, 4f, 0f, "Blur", (cs, v) => cs.SetFloat("_Blur", v), v => $"模糊 {v:0.00}"));
        scroll.Add(CreateAdjustActionCard("自动增强", "应用", ApplyAutoEnhance));
        scroll.Add(CreateAdjustRow("白平衡 色温", -100f, 100f, 0f, "WhiteBalance", (cs, v) =>
        {
            cs.SetFloat("_WhiteBalanceTemperature", v);
            cs.SetFloat("_WhiteBalanceTint", 0f);
        }, v => $"色温 {v:+0;-0;0}"));
        scroll.Add(CreateAdjustRow("白平衡 色调", -100f, 100f, 0f, "WhiteBalance", (cs, v) =>
        {
            cs.SetFloat("_WhiteBalanceTemperature", 0f);
            cs.SetFloat("_WhiteBalanceTint", v);
        }, v => $"色调 {v:+0;-0;0}"));
        scroll.Add(CreateAdjustRow("曝光", -4f, 4f, 0f, "AdjustExposure", (cs, v) => cs.SetFloat("_Exposure", v), v => $"曝光 {v:+0.00;-0.00;0.00} EV"));
        scroll.Add(CreateAdjustRow("曲线", -1f, 1f, 0f, "ToneCurve", (cs, v) => cs.SetFloat("_CurveAmount", v), v => $"曲线 {v:+0.00;-0.00;0.00}"));
        scroll.Add(CreateAdjustRow("颗粒", 0f, 1f, 0f, "AddGrain", (cs, v) => cs.SetFloat("_GrainAmount", v), v => $"颗粒 {v:0.00}"));
        scroll.Add(CreateAdjustRow("暗角", -1f, 1f, 0f, "Vignette", (cs, v) => cs.SetFloat("_VignetteAmount", v), v => $"暗角 {v:+0.00;-0.00;0.00}"));

        var advancedAdjustments = new VisualElement();
        advancedAdjustments.style.display = DisplayStyle.None;
        scroll.Add(BuildToggleRow("高级调色", value =>
        {
            advancedAdjustments.style.display = value ? DisplayStyle.Flex : DisplayStyle.None;
        }, false));
        scroll.Add(advancedAdjustments);

        advancedAdjustments.Add(CreateAdjustRow("色阶 黑场", 0f, 0.50f, 0f, "LevelsBlack", (cs, v) => cs.SetFloat("_LevelBlack", v), v => $"黑场 {v:0.00}"));
        advancedAdjustments.Add(CreateAdjustRow("色阶 白场", 0.50f, 1f, 1f, "LevelsWhite", (cs, v) => cs.SetFloat("_LevelWhite", v), v => $"白场 {v:0.00}"));
        AddHslAdjustmentRows(advancedAdjustments, "红色", 0f);
        AddHslAdjustmentRows(advancedAdjustments, "橙色", 30f / 360f);
        AddHslAdjustmentRows(advancedAdjustments, "黄色", 60f / 360f);
        AddHslAdjustmentRows(advancedAdjustments, "绿色", 120f / 360f);
        AddHslAdjustmentRows(advancedAdjustments, "青色", 180f / 360f);
        AddHslAdjustmentRows(advancedAdjustments, "蓝色", 240f / 360f);
        AddHslAdjustmentRows(advancedAdjustments, "紫色", 280f / 360f);
        AddHslAdjustmentRows(advancedAdjustments, "洋红", 330f / 360f);
        advancedAdjustments.Add(BuildLutCard());
        advancedAdjustments.Add(CreateAdjustRow("LUT 强度", 0f, 1f, 0f, "ApplyLut", ConfigureLut, v => $"LUT {v:0.00}"));
        advancedAdjustments.Add(BuildLocalMaskPanel());
        advancedAdjustments.Add(CreateAdjustRow("局部蒙版 曝光", -4f, 4f, 0f, "LocalMaskExposure", ConfigureLocalMaskExposure, v => $"局部曝光 {v:+0.00;-0.00;0.00} EV"));
        return panel;
    }

    private VisualElement CreateAdjustActionCard(string titleText, string buttonText, Action onClick)
    {
        var card = new VisualElement();
        card.style.flexDirection = FlexDirection.Row;
        card.style.alignItems = Align.Center;
        card.style.backgroundColor = new StyleColor(new Color(0.17f, 0.18f, 0.22f, 0.96f));
        card.style.borderTopLeftRadius = 14;
        card.style.borderTopRightRadius = 14;
        card.style.borderBottomLeftRadius = 14;
        card.style.borderBottomRightRadius = 14;
        card.style.paddingLeft = 12;
        card.style.paddingRight = 12;
        card.style.paddingTop = 10;
        card.style.paddingBottom = 10;
        card.style.marginBottom = 8;

        var title = new Label(titleText);
        title.style.flexGrow = 1;
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        card.Add(title);

        var button = new Button(onClick) { text = buttonText };
        button.style.height = 30;
        button.style.paddingLeft = 12;
        button.style.paddingRight = 12;
        card.Add(button);
        return card;
    }

    private void AddHslAdjustmentRows(VisualElement host, string colorName, float hueCenter)
    {
        host.Add(CreateAdjustRow(
            $"HSL {colorName} 色相",
            -180f,
            180f,
            0f,
            "HslHue",
            (cs, value) =>
            {
                cs.SetFloat("_HslCenter", hueCenter);
                cs.SetFloat("_HslAmount", value);
            },
            value => $"{colorName} 色相 {value:+0;-0;0}"));
        host.Add(CreateAdjustRow(
            $"HSL {colorName} 饱和度",
            -1f,
            1f,
            0f,
            "HslSaturation",
            (cs, value) =>
            {
                cs.SetFloat("_HslCenter", hueCenter);
                cs.SetFloat("_HslAmount", value);
            },
            value => $"{colorName} 饱和度 {value:+0.00;-0.00;0.00}"));
        host.Add(CreateAdjustRow(
            $"HSL {colorName} 明亮度",
            -1f,
            1f,
            0f,
            "HslLuminance",
            (cs, value) =>
            {
                cs.SetFloat("_HslCenter", hueCenter);
                cs.SetFloat("_HslAmount", value);
            },
            value => $"{colorName} 明亮度 {value:+0.00;-0.00;0.00}"));
    }

    private void ApplyAutoEnhance()
    {
        var source = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (source == null)
            return;

        Color32[] pixels;
        try
        {
            pixels = source.GetPixels32();
        }
        catch
        {
            ShowToast("当前图像无法读取自动增强所需的亮度数据", 2400);
            return;
        }

        if (pixels == null || pixels.Length == 0)
            return;

        var stride = Mathf.Max(1, pixels.Length / 65536);
        var sum = 0f;
        var sumSquares = 0f;
        var count = 0;
        for (var i = 0; i < pixels.Length; i += stride)
        {
            var pixel = pixels[i];
            var luminance = (pixel.r * 0.2126f + pixel.g * 0.7152f + pixel.b * 0.0722f) / 255f;
            sum += luminance;
            sumSquares += luminance * luminance;
            count++;
        }

        if (count == 0)
            return;

        var mean = sum / count;
        var standardDeviation = Mathf.Sqrt(Mathf.Max(0f, sumSquares / count - mean * mean));
        var exposure = Mathf.Clamp(Mathf.Log(0.45f / Mathf.Max(0.03f, mean), 2f), -1.25f, 1.25f);
        var contrast = Mathf.Clamp((0.18f - standardDeviation) * 1.2f, -0.10f, 0.22f);
        var vibrance = Mathf.Clamp((0.22f - standardDeviation) * 0.55f, 0f, 0.15f);
        ApplyComputeAdjustmentAsync(
            "AutoEnhance",
            cs =>
            {
                cs.SetFloat("_AutoExposure", exposure);
                cs.SetFloat("_AutoContrast", contrast);
                cs.SetFloat("_AutoVibrance", vibrance);
            },
            "自动增强").Forget();
    }

    private VisualElement BuildLutCard()
    {
        var card = CreateAdjustActionCard("LUT", "选择 .cube", () => SelectLutAsync().Forget());
        _lutPickerButton = card.Q<Button>();
        return card;
    }

    private async UniTaskVoid SelectLutAsync()
    {
        if (Host?.FileDialog == null)
            return;

        var path = await Host.FileDialog.ShowOpenLutAsync();
        if (string.IsNullOrWhiteSpace(path))
            return;

        if (!TryLoadCubeLut(path, out var texture, out var size, out var domainMin, out var domainMax, out var error))
        {
            ShowToast(string.IsNullOrWhiteSpace(error) ? "无法读取 LUT 文件" : error, 2800);
            return;
        }

        DestroyTexture(ref _selectedLut);
        _selectedLut = texture;
        _selectedLutSize = size;
        _selectedLutDomainMin = domainMin;
        _selectedLutDomainMax = domainMax;
        if (_lutPickerButton != null)
            _lutPickerButton.text = Path.GetFileName(path);
        ShowToast("已加载 LUT", 1600);
    }

    private void ConfigureLut(ComputeShader shader, int kernel, float intensity)
    {
        var lut = _selectedLut ?? EnsureIdentityLut();
        shader.SetTexture(kernel, "_Lut3D", lut);
        shader.SetInt("_LutSize", _selectedLut != null ? _selectedLutSize : 16);
        var domainMin = _selectedLut != null ? _selectedLutDomainMin : Vector3.zero;
        var domainMax = _selectedLut != null ? _selectedLutDomainMax : Vector3.one;
        shader.SetVector("_LutDomainMin", new Vector4(domainMin.x, domainMin.y, domainMin.z, 0f));
        shader.SetVector("_LutDomainMax", new Vector4(domainMax.x, domainMax.y, domainMax.z, 0f));
        shader.SetFloat("_LutIntensity", intensity);
    }

    private Texture3D EnsureIdentityLut()
    {
        if (_identityLut != null)
            return _identityLut;

        const int size = 16;
        var colors = new Color[size * size * size];
        for (var blue = 0; blue < size; blue++)
        {
            for (var green = 0; green < size; green++)
            {
                for (var red = 0; red < size; red++)
                {
                    var index = red + size * (green + size * blue);
                    colors[index] = new Color(
                        red / (float)(size - 1),
                        green / (float)(size - 1),
                        blue / (float)(size - 1),
                        1f);
                }
            }
        }

        _identityLut = new Texture3D(size, size, size, TextureFormat.RGBA32, false)
        {
            name = "IdentityLut",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        _identityLut.SetPixels(colors);
        _identityLut.Apply(false, true);
        return _identityLut;
    }

    private static bool TryLoadCubeLut(
        string path,
        out Texture3D texture,
        out int size,
        out Vector3 domainMin,
        out Vector3 domainMax,
        out string error)
    {
        texture = null;
        size = 0;
        domainMin = Vector3.zero;
        domainMax = Vector3.one;
        error = null;

        try
        {
            var colors = new List<Color>();
            foreach (var sourceLine in File.ReadLines(path))
            {
                var line = sourceLine;
                var commentIndex = line.IndexOf('#');
                if (commentIndex >= 0)
                    line = line.Substring(0, commentIndex);
                line = line.Trim();
                if (line.Length == 0)
                    continue;

                var tokens = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                if (tokens.Length == 0)
                    continue;

                if (string.Equals(tokens[0], "TITLE", StringComparison.OrdinalIgnoreCase))
                    continue;

                if (string.Equals(tokens[0], "LUT_3D_SIZE", StringComparison.OrdinalIgnoreCase))
                {
                    if (tokens.Length != 2 || !int.TryParse(tokens[1], out size) || size < 2 || size > 64)
                    {
                        error = "LUT_3D_SIZE 必须在 2 到 64 之间";
                        return false;
                    }
                    continue;
                }

                if (string.Equals(tokens[0], "DOMAIN_MIN", StringComparison.OrdinalIgnoreCase))
                {
                    if (!TryParseCubeVector(tokens, out domainMin))
                    {
                        error = "DOMAIN_MIN 格式无效";
                        return false;
                    }
                    continue;
                }

                if (string.Equals(tokens[0], "DOMAIN_MAX", StringComparison.OrdinalIgnoreCase))
                {
                    if (!TryParseCubeVector(tokens, out domainMax))
                    {
                        error = "DOMAIN_MAX 格式无效";
                        return false;
                    }
                    continue;
                }

                if (!TryParseCubeColor(tokens, out var color))
                {
                    error = "LUT 颜色数据格式无效";
                    return false;
                }
                colors.Add(color);
            }

            if (size < 2 || colors.Count != size * size * size)
            {
                error = "LUT 颜色数据数量与 LUT_3D_SIZE 不匹配";
                return false;
            }

            if (domainMax.x <= domainMin.x || domainMax.y <= domainMin.y || domainMax.z <= domainMin.z)
            {
                error = "LUT DOMAIN 范围无效";
                return false;
            }

            texture = new Texture3D(size, size, size, TextureFormat.RGBA32, false)
            {
                name = Path.GetFileNameWithoutExtension(path),
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            texture.SetPixels(colors.ToArray());
            texture.Apply(false, true);
            return true;
        }
        catch (Exception exception)
        {
            if (texture != null)
                Destroy(texture);
            texture = null;
            error = exception.Message;
            return false;
        }
    }

    private static bool TryParseCubeVector(string[] tokens, out Vector3 value)
    {
        value = Vector3.zero;
        if (tokens.Length != 4)
            return false;

        return float.TryParse(tokens[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out value.x) &&
               float.TryParse(tokens[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out value.y) &&
               float.TryParse(tokens[3], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out value.z);
    }

    private static bool TryParseCubeColor(string[] tokens, out Color color)
    {
        color = Color.black;
        if (tokens.Length != 3)
            return false;

        if (!float.TryParse(tokens[0], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var red) ||
            !float.TryParse(tokens[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var green) ||
            !float.TryParse(tokens[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var blue))
            return false;

        color = new Color(red, green, blue, 1f);
        return true;
    }

    private VisualElement BuildLocalMaskPanel()
    {
        var card = new VisualElement();
        card.style.backgroundColor = new StyleColor(new Color(0.17f, 0.18f, 0.22f, 0.96f));
        card.style.borderTopLeftRadius = 14;
        card.style.borderTopRightRadius = 14;
        card.style.borderBottomLeftRadius = 14;
        card.style.borderBottomRightRadius = 14;
        card.style.paddingLeft = 12;
        card.style.paddingRight = 12;
        card.style.paddingTop = 10;
        card.style.paddingBottom = 10;
        card.style.marginBottom = 8;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        card.Add(header);

        var label = new Label("局部蒙版");
        label.style.flexGrow = 1;
        label.style.color = Color.white;
        label.style.unityFontStyleAndWeight = FontStyle.Bold;
        header.Add(label);

        _localMaskPaintButton = new Button(ToggleLocalMaskPainting) { text = "绘制" };
        _localMaskPaintButton.style.height = 30;
        _localMaskPaintButton.style.paddingLeft = 12;
        _localMaskPaintButton.style.paddingRight = 12;
        header.Add(_localMaskPaintButton);

        var clearButton = new Button(ClearLocalMask) { text = "清除" };
        clearButton.style.height = 30;
        clearButton.style.marginLeft = 6;
        clearButton.style.paddingLeft = 12;
        clearButton.style.paddingRight = 12;
        header.Add(clearButton);

        var brushLabel = new Label("画笔大小");
        brushLabel.style.marginTop = 8;
        brushLabel.style.color = new Color(0.78f, 0.88f, 1f, 1f);
        card.Add(brushLabel);

        var brushSlider = new Slider(0.01f, 0.35f) { value = 0.08f };
        brushSlider.RegisterValueChangedCallback(evt =>
        {
            if (CompareView != null)
                CompareView.LocalMaskBrushSize = evt.newValue;
        });
        card.Add(brushSlider);

        if (CompareView != null)
        {
            CompareView.LocalMaskStroke -= OnLocalMaskStroke;
            CompareView.LocalMaskStroke += OnLocalMaskStroke;
            CompareView.LocalMaskBrushSize = brushSlider.value;
        }

        return card;
    }

    private void ToggleLocalMaskPainting()
    {
        if (CompareView == null || !EnsureLocalMask())
            return;

        _localMaskPainting = !_localMaskPainting;
        _hasLastLocalMaskPoint = false;
        CompareView.LocalMaskPaintingEnabled = _localMaskPainting;
        CompareView.LocalMaskOverlay = _localMaskPainting ? _localAdjustmentMask : null;
        CompareView.MarkDirtyRepaint();
        if (_localMaskPaintButton != null)
            _localMaskPaintButton.text = _localMaskPainting ? "完成" : "绘制";
    }

    private bool EnsureLocalMask()
    {
        var source = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (source == null)
            return false;

        if (_localAdjustmentMask != null &&
            _localAdjustmentMask.width == source.width &&
            _localAdjustmentMask.height == source.height)
        {
            return true;
        }

        DestroyTexture(ref _localAdjustmentMask);
        _localAdjustmentMaskPixels = new Color32[source.width * source.height];
        _localAdjustmentMask = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, true)
        {
            name = "LocalAdjustmentMask",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        _localAdjustmentMask.SetPixels32(_localAdjustmentMaskPixels);
        _localAdjustmentMask.Apply(false, false);
        return true;
    }

    private void OnLocalMaskStroke(Vector2 uv, float brushSize, bool strokeStart)
    {
        if (!_localMaskPainting || !EnsureLocalMask())
            return;

        var previous = strokeStart || !_hasLastLocalMaskPoint ? uv : _lastLocalMaskPoint;
        var radius = Mathf.Max(1, Mathf.RoundToInt(brushSize * Mathf.Max(_localAdjustmentMask.width, _localAdjustmentMask.height) * 0.5f));
        var steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(previous, uv) * Mathf.Max(_localAdjustmentMask.width, _localAdjustmentMask.height) / Mathf.Max(1f, radius * 0.5f)));
        for (var step = 0; step <= steps; step++)
            PaintLocalMaskCircle(Vector2.Lerp(previous, uv, step / (float)steps), radius);

        _localAdjustmentMask.SetPixels32(_localAdjustmentMaskPixels);
        _localAdjustmentMask.Apply(false, false);
        _lastLocalMaskPoint = uv;
        _hasLastLocalMaskPoint = true;
        CompareView?.MarkDirtyRepaint();
    }

    private void PaintLocalMaskCircle(Vector2 uv, int radius)
    {
        var width = _localAdjustmentMask.width;
        var height = _localAdjustmentMask.height;
        var centerX = Mathf.RoundToInt(uv.x * (width - 1));
        var centerY = Mathf.RoundToInt(uv.y * (height - 1));
        var radiusSquared = radius * radius;
        var minX = Mathf.Max(0, centerX - radius);
        var maxX = Mathf.Min(width - 1, centerX + radius);
        var minY = Mathf.Max(0, centerY - radius);
        var maxY = Mathf.Min(height - 1, centerY + radius);
        for (var y = minY; y <= maxY; y++)
        {
            var dy = y - centerY;
            for (var x = minX; x <= maxX; x++)
            {
                var dx = x - centerX;
                if (dx * dx + dy * dy <= radiusSquared)
                    _localAdjustmentMaskPixels[y * width + x] = new Color32(255, 255, 255, 255);
            }
        }
    }

    private void ConfigureLocalMaskExposure(ComputeShader shader, int kernel, float exposure)
    {
        if (!EnsureLocalMask())
            return;

        shader.SetTexture(kernel, "_LocalMask", _localAdjustmentMask);
        shader.SetFloat("_LocalExposure", exposure);
    }

    private void ClearLocalMask()
    {
        _localMaskPainting = false;
        _hasLastLocalMaskPoint = false;
        DestroyTexture(ref _localAdjustmentMask);
        _localAdjustmentMaskPixels = null;
        if (CompareView != null)
        {
            CompareView.LocalMaskPaintingEnabled = false;
            CompareView.LocalMaskOverlay = null;
            CompareView.MarkDirtyRepaint();
        }
        if (_localMaskPaintButton != null)
            _localMaskPaintButton.text = "绘制";
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
        CrossPlatformClipboard.EnableTextFieldClipboard(_apiKeyField);
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
            _panelToggleButton.text = AppLocalization.Translate(collapsed ? "展开调节" : "收起调节");
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

        if (!await Host.EnsureModelGroupsAvailableAsync(
                "CLIP model download",
                _lifetimeCts.Token,
                AIImageModelGroupId.ClipMobileClipS0))
            return;

        _adjustRunning = true;
        ShowProgress("CLIP 分类");
        try
        {
            var currentPath = CurrentImagePath;
            var preferFileIdentity = GetCurrentHistoryTexture() == GetOriginalHistoryTexture() && !string.IsNullOrWhiteSpace(currentPath);
            ClipClassificationResult result;
            void OnProgress(float p, string t) => SetProgress(p, t);
            Host.ClipRunner.ProgressChanged -= OnProgress;
            Host.ClipRunner.ProgressChanged += OnProgress;
            try
            {
                result = await Host.ClipRunner.ProcessAsync(src, _lifetimeCts.Token);
            }
            finally
            {
                Host.ClipRunner.ProgressChanged -= OnProgress;
            }

            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3200);
                return;
            }

            ClipClassificationCache.Store(Host.ClipRunner, result, src, currentPath, preferFileIdentity);
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
    private void OnOneClickSharpen() => ApplyRealEsrganReproAsync().Forget();
    private void OnOneClickFaceRepair() => ApplyOneClickFaceRepairAsync().Forget();
    private void OnOneClickRemovePassers() => ApplyOneClickRemovePassersAsync().Forget();
    private void OnOneClickOptimizeBackground() => ApplyOneClickOptimizeBackgroundAsync().Forget();
    private void OnOneClickAutoTone() => ApplyOneClickAutoToneAsync().Forget();

    private async UniTaskVoid ApplyOneClickAutoToneAsync()
    {
        var clipRunner = Host?.ClipRunner;
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || clipRunner == null)
            return;

        var source = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (source == null)
            return;

        if (!await Host.EnsureModelGroupsAvailableAsync(
                "CLIP model download",
                _lifetimeCts.Token,
                AIImageModelGroupId.ClipMobileClipS0))
            return;

        _adjustRunning = true;
        ShowProgress(L("Auto tone", "一键调色"));
        try
        {
            Action<float, string> onClipProgress = (p, t) => SetProgress(p * 0.48f, string.IsNullOrWhiteSpace(t) ? "CLIP" : t);
            ClipClassificationResult classification;
            clipRunner.ProgressChanged -= onClipProgress;
            clipRunner.ProgressChanged += onClipProgress;
            try
            {
                classification = await clipRunner.ProcessAsync(source, _lifetimeCts.Token);
            }
            finally
            {
                clipRunner.ProgressChanged -= onClipProgress;
            }

            if (!string.IsNullOrWhiteSpace(classification.error))
            {
                ShowToast(classification.error, 3200);
                return;
            }

            var currentPath = CurrentImagePath;
            var preferFileIdentity = source == GetOriginalHistoryTexture() && !string.IsNullOrWhiteSpace(currentPath);
            ClipClassificationCache.Store(clipRunner, classification, source, currentPath, preferFileIdentity);

            SetProgress(0.52f, L("Analyze exposure and color", "分析曝光与色彩"));
            if (!TryAnalyzeAutoTone(source, classification.bestLabel, out var analysis))
            {
                ShowToast(L("Unable to read image statistics.", "无法读取图像统计数据。"), 2800);
                return;
            }

            var appliedSteps = new List<string>(6);
            async UniTask ApplyStepAsync(
                string kernelName,
                Action<ComputeShader> configure,
                string historyLabel,
                string stepLabel)
            {
                SetProgress(0.58f + appliedSteps.Count * 0.07f, stepLabel);
                await ApplyComputeAdjustmentAsync(kernelName, configure, historyLabel);
                appliedSteps.Add(stepLabel);
            }

            if (Mathf.Abs(analysis.temperature) >= 1f || Mathf.Abs(analysis.tint) >= 1f)
            {
                await ApplyStepAsync(
                    "WhiteBalance",
                    cs =>
                    {
                        cs.SetFloat("_WhiteBalanceTemperature", analysis.temperature);
                        cs.SetFloat("_WhiteBalanceTint", analysis.tint);
                    },
                    L("Auto tone white balance", "一键调色 白平衡"),
                    L("White balance", "白平衡"));
            }

            if (Mathf.Abs(analysis.exposure) >= 0.07f)
            {
                await ApplyStepAsync(
                    "AdjustExposure",
                    cs => cs.SetFloat("_Exposure", analysis.exposure),
                    L("Auto tone exposure", "一键调色 曝光"),
                    L("Exposure", "曝光"));
            }

            if (analysis.shadows >= 0.03f)
            {
                await ApplyStepAsync(
                    "AdjustShadows",
                    cs => cs.SetFloat("_Shadows", analysis.shadows),
                    L("Auto tone shadows", "一键调色 阴影"),
                    L("Shadows", "阴影"));
            }

            if (analysis.highlights >= 0.03f)
            {
                await ApplyStepAsync(
                    "AdjustHighlights",
                    cs => cs.SetFloat("_Highlights", analysis.highlights),
                    L("Auto tone highlights", "一键调色 高光"),
                    L("Highlights", "高光"));
            }

            if (analysis.contrast >= 0.03f)
            {
                await ApplyStepAsync(
                    "AdjustContrast",
                    cs => cs.SetFloat("_Contrast", analysis.contrast),
                    L("Auto tone contrast", "一键调色 对比度"),
                    L("Contrast", "对比度"));
            }

            if (analysis.vibrance >= 0.03f)
            {
                await ApplyStepAsync(
                    "AdjustVibrance",
                    cs => cs.SetFloat("_Vibrance", analysis.vibrance),
                    L("Auto tone vibrance", "一键调色 自然饱和度"),
                    L("Vibrance", "自然饱和度"));
            }

            var kind = string.IsNullOrWhiteSpace(classification.bestLabel)
                ? L("image", "图像")
                : classification.bestLabel;
            var summary = appliedSteps.Count == 0
                ? L("No corrective tone adjustment was needed.", "未检测到需要校正的调色问题。")
                : L("Auto tone applied to ", "已对") + kind + L(": ", "应用一键调色：") + string.Join(L(", ", "、"), appliedSteps);
            ShowToast(summary, 3600);
        }
        finally
        {
            _adjustRunning = false;
            HideProgress();
        }
    }

    private static bool TryAnalyzeAutoTone(Texture2D source, string clipLabel, out AutoToneAnalysis analysis)
    {
        analysis = default;
        if (source == null)
            return false;

        try
        {
            var useRawPixels = source.format == TextureFormat.RGBA32;
            var rawPixels = useRawPixels ? source.GetRawTextureData<Color32>() : default;
            var copiedPixels = useRawPixels ? null : source.GetPixels32();
            var length = useRawPixels ? rawPixels.Length : copiedPixels?.Length ?? 0;
            if (length == 0)
                return false;

            var stride = Mathf.Max(1, length / 65536);
            var sumLuminance = 0f;
            var sumSquares = 0f;
            var sumRed = 0f;
            var sumGreen = 0f;
            var sumBlue = 0f;
            var sumSaturation = 0f;
            var darkPixels = 0;
            var brightPixels = 0;
            var clippedPixels = 0;
            var count = 0;
            for (var i = 0; i < length; i += stride)
            {
                var pixel = useRawPixels ? rawPixels[i] : copiedPixels[i];
                var red = pixel.r / 255f;
                var green = pixel.g / 255f;
                var blue = pixel.b / 255f;
                var luminance = red * 0.2126f + green * 0.7152f + blue * 0.0722f;
                var maximum = Mathf.Max(red, Mathf.Max(green, blue));
                var minimum = Mathf.Min(red, Mathf.Min(green, blue));
                sumLuminance += luminance;
                sumSquares += luminance * luminance;
                sumRed += red;
                sumGreen += green;
                sumBlue += blue;
                sumSaturation += maximum - minimum;
                darkPixels += luminance <= 0.12f ? 1 : 0;
                brightPixels += luminance >= 0.88f ? 1 : 0;
                clippedPixels += maximum >= 0.98f ? 1 : 0;
                count++;
            }

            if (count == 0)
                return false;

            var mean = sumLuminance / count;
            var standardDeviation = Mathf.Sqrt(Mathf.Max(0f, sumSquares / count - mean * mean));
            var darkFraction = darkPixels / (float)count;
            var brightFraction = brightPixels / (float)count;
            var clippedFraction = clippedPixels / (float)count;
            var meanRed = sumRed / count;
            var meanGreen = sumGreen / count;
            var meanBlue = sumBlue / count;
            var meanSaturation = sumSaturation / count;
            var profile = ResolveAutoToneProfile(clipLabel);
            var targetLuminance = 0.45f;
            var maximumExposure = 0.48f;
            var shadowStart = 0.18f;
            var shadowScale = 0.46f;
            var shadowMaximum = 0.20f;
            var highlightStart = 0.12f;
            var highlightScale = 0.48f;
            var highlightMaximum = 0.20f;
            var contrastDeviationTarget = 0.18f;
            var contrastScale = 0.85f;
            var contrastMaximum = 0.15f;
            var targetSaturation = 0.25f;
            var vibranceScale = 0.52f;
            var vibranceMaximum = 0.14f;
            var whiteBalanceScale = 1f;
            switch (profile)
            {
                case AutoToneProfile.Portrait:
                    targetLuminance = 0.46f;
                    maximumExposure = 0.36f;
                    shadowScale = 0.34f;
                    shadowMaximum = 0.13f;
                    highlightScale = 0.34f;
                    highlightMaximum = 0.13f;
                    contrastDeviationTarget = 0.17f;
                    contrastScale = 0.55f;
                    contrastMaximum = 0.08f;
                    targetSaturation = 0.21f;
                    vibranceMaximum = 0.06f;
                    whiteBalanceScale = 0.45f;
                    break;
                case AutoToneProfile.Food:
                    targetLuminance = 0.50f;
                    maximumExposure = 0.40f;
                    shadowStart = 0.20f;
                    shadowScale = 0.32f;
                    shadowMaximum = 0.14f;
                    highlightStart = 0.10f;
                    highlightScale = 0.42f;
                    highlightMaximum = 0.18f;
                    contrastDeviationTarget = 0.20f;
                    contrastScale = 0.75f;
                    contrastMaximum = 0.12f;
                    targetSaturation = 0.32f;
                    vibranceMaximum = 0.15f;
                    whiteBalanceScale = 0.65f;
                    break;
                case AutoToneProfile.Landscape:
                    targetLuminance = 0.46f;
                    maximumExposure = 0.45f;
                    shadowStart = 0.17f;
                    shadowScale = 0.46f;
                    shadowMaximum = 0.20f;
                    highlightStart = 0.10f;
                    highlightScale = 0.54f;
                    highlightMaximum = 0.22f;
                    contrastDeviationTarget = 0.21f;
                    contrastScale = 0.90f;
                    contrastMaximum = 0.16f;
                    targetSaturation = 0.29f;
                    vibranceMaximum = 0.16f;
                    whiteBalanceScale = 0.80f;
                    break;
                case AutoToneProfile.Night:
                    targetLuminance = 0.35f;
                    maximumExposure = 0.45f;
                    shadowStart = 0.50f;
                    shadowScale = 0.28f;
                    shadowMaximum = 0.10f;
                    highlightStart = 0.18f;
                    highlightScale = 0.25f;
                    highlightMaximum = 0.10f;
                    contrastDeviationTarget = 0.16f;
                    contrastScale = 0.45f;
                    contrastMaximum = 0.06f;
                    targetSaturation = 0.19f;
                    vibranceMaximum = 0.06f;
                    whiteBalanceScale = 0.40f;
                    break;
                case AutoToneProfile.Architecture:
                    targetLuminance = 0.48f;
                    maximumExposure = 0.42f;
                    shadowScale = 0.35f;
                    shadowMaximum = 0.16f;
                    highlightStart = 0.10f;
                    highlightScale = 0.55f;
                    highlightMaximum = 0.22f;
                    contrastDeviationTarget = 0.21f;
                    contrastScale = 0.75f;
                    contrastMaximum = 0.14f;
                    targetSaturation = 0.18f;
                    vibranceMaximum = 0.07f;
                    whiteBalanceScale = 0.75f;
                    break;
                case AutoToneProfile.Document:
                    targetLuminance = 0.60f;
                    maximumExposure = 0.45f;
                    shadowStart = 0.35f;
                    shadowScale = 0.35f;
                    shadowMaximum = 0.15f;
                    highlightStart = 0.09f;
                    highlightScale = 0.60f;
                    highlightMaximum = 0.28f;
                    contrastDeviationTarget = 0.15f;
                    contrastScale = 0.60f;
                    contrastMaximum = 0.08f;
                    targetSaturation = 0f;
                    vibranceScale = 0f;
                    vibranceMaximum = 0f;
                    whiteBalanceScale = 1.15f;
                    break;
            }

            analysis.exposure = Mathf.Clamp(
                Mathf.Log(targetLuminance / Mathf.Max(0.04f, mean), 2f),
                -maximumExposure,
                maximumExposure);
            if (brightFraction >= 0.18f || clippedFraction >= 0.06f)
                analysis.exposure = Mathf.Min(analysis.exposure, -0.10f);
            else if (darkFraction >= 0.48f && mean <= 0.30f)
                analysis.exposure = Mathf.Max(analysis.exposure, 0.12f);

            analysis.shadows = Mathf.Clamp((darkFraction - shadowStart) * shadowScale, 0f, shadowMaximum);
            analysis.highlights = Mathf.Clamp(
                (brightFraction - highlightStart) * highlightScale + clippedFraction * 0.26f,
                0f,
                highlightMaximum);
            analysis.contrast = Mathf.Clamp(
                (contrastDeviationTarget - standardDeviation) * contrastScale,
                0f,
                contrastMaximum);
            analysis.vibrance = Mathf.Clamp(
                (targetSaturation - meanSaturation) * vibranceScale,
                0f,
                vibranceMaximum);

            var chromaSpread = Mathf.Max(meanRed, Mathf.Max(meanGreen, meanBlue))
                               - Mathf.Min(meanRed, Mathf.Min(meanGreen, meanBlue));
            if (chromaSpread >= 0.10f)
            {
                analysis.temperature = Mathf.Clamp((meanBlue - meanRed) * 52f * whiteBalanceScale, -12f, 12f);
                analysis.tint = Mathf.Clamp(((meanRed + meanBlue) * 0.5f - meanGreen) * 42f * whiteBalanceScale, -9f, 9f);
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private static AutoToneProfile ResolveAutoToneProfile(string clipLabel)
    {
        if (string.Equals(clipLabel, "Portrait", StringComparison.OrdinalIgnoreCase)
            || string.Equals(clipLabel, "Group", StringComparison.OrdinalIgnoreCase)
            || string.Equals(clipLabel, "Pet", StringComparison.OrdinalIgnoreCase))
        {
            return AutoToneProfile.Portrait;
        }

        if (string.Equals(clipLabel, "Food", StringComparison.OrdinalIgnoreCase))
            return AutoToneProfile.Food;
        if (string.Equals(clipLabel, "Landscape", StringComparison.OrdinalIgnoreCase))
            return AutoToneProfile.Landscape;
        if (string.Equals(clipLabel, "Night", StringComparison.OrdinalIgnoreCase))
            return AutoToneProfile.Night;
        if (string.Equals(clipLabel, "Architecture", StringComparison.OrdinalIgnoreCase))
            return AutoToneProfile.Architecture;
        if (string.Equals(clipLabel, "Document", StringComparison.OrdinalIgnoreCase))
            return AutoToneProfile.Document;
        return AutoToneProfile.General;
    }

    private enum PeopleRemovalInpaintBackend
    {
        DeepFillV2Onnx,
        DeepFillV2Ncnn,
        Sd15
    }

    private static PeopleRemovalInpaintBackend ResolvePeopleRemovalInpaintBackend()
    {
        var value = Environment.GetEnvironmentVariable(InpaintBackendEnvVar);
        if (string.IsNullOrWhiteSpace(value))
            return PeopleRemovalInpaintBackend.DeepFillV2Ncnn;
        value = value.Trim();
        if (string.Equals(value, "sd", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "sd15", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "sdinpainting1.5", StringComparison.OrdinalIgnoreCase))
            return PeopleRemovalInpaintBackend.Sd15;
        if (string.Equals(value, "deepfillv2_ncnn", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "ncnn", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "bin", StringComparison.OrdinalIgnoreCase))
            return PeopleRemovalInpaintBackend.DeepFillV2Ncnn;
        return PeopleRemovalInpaintBackend.DeepFillV2Onnx;
    }

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

        var esrganModelName = Host.RealEsrganReproRunner.modelName;
        var esrganGroup = (string.IsNullOrWhiteSpace(esrganModelName)
            || string.Equals(esrganModelName, "realesr-animevideov3-x4", StringComparison.OrdinalIgnoreCase))
            ? AIImageModelGroupId.RealEsrganX4PlusAnime
            : AIImageModelGroupId.RealEsrganOptionalModels;
        if (!await Host.EnsureModelGroupsAvailableAsync(
                "Real-ESRGAN model download",
                _lifetimeCts.Token,
                esrganGroup))
            return;

        _adjustRunning = true;
        ShowProgress("ESRGAN");
        Action<float, string> onProgress = (p, t) => SetProgress(p, t);
        try
        {
            Host.RealEsrganReproRunner.ProgressChanged -= onProgress;
            Host.RealEsrganReproRunner.ProgressChanged += onProgress;
            var result = await Host.RealEsrganReproRunner.ProcessAsync(src, _lifetimeCts.Token);
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
            Host.RealEsrganReproRunner.ProgressChanged -= onProgress;
            Host.RealEsrganReproRunner.ReleaseRuntimeResources();
            _adjustRunning = false;
            HideProgress();
        }
    }

    private async UniTaskVoid ApplyOneClickFaceRepairAsync()
    {
        var codeFormerRunner = Host?.CodeFormerReproRunner;
        var esrganRunner = Host?.RealEsrganReproRunner;
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || codeFormerRunner == null || esrganRunner == null)
            return;

        var source = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (source == null)
            return;

        var esrganGroup = (string.IsNullOrWhiteSpace(esrganRunner.modelName)
            || string.Equals(esrganRunner.modelName, "realesr-animevideov3-x4", StringComparison.OrdinalIgnoreCase))
            ? AIImageModelGroupId.RealEsrganX4PlusAnime
            : AIImageModelGroupId.RealEsrganOptionalModels;
        if (!await Host.EnsureModelGroupsAvailableAsync(
                "Face repair model download",
                _lifetimeCts.Token,
                AIImageModelGroupId.CodeFormerDefault,
                esrganGroup))
            return;

        _adjustRunning = true;
        ShowProgress("一键修复人脸");
        Texture2D codeFormerTexture = null;
        Texture2D finalTexture = null;
        Action<float, string> onCodeFormerProgress = (p, t) =>
            SetProgress(p * 0.55f, string.IsNullOrWhiteSpace(t) ? "CodeFormer" : t);
        Action<float, string> onEsrganProgress = (p, t) =>
            SetProgress(0.55f + p * 0.45f, string.IsNullOrWhiteSpace(t) ? "ESRGAN" : t);
        try
        {
            codeFormerRunner.ProgressChanged -= onCodeFormerProgress;
            codeFormerRunner.ProgressChanged += onCodeFormerProgress;
            var codeFormerResult = await codeFormerRunner.ProcessAsync(source, _lifetimeCts.Token);
            if (!string.IsNullOrWhiteSpace(codeFormerResult.error))
            {
                ShowToast(codeFormerResult.error, 3400);
                return;
            }

            codeFormerTexture = codeFormerResult.texture;
            if (codeFormerTexture == null)
            {
                ShowToast("CodeFormer 未返回修复结果", 3000);
                return;
            }

            codeFormerRunner.ProgressChanged -= onCodeFormerProgress;
            codeFormerRunner.ReleaseRuntimeResources();
            esrganRunner.ProgressChanged -= onEsrganProgress;
            esrganRunner.ProgressChanged += onEsrganProgress;
            var esrganResult = await esrganRunner.ProcessAsync(codeFormerTexture, _lifetimeCts.Token);
            if (!string.IsNullOrWhiteSpace(esrganResult.error))
            {
                ShowToast(esrganResult.error, 3400);
                return;
            }

            finalTexture = esrganResult.texture;
            if (finalTexture != null)
                AddHistory(finalTexture, "一键修复人脸");
            else
                ShowToast("ESRGAN 未返回清晰化结果", 3000);
        }
        finally
        {
            codeFormerRunner.ProgressChanged -= onCodeFormerProgress;
            codeFormerRunner.ReleaseRuntimeResources();
            esrganRunner.ProgressChanged -= onEsrganProgress;
            esrganRunner.ReleaseRuntimeResources();
            if (codeFormerTexture != null
                && !ReferenceEquals(codeFormerTexture, source)
                && !ReferenceEquals(codeFormerTexture, finalTexture))
            {
                Destroy(codeFormerTexture);
            }
            _adjustRunning = false;
            HideProgress();
        }
    }

    private async UniTaskVoid ApplyOneClickRemovePassersAsync()
    {
        var yoloRunner = Host?.YoloSegRunner;
        var deepFillRunner = Host?.DeepFillV2Runner;
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || yoloRunner == null || deepFillRunner == null)
            return;

        var source = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (source == null)
            return;

        var yoloGroup = yoloRunner.modelVariant == YoloSegNcnnReproRunner.YoloSegModelVariant.Yolo11nSeg
            ? AIImageModelGroupId.Yolo11PersonSegmentation
            : AIImageModelGroupId.YoloV8PersonSegmentation;
        var deepFillGroup = await ResolveDeepFillV2ModelGroupAsync(
            deepFillRunner,
            PeopleRemovalInpaintBackend.DeepFillV2Ncnn);
        if (!await Host.EnsureModelGroupsAvailableAsync(
                "Passer-by removal model download",
                _lifetimeCts.Token,
                yoloGroup,
                deepFillGroup))
            return;

        _adjustRunning = true;
        ShowProgress("一键去路人");
        var yoloResult = default(YoloSegResult);
        Texture2D passerbyMask = null;
        Action<float, string> onYoloProgress = (p, t) =>
            SetProgress(p * 0.38f, string.IsNullOrWhiteSpace(t) ? "YOLO" : t);
        Action<float, string> onInpaintProgress = (p, t) =>
            SetProgress(0.38f + p * 0.62f, string.IsNullOrWhiteSpace(t) ? "DeepFillV2" : t);
        try
        {
            var previousTargetPersonOnly = yoloRunner.targetPersonOnly;
            yoloRunner.disallowBufferAccess = true;
            yoloRunner.disallowBufferOutputs = true;
            yoloRunner.disallowBufferToTextureMaterialization = true;
            yoloRunner.targetPersonOnly = true;
            try
            {
                yoloRunner.ProgressChanged -= onYoloProgress;
                yoloRunner.ProgressChanged += onYoloProgress;
                yoloResult = await yoloRunner.ProcessAsync(source, _lifetimeCts.Token);
            }
            finally
            {
                yoloRunner.ProgressChanged -= onYoloProgress;
                yoloRunner.targetPersonOnly = previousTargetPersonOnly;
            }

            if (!string.IsNullOrWhiteSpace(yoloResult.error))
            {
                ShowToast(yoloResult.error, 3400);
                return;
            }

            if (!TryBuildPasserbyMask(yoloResult, out passerbyMask, out var passerbyCount))
            {
                ShowToast("未找到可去除的非主体路人", 2800);
                return;
            }

            DisposeYoloSegOutputTextures(ref yoloResult);
            yoloRunner.ReleaseRuntimeResources();
            await ReleaseGpuPressureBeforeInpaintAsync(_lifetimeCts.Token);

            // This one-click workflow never switches to SD inpainting. DeepFillV2
            // will use its NCNN payload first and its existing ONNX fallback only
            // when that is the representation installed for the current sample.
            deepFillRunner.backend = DeepFillV2Backend.NcnnBin;
            deepFillRunner.enableDebugDump = false;
            deepFillRunner.precisionMode = AexisPrecisionMode.Auto;
            deepFillRunner.useArgbFloatTensor = false;
            deepFillRunner.enableGeneralTextureConvolution = true;
            deepFillRunner.enableDepthWiseTextureConvolution = true;
            deepFillRunner.enableConv1x1TextureConvolution = true;
            deepFillRunner.ProgressChanged -= onInpaintProgress;
            deepFillRunner.ProgressChanged += onInpaintProgress;
            var deepFillResult = await deepFillRunner.ProcessAsync(source, passerbyMask, _lifetimeCts.Token);
            if (!string.IsNullOrWhiteSpace(deepFillResult.error))
            {
                ShowToast(deepFillResult.error, 3600);
                return;
            }

            if (deepFillResult.texture != null)
                AddHistory(deepFillResult.texture, $"一键去路人（{passerbyCount} 人）");
            else
                ShowToast("DeepFillV2 未返回修复结果", 3000);
        }
        finally
        {
            yoloRunner.ProgressChanged -= onYoloProgress;
            yoloRunner.ReleaseRuntimeResources();
            deepFillRunner.ProgressChanged -= onInpaintProgress;
            deepFillRunner.Release();
            DisposeYoloSegOutputTextures(ref yoloResult);
            if (passerbyMask != null)
                Destroy(passerbyMask);
            _adjustRunning = false;
            HideProgress();
        }
    }

    private static bool TryBuildPasserbyMask(
        YoloSegResult result,
        out Texture2D passerbyMask,
        out int passerbyCount)
    {
        passerbyMask = null;
        passerbyCount = 0;
        var detections = result.detections;
        if (result.mask == null || detections == null || detections.Length < 2)
            return false;

        var width = result.mask.width;
        var height = result.mask.height;
        if (width <= 0 || height <= 0)
            return false;

        var mainSubjectIndex = SelectMainSubjectIndex(detections, width, height);
        if (mainSubjectIndex < 0)
            return false;

        var foregroundSubjects = BuildForegroundSubjectGroup(detections, mainSubjectIndex, width, height);

        var sourcePixels = result.mask.GetPixels32();
        if (sourcePixels == null || sourcePixels.Length != width * height)
            return false;

        var protectedPrimaryRects = new Rect[detections.Length];
        var passerRects = new Rect[detections.Length];
        for (var i = 0; i < detections.Length; i++)
        {
            var textureRect = GetTextureSpaceDetectionRect(detections[i].rect, width, height);
            if (foregroundSubjects[i])
            {
                protectedPrimaryRects[i] = ExpandRect(
                    textureRect,
                    Mathf.Max(8f, Mathf.Min(width, height) * 0.025f),
                    width,
                    height);
            }
            else
            {
                passerRects[i] = textureRect;
            }
        }

        var outputPixels = new Color32[sourcePixels.Length];
        var selectedPassers = new bool[detections.Length];
        var selectedPixelCount = 0;
        var instanceLabels = result.instanceMaskLabels;
        var hasInstanceLabels = instanceLabels != null && instanceLabels.Length == sourcePixels.Length;
        for (var y = 0; y < height; y++)
        {
            var rowOffset = y * width;
            for (var x = 0; x < width; x++)
            {
                var index = rowOffset + x;
                var personPixel = sourcePixels[index];
                if (Mathf.Max(personPixel.r, Mathf.Max(personPixel.g, personPixel.b)) == 0)
                    continue;

                var point = new Vector2(x + 0.5f, y + 0.5f);
                var instanceIndex = hasInstanceLabels ? instanceLabels[index] - 1 : -1;
                if (instanceIndex >= 0 && instanceIndex < detections.Length)
                {
                    if (foregroundSubjects[instanceIndex])
                        continue;

                    outputPixels[index] = new Color32(255, 255, 255, 255);
                    selectedPassers[instanceIndex] = true;
                    selectedPixelCount++;
                    continue;
                }

                var insidePrimaryProtection = false;
                for (var primaryIndex = 0; primaryIndex < protectedPrimaryRects.Length; primaryIndex++)
                {
                    if (foregroundSubjects[primaryIndex] && protectedPrimaryRects[primaryIndex].Contains(point))
                    {
                        insidePrimaryProtection = true;
                        break;
                    }
                }
                if (insidePrimaryProtection)
                    continue;

                for (var detectionIndex = 0; detectionIndex < passerRects.Length; detectionIndex++)
                {
                    if (foregroundSubjects[detectionIndex] || !passerRects[detectionIndex].Contains(point))
                        continue;

                    outputPixels[index] = new Color32(255, 255, 255, 255);
                    selectedPassers[detectionIndex] = true;
                    selectedPixelCount++;
                    break;
                }
            }
        }

        if (selectedPixelCount == 0)
            return false;

        for (var i = 0; i < selectedPassers.Length; i++)
        {
            if (selectedPassers[i])
                passerbyCount++;
        }

        if (passerbyCount == 0)
            return false;

        passerbyMask = new Texture2D(width, height, TextureFormat.RGBA32, false, true)
        {
            name = "YOLO_Passerby_Mask",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        passerbyMask.SetPixels32(outputPixels);
        passerbyMask.Apply(false, false);
        return true;
    }

    private static bool[] BuildForegroundSubjectGroup(
        IReadOnlyList<YoloSegDetection> detections,
        int mainSubjectIndex,
        int width,
        int height)
    {
        var foregroundSubjects = new bool[detections.Count];
        var dominantArea = GetDetectionArea(detections[mainSubjectIndex]);
        var dominantHeight = Mathf.Max(1f, detections[mainSubjectIndex].rect.height);
        var hasClearlyForegroundPerson = false;
        for (var i = 0; i < detections.Count; i++)
        {
            if (!IsClearlyForegroundPerson(detections[i], dominantArea, dominantHeight, width, height))
                continue;

            foregroundSubjects[i] = true;
            hasClearlyForegroundPerson = true;
        }

        // A sparse or unusually framed photo can have no box that meets the global
        // foreground thresholds. Keep the best candidate rather than removing it.
        if (!hasClearlyForegroundPerson)
            foregroundSubjects[mainSubjectIndex] = true;

        var changed = true;
        while (changed)
        {
            changed = false;
            for (var candidateIndex = 0; candidateIndex < detections.Count; candidateIndex++)
            {
                if (foregroundSubjects[candidateIndex])
                    continue;

                for (var anchorIndex = 0; anchorIndex < detections.Count; anchorIndex++)
                {
                    if (!foregroundSubjects[anchorIndex]
                        || !IsForegroundCompanion(
                            detections[candidateIndex],
                            detections[anchorIndex],
                            dominantArea,
                            dominantHeight,
                            width,
                            height))
                    {
                        continue;
                    }

                    foregroundSubjects[candidateIndex] = true;
                    changed = true;
                    break;
                }
            }
        }

        return foregroundSubjects;
    }

    private static bool IsClearlyForegroundPerson(
        YoloSegDetection candidate,
        float dominantArea,
        float dominantHeight,
        int width,
        int height)
    {
        var candidateArea = GetDetectionArea(candidate);
        var imageArea = Mathf.Max(1f, width * (float)height);
        return candidateArea >= Mathf.Max(dominantArea * 0.07f, imageArea * 0.003f)
               && candidate.rect.height >= Mathf.Max(dominantHeight * 0.28f, height * 0.20f)
               && candidate.rect.yMax >= height * 0.58f;
    }

    private static bool IsForegroundCompanion(
        YoloSegDetection candidate,
        YoloSegDetection anchor,
        float dominantArea,
        float dominantHeight,
        int width,
        int height)
    {
        var candidateArea = GetDetectionArea(candidate);
        if (candidateArea < Mathf.Max(dominantArea * 0.05f, width * (float)height * 0.002f)
            || candidate.rect.height < Mathf.Max(dominantHeight * 0.24f, height * 0.17f)
            || candidate.rect.yMax < Mathf.Max(height * 0.52f, anchor.rect.yMax - height * 0.18f))
        {
            return false;
        }

        var verticalOverlap = Mathf.Max(0f, Mathf.Min(candidate.rect.yMax, anchor.rect.yMax)
                                            - Mathf.Max(candidate.rect.yMin, anchor.rect.yMin));
        verticalOverlap /= Mathf.Max(1f, Mathf.Min(candidate.rect.height, anchor.rect.height));
        var horizontalGap = Mathf.Max(0f, Mathf.Max(candidate.rect.xMin - anchor.rect.xMax, anchor.rect.xMin - candidate.rect.xMax));
        var overlap = GetRectIntersectionArea(candidate.rect, anchor.rect);
        return overlap > 0f
               || (verticalOverlap >= 0.35f && horizontalGap <= Mathf.Max(width * 0.10f, Mathf.Min(candidate.rect.width, anchor.rect.width) * 0.45f));
    }

    private static int SelectMainSubjectIndex(IReadOnlyList<YoloSegDetection> detections, int width, int height)
    {
        if (detections == null || detections.Count == 0)
            return -1;

        var maxArea = 1f;
        for (var i = 0; i < detections.Count; i++)
            maxArea = Mathf.Max(maxArea, GetDetectionArea(detections[i]));

        var mainSubjectIndex = -1;
        var bestScore = float.NegativeInfinity;
        for (var i = 0; i < detections.Count; i++)
        {
            var detection = detections[i];
            var area = GetDetectionArea(detection);
            var centerX01 = Mathf.Clamp01(detection.rect.center.x / Mathf.Max(1f, width));
            var centrality = 1f - Mathf.Clamp01(Mathf.Abs(centerX01 - 0.5f) * 2f);
            // YOLO image coordinates start at the top. A lower box together with a
            // larger box is a practical foreground cue, not a depth-model result.
            var foregroundPlacement = Mathf.Clamp01(detection.rect.yMax / Mathf.Max(1f, height));
            var score = area / maxArea * 0.55f
                        + centrality * 0.20f
                        + Mathf.Clamp01(detection.probability) * 0.15f
                        + foregroundPlacement * 0.10f;

            for (var otherIndex = 0; otherIndex < detections.Count; otherIndex++)
            {
                if (otherIndex == i)
                    continue;

                var other = detections[otherIndex];
                var overlap = GetRectIntersectionArea(detection.rect, other.rect)
                              / Mathf.Max(1f, Mathf.Min(detection.rect.width * detection.rect.height, other.rect.width * other.rect.height));
                if (overlap < 0.15f)
                    continue;

                var otherArea = GetDetectionArea(other);
                var isLarger = area > otherArea * 1.08f;
                var isLower = detection.rect.yMax > other.rect.yMax + height * 0.025f;
                var otherIsLarger = otherArea > area * 1.08f;
                var otherIsLower = other.rect.yMax > detection.rect.yMax + height * 0.025f;
                if ((isLarger || isLower) && !(otherIsLarger || otherIsLower))
                    score += 0.12f;
                else if ((otherIsLarger || otherIsLower) && !(isLarger || isLower))
                    score -= 0.12f;
            }

            if (score > bestScore)
            {
                bestScore = score;
                mainSubjectIndex = i;
            }
        }

        return mainSubjectIndex;
    }

    private static float GetDetectionArea(YoloSegDetection detection)
    {
        var rectArea = Mathf.Max(1f, detection.rect.width * detection.rect.height);
        return Mathf.Max(detection.maskPixelCount, rectArea * 0.25f);
    }

    private static float GetRectIntersectionArea(Rect a, Rect b)
    {
        var minX = Mathf.Max(a.xMin, b.xMin);
        var minY = Mathf.Max(a.yMin, b.yMin);
        var maxX = Mathf.Min(a.xMax, b.xMax);
        var maxY = Mathf.Min(a.yMax, b.yMax);
        return maxX > minX && maxY > minY ? (maxX - minX) * (maxY - minY) : 0f;
    }

    private static Rect GetTextureSpaceDetectionRect(Rect detectionRect, int width, int height)
    {
        return Rect.MinMaxRect(
            Mathf.Clamp(detectionRect.xMin, 0f, width),
            Mathf.Clamp(height - detectionRect.yMax, 0f, height),
            Mathf.Clamp(detectionRect.xMax, 0f, width),
            Mathf.Clamp(height - detectionRect.yMin, 0f, height));
    }

    private static Rect ExpandRect(Rect rect, float padding, int width, int height)
    {
        return Rect.MinMaxRect(
            Mathf.Clamp(rect.xMin - padding, 0f, width),
            Mathf.Clamp(rect.yMin - padding, 0f, height),
            Mathf.Clamp(rect.xMax + padding, 0f, width),
            Mathf.Clamp(rect.yMax + padding, 0f, height));
    }

    private static void DisposeYoloSegOutputTextures(ref YoloSegResult result)
    {
        var texture = result.texture;
        var mask = result.mask;
        var overlay = result.overlay;
        if (texture != null)
            UnityEngine.Object.Destroy(texture);
        if (mask != null && !ReferenceEquals(mask, texture))
            UnityEngine.Object.Destroy(mask);
        if (overlay != null && !ReferenceEquals(overlay, texture) && !ReferenceEquals(overlay, mask))
            UnityEngine.Object.Destroy(overlay);
        result = default;
    }

    private async UniTaskVoid ApplyOneClickOptimizeBackgroundAsync()
    {
        var mattingRunner = Host?.MattingReproRunner;
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || mattingRunner == null)
            return;

        var source = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (source == null)
            return;

        if (!await Host.EnsureModelGroupsAvailableAsync(
                "Matting model download",
                _lifetimeCts.Token,
                AIImageModelGroupId.Matting))
            return;

        _adjustRunning = true;
        ShowProgress("一键优化背景");
        var mattingResult = default(MattingResult);
        Action<float, string> onMattingProgress = (p, t) => SetProgress(p, string.IsNullOrWhiteSpace(t) ? "Matting" : t);
        try
        {
            mattingRunner.ProgressChanged -= onMattingProgress;
            mattingRunner.ProgressChanged += onMattingProgress;
            mattingResult = await mattingRunner.ProcessAsync(source, _lifetimeCts.Token, false);
            if (!string.IsNullOrWhiteSpace(mattingResult.error))
            {
                ShowToast(mattingResult.error, 3400);
                return;
            }

            if (mattingResult.alpha == null || mattingResult.alpha.Length != source.width * source.height)
            {
                ShowToast("Matting 未返回有效 alpha", 3000);
                return;
            }

            mattingRunner.ProgressChanged -= onMattingProgress;
            mattingRunner.ReleaseRuntimeResources();
            var composited = await ComposeNaturalSkyBackgroundAsync(source, mattingResult.alpha, _lifetimeCts.Token);
            if (composited == null)
            {
                ShowToast("无法生成天空优化结果", 3000);
                return;
            }

            AddHistory(composited, "一键优化背景");
        }
        finally
        {
            mattingRunner.ProgressChanged -= onMattingProgress;
            mattingRunner.ReleaseRuntimeResources();
            if (mattingResult.texture != null)
                Destroy(mattingResult.texture);
            if (mattingResult.matte != null)
                Destroy(mattingResult.matte);
            mattingResult.alpha = null;
            _adjustRunning = false;
            HideProgress();
        }
    }

    private static async UniTask<Texture2D> ComposeNaturalSkyBackgroundAsync(
        Texture2D source,
        float[] foregroundAlpha,
        CancellationToken ct)
    {
        if (source == null || foregroundAlpha == null || foregroundAlpha.Length != source.width * source.height)
            return null;

        var useRawSourcePixels = source.format == TextureFormat.RGBA32;
        var rawSourcePixels = useRawSourcePixels
            ? source.GetRawTextureData<Color32>()
            : default;
        var copiedSourcePixels = useRawSourcePixels ? null : source.GetPixels32();
        var sourcePixelCount = useRawSourcePixels ? rawSourcePixels.Length : copiedSourcePixels?.Length ?? 0;
        if (sourcePixelCount != foregroundAlpha.Length)
            return null;

        var width = source.width;
        var height = source.height;
        Texture2D output = null;
        try
        {
            output = new Texture2D(width, height, TextureFormat.RGBA32, false, false)
            {
                name = "Matting_NaturalSky_Composite",
                filterMode = source.filterMode,
                wrapMode = TextureWrapMode.Clamp
            };
            var outputPixels = output.GetRawTextureData<Color32>();
            if (outputPixels.Length != sourcePixelCount)
            {
                Destroy(output);
                return null;
            }

            const int rowsPerFrame = 64;
            for (var y = 0; y < height; y++)
            {
                ct.ThrowIfCancellationRequested();
                var vertical01 = height > 1 ? (float)y / (height - 1) : 0.5f;
                var fromTop01 = 1f - vertical01;
                var upperSkyWeight = 1f - Mathf.SmoothStep(0.18f, 0.82f, fromTop01);
                var rowOffset = y * width;
                for (var x = 0; x < width; x++)
                {
                    var index = rowOffset + x;
                    var original = useRawSourcePixels ? rawSourcePixels[index] : copiedSourcePixels[index];
                    var foreground = Mathf.Clamp01(foregroundAlpha[index]);
                    var r = original.r / 255f;
                    var g = original.g / 255f;
                    var b = original.b / 255f;
                    var maxChannel = Mathf.Max(r, Mathf.Max(g, b));
                    var minChannel = Mathf.Min(r, Mathf.Min(g, b));
                    var graySkyWeight = 1f - Mathf.SmoothStep(0.055f, 0.24f, maxChannel - minChannel);
                    var luminance = r * 0.2126f + g * 0.7152f + b * 0.0722f;
                    var luminanceWeight = Mathf.Lerp(0.35f, 1f, Mathf.SmoothStep(0.06f, 0.85f, luminance));
                    var replacementWeight = Mathf.Clamp01(
                        (1f - foreground) * upperSkyWeight * graySkyWeight * luminanceWeight);

                    var sky = EvaluateNaturalSkyColor(
                        width > 1 ? (float)x / (width - 1) : 0.5f,
                        vertical01);
                    outputPixels[index] = new Color32(
                        (byte)Mathf.RoundToInt(Mathf.Lerp(original.r, sky.r * 255f, replacementWeight)),
                        (byte)Mathf.RoundToInt(Mathf.Lerp(original.g, sky.g * 255f, replacementWeight)),
                        (byte)Mathf.RoundToInt(Mathf.Lerp(original.b, sky.b * 255f, replacementWeight)),
                        255);
                }

                if ((y + 1) % rowsPerFrame == 0)
                    await UniTask.Yield(PlayerLoopTiming.Update, ct);
            }

            output.Apply(false, false);
            return output;
        }
        catch
        {
            if (output != null)
                Destroy(output);
            throw;
        }
    }

    private static Color EvaluateNaturalSkyColor(float horizontal01, float vertical01)
    {
        var skyBlend = Mathf.SmoothStep(0.04f, 0.86f, vertical01);
        var horizon = new Color(0.72f, 0.84f, 0.96f);
        var zenith = new Color(0.16f, 0.48f, 0.88f);
        var sky = Color.Lerp(horizon, zenith, skyBlend);
        var broadCloud = Mathf.PerlinNoise(horizontal01 * 2.15f + 31.7f, vertical01 * 1.65f + 8.4f);
        var cloudDetail = Mathf.PerlinNoise(horizontal01 * 5.6f + 6.2f, vertical01 * 4.1f + 19.8f);
        var cloudField = broadCloud * 0.72f + cloudDetail * 0.28f;
        var cloudWeight = Mathf.SmoothStep(0.59f, 0.76f, cloudField)
                          * Mathf.SmoothStep(0.08f, 0.70f, vertical01);
        return Color.Lerp(sky, new Color(0.98f, 0.99f, 1f), cloudWeight * 0.82f);
    }

    private async UniTaskVoid ApplyYoloAndInpaintingReproAsync()
    {
        if (_aiRunning || _adjustRunning || _lifetimeCts == null || Host?.YoloSegRunner == null)
            return;
        var inpaintBackend = ResolvePeopleRemovalInpaintBackend();
        var useSdInpainting = inpaintBackend == PeopleRemovalInpaintBackend.Sd15;
        if (useSdInpainting && Host.SDInpaintingRunner == null)
            return;
        if (!useSdInpainting && Host.DeepFillV2Runner == null)
            return;
        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null)
            return;

        var requestedGroups = new List<AIImageModelGroupId>
        {
            Host.YoloSegRunner.modelVariant == YoloSegNcnnReproRunner.YoloSegModelVariant.Yolo11nSeg
                ? AIImageModelGroupId.Yolo11PersonSegmentation
                : AIImageModelGroupId.YoloV8PersonSegmentation
        };
        if (useSdInpainting)
            requestedGroups.Add(AIImageModelGroupId.StableDiffusion);
        else
            requestedGroups.Add(await ResolveDeepFillV2ModelGroupAsync(Host.DeepFillV2Runner, inpaintBackend));
        if (!await Host.EnsureModelGroupsAvailableAsync(
                "People removal model download",
                _lifetimeCts.Token,
                requestedGroups.ToArray()))
            return;

        _adjustRunning = true;
        ShowProgress("YOLO + Inpainting");
        try
        {
            AexisGpuResourceTracker.Enabled = true;
            AexisGpuResourceTracker.Reset("MainView2.YoloInpaint");
            YoloInpaintResourceSnapshotLines.Clear();
            LogYoloInpaintResourceSnapshot("begin");
            var oldTargetPersonOnly = Host.YoloSegRunner.targetPersonOnly;
            Host.YoloSegRunner.disallowBufferAccess = true;
            Host.YoloSegRunner.disallowBufferOutputs = true;
            Host.YoloSegRunner.disallowBufferToTextureMaterialization = true;
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

            if (!useSdInpainting)
            {
                var deepFillRunner = Host.DeepFillV2Runner;
                deepFillRunner.backend = inpaintBackend == PeopleRemovalInpaintBackend.DeepFillV2Ncnn
                    ? DeepFillV2Backend.NcnnBin
                    : DeepFillV2Backend.OnnxDirect;
                deepFillRunner.enableDebugDump = false;
                deepFillRunner.precisionMode = AexisPrecisionMode.Auto;
                deepFillRunner.useArgbFloatTensor = false;
                deepFillRunner.enableGeneralTextureConvolution = true;
                deepFillRunner.enableDepthWiseTextureConvolution = true;
                deepFillRunner.enableConv1x1TextureConvolution = true;
#if DEVELOPMENT_BUILD
                deepFillRunner.enableLayerPathDebugLog = true;
#endif
                SetProgress(0.40f, deepFillRunner.backend == DeepFillV2Backend.OnnxDirect ? "DeepFillV2 ONNX" : "DeepFillV2 NCNN");
                await ReleaseGpuPressureBeforeInpaintAsync(_lifetimeCts.Token);
                LogYoloInpaintResourceSnapshot("before_deepfillv2_process");

                deepFillRunner.ProgressChanged -= OnInpaintProgress;
                deepFillRunner.ProgressChanged += OnInpaintProgress;
                DeepFillV2Result deepFillResult;
                try
                {
                    deepFillResult = await deepFillRunner.ProcessAsync(src, result.mask, _lifetimeCts.Token);
                }
                finally
                {
                    deepFillRunner.ProgressChanged -= OnInpaintProgress;
                }
                LogYoloInpaintResourceSnapshot("after_deepfillv2_process");
                if (!string.IsNullOrWhiteSpace(deepFillResult.error))
                {
                    ShowToast(deepFillResult.error, 3600);
                    if (result.overlay != null)
                        AddHistory(result.overlay, $"YOLO 璇嗗埆 {result.personCount}");
                    return;
                }

                if (deepFillResult.texture != null)
                {
                    AddHistory(deepFillResult.texture, $"YOLO DeepFillV2 {result.personCount}");
                    LogYoloInpaintResourceSnapshot("after_add_history");
                }
                return;
            }

            Host.SDInpaintingRunner.useOfficialUnetCache = false;
            Host.SDInpaintingRunner.keepRawConvWeightsForTexturePath = false;
            Host.SDInpaintingRunner.tensorTextureFormat = RenderTextureFormat.ARGBHalf;
            Host.SDInpaintingRunner.encoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
            Host.SDInpaintingRunner.decoderTensorTextureFormat = RenderTextureFormat.ARGBHalf;
            Host.SDInpaintingRunner.enableAttentionMatMulPack4Specializations = true;
            Host.SDInpaintingRunner.useCommandBuffer = false;
            Host.SDInpaintingRunner.useAsyncComputeCommandBuffer = false;
            Host.SDInpaintingRunner.disallowInferenceTempComputeBuffers = true;
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
            // DeepFillV2 returns a CPU Texture2D, so its Pack4 graph and pooled RTs
            // are no longer needed after this one-shot people-removal operation.
            // Keeping them alive lets a TBDR render frame overlap stale compute work.
            Host?.DeepFillV2Runner?.Release();
            LogYoloInpaintResourceSnapshot("finally");
            TryWriteYoloInpaintResourceReport();
            AexisGpuResourceTracker.Enabled = false;
            _adjustRunning = false;
            HideProgress();
        }
    }

    private static AIImageModelGroupId ResolveQwen35ModelGroup()
    {
        const string mobileModelDirectoryName = "qwen3.5_0.8b_mobile_q4";
        var configured = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            var resolved = Qwen35ModelDirectoryResolver.Resolve(configured, mobileModelDirectoryName);
            var directoryName = Path.GetFileName(resolved.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            if (string.Equals(directoryName, "qwen3.5_0.8b", StringComparison.OrdinalIgnoreCase))
                return AIImageModelGroupId.Qwen35FullPrecision;
            if (string.Equals(directoryName, "qwen3.5_0.8b_mobile_q8", StringComparison.OrdinalIgnoreCase))
                return AIImageModelGroupId.Qwen35MobileQ8;
        }

        return AIImageModelGroupId.Qwen35MobileQ4;
    }

    private static async UniTask<AIImageModelGroupId> ResolveDeepFillV2ModelGroupAsync(
        DeepFillV2Runner runner,
        PeopleRemovalInpaintBackend backend)
    {
        if (runner != null)
        {
            var selectedPath = backend == PeopleRemovalInpaintBackend.DeepFillV2Ncnn
                ? runner.ncnnBinRelativePath
                : runner.sourceOnnxRelativePath;
            if (!string.IsNullOrWhiteSpace(selectedPath)
                && selectedPath.IndexOf("deepfillv2_hifill", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return AIImageModelGroupId.DeepFillV2HiFill;
            }
        }

        var preferred = backend == PeopleRemovalInpaintBackend.DeepFillV2Ncnn
            ? AIImageModelGroupId.DeepFillV2Case1Ncnn
            : AIImageModelGroupId.DeepFillV2Case1Onnx;
        var alternative = preferred == AIImageModelGroupId.DeepFillV2Case1Ncnn
            ? AIImageModelGroupId.DeepFillV2Case1Onnx
            : AIImageModelGroupId.DeepFillV2Case1Ncnn;

        // Match DeepFillV2Runner's representation fallback before showing a
        // download prompt. A build containing either supported representation
        // should run without requesting the trimmed sibling payload.
        if (await AIImageModelDelivery.IsAvailableAsync(AIImageModelDelivery.GetGroup(preferred)))
            return preferred;
        if (await AIImageModelDelivery.IsAvailableAsync(AIImageModelDelivery.GetGroup(alternative)))
            return alternative;
        return preferred;
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
                + " | " + AexisGpuResourceTracker.BuildSummary();
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
            AexisGpuResourceTracker.WriteReport(dir, "gpu_resource_stats.txt");
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

        if (!await Host.EnsureModelGroupsAvailableAsync(
                "Matting model download",
                _lifetimeCts.Token,
                AIImageModelGroupId.Matting))
            return;

        _adjustRunning = true;
        ShowProgress("Matting");
        Action<float, string> onProgress = (p, t) => SetProgress(p, t);
        try
        {
            Host.MattingReproRunner.ProgressChanged -= onProgress;
            Host.MattingReproRunner.ProgressChanged += onProgress;
            var result = await Host.MattingReproRunner.ProcessAsync(src, _lifetimeCts.Token);
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
            Host.MattingReproRunner.ProgressChanged -= onProgress;
            Host.MattingReproRunner.ReleaseRuntimeResources();
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

        if (!await Host.EnsureModelGroupsAvailableAsync(
                "GFPGAN model download",
                _lifetimeCts.Token,
                AIImageModelGroupId.GfpganDefault))
            return;

        _adjustRunning = true;
        ShowProgress("GFPGAN");
        Action<float, string> onProgress = (p, t) => SetProgress(p, t);
        try
        {
            Host.GfpganReproRunner.ProgressChanged -= onProgress;
            Host.GfpganReproRunner.ProgressChanged += onProgress;
            var result = await Host.GfpganReproRunner.ProcessAsync(src, _lifetimeCts.Token);
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
            Host.GfpganReproRunner.ProgressChanged -= onProgress;
            Host.GfpganReproRunner.ReleaseRuntimeResources();
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

        if (!await Host.EnsureModelGroupsAvailableAsync(
                "CodeFormer model download",
                _lifetimeCts.Token,
                AIImageModelGroupId.CodeFormerDefault))
            return;

        _adjustRunning = true;
        ShowProgress("CodeFormer");
        Action<float, string> onProgress = (p, t) => SetProgress(p, t);
        try
        {
            Host.CodeFormerReproRunner.ProgressChanged -= onProgress;
            Host.CodeFormerReproRunner.ProgressChanged += onProgress;
            var result = await Host.CodeFormerReproRunner.ProcessAsync(src, _lifetimeCts.Token);
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
            Host.CodeFormerReproRunner.ProgressChanged -= onProgress;
            Host.CodeFormerReproRunner.ReleaseRuntimeResources();
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
        if (_saveCurrentImageRunning)
            return;

        _saveCurrentImageRunning = true;
        var path = CurrentImagePath;
        Texture2D tex = null;
        try
        {
            if (string.IsNullOrWhiteSpace(path))
                return;

            tex = GetCurrentHistoryTexture();
            if (tex == null)
                return;

            var cancellationToken = _lifetimeCts != null
                ? _lifetimeCts.Token
                : CancellationToken.None;
            var isRaw = RawPhotoParser.IsRawExtension(path);
            string existingOriginalPath = null;
            if (!isRaw)
            {
                try
                {
                    existingOriginalPath = await ResolveExistingOriginalSourcePathAsync(path, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    return;
                }
            }

            var preferredPath = isRaw ? Path.ChangeExtension(path, ".jpg") : path;
            if (!StandardImageIO.TryEncodeTextureWithMetadata(
                    tex,
                    preferredPath,
                    path,
                    95,
                    out var outputBytes,
                    out var encodeError))
            {
                ShowToast(string.IsNullOrWhiteSpace(encodeError) ? "Save failed" : encodeError, 2400);
                return;
            }

            var isUnlinkedNormalImage = !isRaw && string.IsNullOrWhiteSpace(existingOriginalPath);
            OriginalBackupResult backup = null;
            if (isUnlinkedNormalImage && await CanAttemptDirectSaveAsync(preferredPath, cancellationToken))
            {
                cancellationToken.ThrowIfCancellationRequested();
                // Once source preservation starts, let the commit/rollback sequence finish even if the page closes.
                backup = await TryPreserveOriginalAsync(path, CancellationToken.None);
            }

            SaveFileResult saveResult;
            string originalSourcePath;
            if (isUnlinkedNormalImage)
            {
                originalSourcePath = backup != null && backup.success ? backup.path : path;
                if (backup != null && backup.success)
                {
                    var commitToken = backup.moved ? CancellationToken.None : cancellationToken;
                    saveResult = await TryWriteBytesAsync(preferredPath, outputBytes, commitToken);
                    if (!saveResult.success)
                    {
                        saveResult = await TryWriteFallbackAsync(outputBytes, preferredPath, commitToken);
                        if (!saveResult.success && backup.moved)
                            await TryRestoreMovedOriginalAsync(path, backup.path, CancellationToken.None);
                    }
                }
                else
                {
                    // Do not overwrite the only copy when Originals cannot be created or copied.
                    saveResult = await TryWriteFallbackAsync(outputBytes, preferredPath, cancellationToken);
                }
            }
            else
            {
                originalSourcePath = isRaw ? path : existingOriginalPath;
                saveResult = await TrySavePreferredOrFallbackAsync(
                    outputBytes,
                    preferredPath,
                    cancellationToken);
            }

            if (!saveResult.success)
            {
                UnityEngine.Debug.LogWarning("[AIIMAGE-SAVE] File operation failed: " + (saveResult.error ?? "unknown error"));
                ShowToast(
                    L("Save failed. The original image was kept.", "保存失败，原图已保留。"),
                    3000);
                return;
            }

            CompleteSavedImage(saveResult.path, originalSourcePath);
            if (saveResult.usedFallback)
            {
                ShowToast(
                    L("Saved to app storage because the original folder is not writable.", "原目录不可写，已保存到应用存储。"),
                    3600);
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception exception)
        {
            UnityEngine.Debug.LogWarning("[AIIMAGE-SAVE] Save failed: " + exception.Message);
            ShowToast(L("Save failed. The original image was kept.", "保存失败，原图已保留。"), 3000);
        }
        finally
        {
            _saveCurrentImageRunning = false;
        }

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

    private async UniTask<string> ResolveExistingOriginalSourcePathAsync(string imagePath, CancellationToken cancellationToken)
    {
        if (Host?.LibraryPage != null &&
            Host.LibraryPage.TryGetLinkedOriginalSourcePath(imagePath, out var libraryOriginalPath) &&
            File.Exists(libraryOriginalPath))
        {
            return libraryOriginalPath;
        }

        var cachedOriginalPath = await ClipClassificationCache.GetOriginalSourcePathForFileAsync(imagePath, cancellationToken);
        return !string.IsNullOrWhiteSpace(cachedOriginalPath) && File.Exists(cachedOriginalPath)
            ? cachedOriginalPath
            : null;
    }

    private void CompleteSavedImage(string savedPath, string originalPath)
    {
        try
        {
            ClipClassificationCache.StoreOriginalSourcePath(savedPath, originalPath);
            Host?.LibraryPage?.RegisterSavedImageSourceLink(savedPath, originalPath);
            Host?.InvalidateTextureCacheForPath(savedPath);
            if (Host != null && Host.ReloadMainImageFromDisk(savedPath))
                ShowToast(L("Saved and reloaded", "已保存并重新载入"), 1800);
            else
                ShowToast(L("Saved", "已保存"), 1800);
        }
        catch
        {
            ShowToast(L("Save failed", "保存失败"), 2200);
        }
    }

    private async UniTask<SaveFileResult> TrySavePreferredOrFallbackAsync(
        byte[] outputBytes,
        string preferredPath,
        CancellationToken cancellationToken)
    {
        SaveFileResult primary = null;
        if (await CanAttemptDirectSaveAsync(preferredPath, cancellationToken))
            primary = await TryWriteBytesAsync(preferredPath, outputBytes, cancellationToken);

        if (primary != null && primary.success)
            return primary;

        var fallback = await TryWriteFallbackAsync(outputBytes, preferredPath, cancellationToken);
        if (fallback.success)
            return fallback;

        return new SaveFileResult
        {
            success = false,
            error = primary?.error ?? fallback.error
        };
    }

    private async UniTask<OriginalBackupResult> TryPreserveOriginalAsync(
        string imagePath,
        CancellationToken cancellationToken)
    {
        return await UniTask.RunOnThreadPool(() =>
        {
            var result = new OriginalBackupResult();
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
                {
                    result.error = "Image file is missing";
                    return result;
                }

                var directory = Path.GetDirectoryName(imagePath);
                var fileName = Path.GetFileName(imagePath);
                if (string.IsNullOrWhiteSpace(directory) || string.IsNullOrWhiteSpace(fileName))
                {
                    result.error = "Image path is invalid";
                    return result;
                }

                var originalsDirectory = Path.Combine(directory, "Originals");
                if (!Directory.Exists(originalsDirectory))
                    Directory.CreateDirectory(originalsDirectory);

                var originalPath = GetAvailableOriginalPath(originalsDirectory, fileName);
                try
                {
                    File.Move(imagePath, originalPath);
                    result.success = true;
                    result.moved = true;
                    result.path = originalPath;
                    return result;
                }
                catch (Exception moveException)
                {
                    // Scoped storage, cross-volume moves, and provider-backed files can reject Move.
                    try
                    {
                        File.Copy(imagePath, originalPath, false);
                        result.success = true;
                        result.path = originalPath;
                        result.error = moveException.Message;
                        return result;
                    }
                    catch (Exception copyException)
                    {
                        result.error = moveException.Message + " | " + copyException.Message;
                        return result;
                    }
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception exception)
            {
                result.error = exception.Message;
                return result;
            }
        }, cancellationToken: cancellationToken);
    }

    private async UniTask<SaveFileResult> TryWriteFallbackAsync(
        byte[] outputBytes,
        string preferredPath,
        CancellationToken cancellationToken)
    {
        var fallbackPath = BuildFallbackSavePath(preferredPath);
        var result = await TryWriteBytesAsync(fallbackPath, outputBytes, cancellationToken);
        if (result.success)
            result.usedFallback = true;
        return result;
    }

    private static async UniTask<SaveFileResult> TryWriteBytesAsync(
        string destinationPath,
        byte[] outputBytes,
        CancellationToken cancellationToken)
    {
        return await UniTask.RunOnThreadPool(() =>
        {
            var result = new SaveFileResult { path = destinationPath };
            string temporaryPath = null;
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (string.IsNullOrWhiteSpace(destinationPath) || outputBytes == null || outputBytes.Length == 0)
                {
                    result.error = "Output image is empty";
                    return result;
                }

                var directory = Path.GetDirectoryName(destinationPath);
                if (string.IsNullOrWhiteSpace(directory))
                {
                    result.error = "Output path is invalid";
                    return result;
                }

                if (!Directory.Exists(directory))
                    Directory.CreateDirectory(directory);

                temporaryPath = Path.Combine(
                    directory,
                    "." + Path.GetFileName(destinationPath) + ".aexis-" + Guid.NewGuid().ToString("N") + ".tmp");
                File.WriteAllBytes(temporaryPath, outputBytes);

                if (File.Exists(destinationPath))
                {
                    try
                    {
                        File.Replace(temporaryPath, destinationPath, null);
                        temporaryPath = null;
                    }
                    catch
                    {
                        File.Copy(temporaryPath, destinationPath, true);
                    }
                }
                else
                {
                    File.Move(temporaryPath, destinationPath);
                    temporaryPath = null;
                }

                result.success = File.Exists(destinationPath);
                if (!result.success)
                    result.error = "Output file was not created";
                return result;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception exception)
            {
                result.error = exception.Message;
                return result;
            }
            finally
            {
                if (!string.IsNullOrWhiteSpace(temporaryPath))
                {
                    try
                    {
                        if (File.Exists(temporaryPath))
                            File.Delete(temporaryPath);
                    }
                    catch
                    {
                    }
                }
            }
        }, cancellationToken: cancellationToken);
    }

    private async UniTask<bool> CanAttemptDirectSaveAsync(string destinationPath, CancellationToken cancellationToken)
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!IsPathInsideApplicationStorage(destinationPath))
            return await EnsureAndroidExternalWritePermissionAsync(cancellationToken);
#endif
        return true;
    }

    private static string BuildFallbackSavePath(string preferredPath)
    {
        var directory = Path.Combine(Application.persistentDataPath, "SavedEdits");
        var baseName = Path.GetFileNameWithoutExtension(preferredPath);
        if (string.IsNullOrWhiteSpace(baseName))
            baseName = "image";
        var extension = Path.GetExtension(preferredPath);
        if (string.IsNullOrWhiteSpace(extension))
            extension = ".jpg";

        return Path.Combine(
            directory,
            baseName + "_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff") + "_" +
            Guid.NewGuid().ToString("N") + extension);
    }

    private static bool IsPathInsideApplicationStorage(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return false;

        return IsPathInsideRoot(path, Application.persistentDataPath) ||
               IsPathInsideRoot(path, Application.temporaryCachePath);
    }

    private static bool IsPathInsideRoot(string path, string root)
    {
        try
        {
            var normalizedPath = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var normalizedRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return string.Equals(normalizedPath, normalizedRoot, StringComparison.OrdinalIgnoreCase) ||
                   normalizedPath.StartsWith(normalizedRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) ||
                   normalizedPath.StartsWith(normalizedRoot + Path.AltDirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

#if UNITY_ANDROID
    private async UniTask<bool> EnsureAndroidExternalWritePermissionAsync(CancellationToken cancellationToken)
    {
        const string writeExternalStoragePermission = "android.permission.WRITE_EXTERNAL_STORAGE";
        if (Permission.HasUserAuthorizedPermission(writeExternalStoragePermission))
            return true;

        if (_saveStoragePermissionRequestInFlight)
        {
            while (_saveStoragePermissionRequestInFlight)
            {
                cancellationToken.ThrowIfCancellationRequested();
                await UniTask.Yield();
            }

            return Permission.HasUserAuthorizedPermission(writeExternalStoragePermission);
        }

        _saveStoragePermissionRequestInFlight = true;
        try
        {
            var callbacks = new PermissionCallbacks();
            var finished = false;
            var granted = false;
            callbacks.PermissionGranted += _ =>
            {
                granted = true;
                finished = true;
            };
            callbacks.PermissionDenied += _ => finished = true;
            callbacks.PermissionDeniedAndDontAskAgain += _ => finished = true;

            try
            {
                Permission.RequestUserPermission(writeExternalStoragePermission, callbacks);
            }
            catch
            {
                return false;
            }

            while (!finished)
            {
                cancellationToken.ThrowIfCancellationRequested();
                await UniTask.Yield();
            }

            return granted || Permission.HasUserAuthorizedPermission(writeExternalStoragePermission);
        }
        finally
        {
            _saveStoragePermissionRequestInFlight = false;
        }
    }
#endif

    private async UniTask TryRestoreMovedOriginalAsync(
        string imagePath,
        string originalPath,
        CancellationToken cancellationToken)
    {
        await UniTask.RunOnThreadPool(() =>
        {
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (string.IsNullOrWhiteSpace(imagePath) || string.IsNullOrWhiteSpace(originalPath) ||
                    !File.Exists(originalPath))
                {
                    return;
                }

                if (File.Exists(imagePath))
                    File.Delete(imagePath);
                File.Move(originalPath, imagePath);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
            }
        }, cancellationToken: cancellationToken);
    }

    private static string GetAvailableOriginalPath(string originalsDirectory, string fileName)
    {
        var candidate = Path.Combine(originalsDirectory, fileName);
        if (!File.Exists(candidate))
            return candidate;

        var baseName = Path.GetFileNameWithoutExtension(fileName);
        var extension = Path.GetExtension(fileName);
        for (var index = 2; ; index++)
        {
            candidate = Path.Combine(originalsDirectory, baseName + "_" + index + extension);
            if (!File.Exists(candidate))
                return candidate;
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

    private static void DestroyTexture<T>(ref T texture) where T : Texture
    {
        if (texture != null)
            Destroy(texture);
        texture = null;
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
