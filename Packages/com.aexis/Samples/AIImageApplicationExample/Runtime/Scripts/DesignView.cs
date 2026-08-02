using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Aexis.Samples.Async;
using Aexis.Execution;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UIElements;

public sealed class DesignView : BasePageView
{
    private const float MinLayerSize = 48f;
    private const float MinNormalizedLayerSize = 0.03f;
    private const int DefaultEdgeCloseRadius = 1;
    private const int DefaultEdgeFeatherRadius = 2;
    private const float DefaultEdgePreserve = 0.35f;

    private sealed class LayerBoxData
    {
        public string title;
        public Rect normalizedRect;
        public Color color;
        public RenderTexture contentRenderTexture;
        public Texture previewTexture;
        public RectInt sourceTextureRect;
    }

    private sealed class DetectionBuildArtifacts
    {
        public RenderTexture maskedBackgroundPreview;
        public RenderTexture maskedBackgroundHoleMask;
        public List<LayerBoxData> layers = new List<LayerBoxData>();
    }

    private sealed class ApplyCompositeResult
    {
        public Texture2D composedTexture;
        public Texture2D remainingMask;
        public int remainingMaskPixels;
        public string debugDirectory;
    }

    public override AppPageId PageId => AppPageId.DesignView;

    private readonly List<LayerBoxData> _layerData = new List<LayerBoxData>();
    private readonly List<VisualElement> _layerElements = new List<VisualElement>();
    private VisualElement _canvasOverlay;
    private VisualElement _tipsPanel;
    private VisualElement _tipsBody;
    private Label _tipsLabel;
    private Button _tipsToggleButton;
    private Button _applyButton;
    private Button _detectButton;
    private RenderTexture _maskedBackgroundPreview;
    private RenderTexture _maskedBackgroundHoleMask;
    private int _edgeCloseRadius = DefaultEdgeCloseRadius;
    private int _edgeFeatherRadius = DefaultEdgeFeatherRadius;
    private float _edgePreserve = DefaultEdgePreserve;
    private bool _tipsPanelCollapsed;
    private bool _hasAppliedTipsLayout;
    private bool _lastTipsLayoutWasPortrait;
    public bool _exportCompositeDebug = false;

    protected override AppPageId? ResolveSwipeTarget(SwipeDirection direction)
    {
        return direction == SwipeDirection.Left ? AppPageId.MainView2 : null;
    }

    protected override bool UseOverlaySwitchZone => true;

    protected override float GetSwitchPillAlignment01() => 1f;

    protected override bool HandleDirectionalImageNavigation(int direction)
    {
        return Host != null && Host.TryOpenAdjacentMainImage(direction);
    }

    protected override void BuildPage(VisualElement contentRoot)
    {
        PageRoot.RegisterCallback<KeyDownEvent>(OnPageKeyDown, TrickleDown.TrickleDown);
        BuildPageContent(contentRoot);
    }

    private void OnPageKeyDown(KeyDownEvent evt)
    {
        if ((evt.keyCode != KeyCode.Delete && evt.keyCode != KeyCode.Backspace) || IsTextFieldFocused())
            return;

        DeleteSelectedHistoryEntry();
        evt.StopPropagation();
        evt.PreventDefault();
    }

    protected override void OnShown()
    {
        if (CompareView != null)
        {
            CompareView.ViewTransformChanged -= OnCompareViewTransformChanged;
            CompareView.ViewTransformChanged += OnCompareViewTransformChanged;
        }

        var current = GetCurrentHistoryTexture();
        var originalHistory = GetOriginalHistoryTexture();
        if (current != null || originalHistory != null)
        {
            RefreshCompareSources();
            CompareView?.FitToView();
            RebuildLayerBoxes();
        }
    }

    protected override void OnBeforeDetach()
    {
        if (CompareView != null)
            CompareView.ViewTransformChanged -= OnCompareViewTransformChanged;
    }

    protected override void OnDestroy()
    {
        ClearDetectedLayerState();
        base.OnDestroy();
    }

    private void BuildPageContent(VisualElement contentRoot)
    {
        _hasAppliedTipsLayout = false;
        contentRoot.style.flexDirection = FlexDirection.Column;
        contentRoot.style.flexGrow = 1;
        contentRoot.style.minHeight = 0;

        contentRoot.Add(BuildTopBar());

        var body = new VisualElement();
        body.style.flexGrow = 1;
        body.style.minHeight = 0;
        body.style.position = Position.Relative;
        body.style.paddingLeft = 12;
        body.style.paddingRight = 12;
        body.style.paddingTop = 8;
        contentRoot.Add(body);

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
        body.Add(canvasHost);

        CreateCompareView(canvasHost, true);

        _canvasOverlay = new VisualElement();
        _canvasOverlay.style.position = Position.Absolute;
        _canvasOverlay.style.left = 0;
        _canvasOverlay.style.top = 0;
        _canvasOverlay.style.right = 0;
        _canvasOverlay.style.bottom = 0;
        _canvasOverlay.pickingMode = PickingMode.Ignore;
        canvasHost.Add(_canvasOverlay);

        canvasHost.Add(CreateFloatingHistoryPanel(230f, L("Design history", "设计历史")));

        _tipsPanel = new VisualElement();
        _tipsPanel.style.position = Position.Absolute;
        _tipsPanel.style.right = 18;
        _tipsPanel.style.top = 18;
        _tipsPanel.style.width = 280;
        _tipsPanel.style.backgroundColor = new StyleColor(new Color(0.11f, 0.12f, 0.16f, 0.88f));
        _tipsPanel.style.borderTopLeftRadius = 18;
        _tipsPanel.style.borderTopRightRadius = 18;
        _tipsPanel.style.borderBottomLeftRadius = 18;
        _tipsPanel.style.borderBottomRightRadius = 18;
        _tipsPanel.style.paddingLeft = 14;
        _tipsPanel.style.paddingRight = 14;
        _tipsPanel.style.paddingTop = 12;
        _tipsPanel.style.paddingBottom = 12;
        canvasHost.Add(_tipsPanel);
        EnableFloatingPanelDrag(_tipsPanel, _tipsPanel);

        var tipsHeader = new VisualElement();
        tipsHeader.style.flexDirection = FlexDirection.Row;
        tipsHeader.style.alignItems = Align.Center;
        _tipsPanel.Add(tipsHeader);

        var tipsTitle = new Label(L("Design notes", "设计说明"));
        tipsTitle.style.color = Color.white;
        tipsTitle.style.unityFontStyleAndWeight = FontStyle.Bold;
        tipsTitle.style.flexGrow = 1;
        tipsHeader.Add(tipsTitle);

        _tipsToggleButton = new Button(() => SetTipsPanelCollapsed(!_tipsPanelCollapsed));
        _tipsToggleButton.style.width = 28;
        _tipsToggleButton.style.minWidth = 28;
        _tipsToggleButton.style.height = 28;
        _tipsToggleButton.style.paddingLeft = 0;
        _tipsToggleButton.style.paddingRight = 0;
        _tipsToggleButton.style.paddingTop = 0;
        _tipsToggleButton.style.paddingBottom = 0;
        _tipsToggleButton.style.marginLeft = 8;
        _tipsToggleButton.style.borderTopLeftRadius = 14;
        _tipsToggleButton.style.borderTopRightRadius = 14;
        _tipsToggleButton.style.borderBottomLeftRadius = 14;
        _tipsToggleButton.style.borderBottomRightRadius = 14;
        tipsHeader.Add(_tipsToggleButton);

        _tipsBody = new VisualElement();
        _tipsPanel.Add(_tipsBody);

        _tipsLabel = new Label(L(
            "Select Detect layers to create draggable, resizable layers from YOLO Seg results. Detected people or objects are placed in layer boxes, and the removed background area is shown in black.",
            "点击“识别图层”后，会基于 YOLO Seg 结果生成可拖动、可缩放的图层。识别出的人物或对象会放进图层框里，被切走的背景区域会变黑。"));
        _tipsLabel.style.color = new Color(0.82f, 0.86f, 0.92f, 1f);
        _tipsLabel.style.whiteSpace = WhiteSpace.Normal;
        _tipsLabel.style.marginTop = 4;
        _tipsBody.Add(_tipsLabel);

        _tipsBody.Add(BuildBlendControls());

        BuildStandardOverlays();
        ApplyTipsPanelLayout(IsPortraitLayout);
    }

    protected override void OnLayoutChanged(bool isPortrait, Rect layoutRect)
    {
        RebuildLayerBoxes();
        ApplyTipsPanelLayout(isPortrait);
    }

    private void ApplyTipsPanelLayout(bool isPortrait)
    {
        if (_tipsPanel == null)
            return;

        if (isPortrait)
        {
            _tipsPanel.style.left = 18;
            _tipsPanel.style.right = 18;
            _tipsPanel.style.top = 18;
            _tipsPanel.style.width = new StyleLength(StyleKeyword.Auto);
        }
        else
        {
            _tipsPanel.style.left = new StyleLength(StyleKeyword.Auto);
            _tipsPanel.style.right = 18;
            _tipsPanel.style.top = 18;
            _tipsPanel.style.width = 280;
        }

        var orientationChanged = !_hasAppliedTipsLayout || _lastTipsLayoutWasPortrait != isPortrait;
        if (orientationChanged)
            SetTipsPanelCollapsed(isPortrait);

        _lastTipsLayoutWasPortrait = isPortrait;
        _hasAppliedTipsLayout = true;
    }

    private void SetTipsPanelCollapsed(bool collapsed)
    {
        _tipsPanelCollapsed = collapsed;
        if (_tipsBody != null)
            _tipsBody.style.display = collapsed ? DisplayStyle.None : DisplayStyle.Flex;
        if (_tipsToggleButton != null)
        {
            _tipsToggleButton.text = collapsed ? "\u25BE" : "\u25B4";
            _tipsToggleButton.tooltip = collapsed ? "\u5C55\u5F00\u8BBE\u8BA1\u8BF4\u660E" : "\u6536\u8D77\u8BBE\u8BA1\u8BF4\u660E";
        }
    }

    public void SyncFromMainView(string path, Texture2D currentTexture, Texture2D originalTexture, string label)
    {
        if (currentTexture == null && originalTexture == null)
        {
            ClearDerivedDesignState();
            return;
        }

        ClearDerivedDesignState();
        SetHistoryFromSharedTextures(originalTexture, currentTexture, label, path);
        RefreshCompareSources();
        CompareView?.FitToView();
        RebuildLayerBoxes();
    }

    private void OnCompareViewTransformChanged()
    {
        RebuildLayerBoxes();
    }

    private void RefreshCompareSources()
    {
        if (CompareView == null)
            return;

        var current = GetCurrentHistoryTexture();
        var original = GetOriginalHistoryTexture();
        CompareView.SetPreview(null);

        if (_maskedBackgroundPreview != null)
        {
            Texture compareOriginal = original != null ? original : (current != null ? current : _maskedBackgroundPreview);
            CompareView.SetSources(
                _maskedBackgroundPreview,
                compareOriginal,
                GetCurrentHistoryLabel());
            return;
        }

        CompareView.SetSources(current ?? original, original ?? current, GetCurrentHistoryLabel());
    }

    private VisualElement BuildTopBar()
    {
        var bar = new VisualElement();
        bar.style.flexShrink = 0;
        bar.style.paddingLeft = 12;
        bar.style.paddingRight = 12;
        bar.style.paddingTop = 10;
        bar.style.paddingBottom = 8;
        bar.style.flexDirection = FlexDirection.Row;
        bar.style.alignItems = Align.Center;

        var title = new Label(L("Design", "设计"));
        title.style.fontSize = 18;
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        title.style.marginRight = 12;
        bar.Add(title);

        _detectButton = CreateActionButton(L("Detect layers", "识别图层"), OnDetectLayers);
        bar.Add(_detectButton);

        _applyButton = CreateActionButton(L("Apply design", "应用生成"), OnApplyDesign);
        _applyButton.style.marginLeft = 8;
        bar.Add(_applyButton);
        return bar;
    }

    private static Button CreateActionButton(string text, Action onClick)
    {
        var button = new Button(onClick) { text = text };
        button.style.height = 36;
        button.style.paddingLeft = 16;
        button.style.paddingRight = 16;
        button.style.backgroundColor = new StyleColor(new Color(0.17f, 0.54f, 0.95f, 1f));
        button.style.color = Color.white;
        button.style.borderTopLeftRadius = 18;
        button.style.borderTopRightRadius = 18;
        button.style.borderBottomLeftRadius = 18;
        button.style.borderBottomRightRadius = 18;
        return button;
    }

    private VisualElement BuildBlendControls()
    {
        var host = new VisualElement();
        host.style.marginTop = 10;
        host.style.paddingTop = 10;
        host.style.borderTopWidth = 1;
        host.style.borderTopColor = new StyleColor(new Color(1f, 1f, 1f, 0.08f));

        var title = new Label(L("Edge blending", "边缘融合"));
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        host.Add(title);

        var desc = new Label(L(
            "Applies light morphological closing and alpha feathering to subject edges, suitable for hair, translucent edges, and minor jaggies.",
            "默认会对人物边缘做轻量形态学闭运算和 alpha 羽化，适合发丝、半透明边缘和轻微锯齿。"));
        desc.style.color = new Color(0.76f, 0.82f, 0.9f, 1f);
        desc.style.whiteSpace = WhiteSpace.Normal;
        desc.style.marginTop = 4;
        host.Add(desc);

        host.Add(CreateBlendSliderRow(L("Closing", "闭运算"), 0, 4, _edgeCloseRadius, "px", v => _edgeCloseRadius = Mathf.RoundToInt(v)));
        host.Add(CreateBlendSliderRow(L("Feather", "羽化"), 0, 80, _edgeFeatherRadius, "px", v => _edgeFeatherRadius = Mathf.RoundToInt(v)));
        host.Add(CreateBlendSliderRow(L("Edge fidelity", "边缘保真"), 0f, 1f, _edgePreserve, "", v => _edgePreserve = Mathf.Clamp01(v), "0.00"));

        host.Add(CreateToggleRow("Debug Composite Export", _exportCompositeDebug, v => _exportCompositeDebug = v));
        return host;
    }

    private static VisualElement CreateToggleRow(string labelText, bool defaultValue, Action<bool> onChanged)
    {
        var row = new VisualElement();
        row.style.marginTop = 10;
        row.style.flexDirection = FlexDirection.Row;
        row.style.alignItems = Align.Center;

        var label = new Label(labelText);
        label.style.flexGrow = 1;
        label.style.color = Color.white;
        row.Add(label);

        var toggle = new Toggle { value = defaultValue };
        toggle.RegisterValueChangedCallback(evt => onChanged?.Invoke(evt.newValue));
        row.Add(toggle);

        return row;
    }

    private static VisualElement CreateBlendSliderRow(string labelText, float min, float max, float defaultValue, string suffix, Action<float> onChanged, string valueFormat = "0")
    {
        var row = new VisualElement();
        row.style.marginTop = 8;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        row.Add(header);

        var label = new Label(labelText);
        label.style.flexGrow = 1;
        label.style.color = Color.white;
        header.Add(label);

        var valueLabel = new Label(FormatBlendSliderValue(defaultValue, suffix, valueFormat));
        valueLabel.style.minWidth = 54;
        valueLabel.style.unityTextAlign = TextAnchor.MiddleRight;
        valueLabel.style.color = new Color(0.78f, 0.88f, 1f, 1f);
        header.Add(valueLabel);

        var slider = new Slider(min, max) { value = defaultValue };
        slider.style.marginTop = 4;
        slider.RegisterValueChangedCallback(evt =>
        {
            valueLabel.text = FormatBlendSliderValue(evt.newValue, suffix, valueFormat);
            onChanged?.Invoke(evt.newValue);
        });
        row.Add(slider);

        return row;
    }

    private static string FormatBlendSliderValue(float value, string suffix, string valueFormat)
    {
        if (string.IsNullOrEmpty(suffix))
            return value.ToString(valueFormat);
        return value.ToString(valueFormat) + suffix;
    }

    private void OnDetectLayers()
    {
        DetectLayersAsync().Forget();
    }

    private async UniTaskVoid DetectLayersAsync()
    {
        var src = GetCurrentHistoryTexture() ?? GetOriginalHistoryTexture();
        if (src == null || Host?.YoloSegRunner == null)
            return;

        ShowProgress(L("Detect layers", "识别图层"));
        YoloSegResult result = default;
        DetectionBuildArtifacts artifacts = null;
        try
        {
            var oldTargetPersonOnly = Host.YoloSegRunner.targetPersonOnly;
            Host.YoloSegRunner.targetPersonOnly = false;
            void OnProgress(float p, string t) => SetProgress(p, t);
            Host.YoloSegRunner.ProgressChanged -= OnProgress;
            Host.YoloSegRunner.ProgressChanged += OnProgress;
            try
            {
                result = await Host.YoloSegRunner.ProcessAsync(src, default);
            }
            finally
            {
                Host.YoloSegRunner.ProgressChanged -= OnProgress;
                Host.YoloSegRunner.targetPersonOnly = oldTargetPersonOnly;
            }

            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3000);
                return;
            }

            SetProgress(0.9f, "Build design layers");
            artifacts = await BuildDetectionArtifactsAsync(src, result);
            if (artifacts == null || artifacts.layers == null || artifacts.layers.Count == 0)
            {
                ShowToast("未检测到可用的人物或对象图层", 2600);
                return;
            }

            ReplaceDetectedLayerState(artifacts.layers, artifacts.maskedBackgroundPreview, artifacts.maskedBackgroundHoleMask);
            artifacts.layers = null;
            artifacts.maskedBackgroundPreview = null;
            artifacts.maskedBackgroundHoleMask = null;

            if (_tipsLabel != null)
                _tipsLabel.text = L(
                    $"Created {_layerData.Count} layers. They contain the extracted people or objects, with their background areas shown in black. Moving or resizing a layer moves its content with it.",
                    $"已生成 {_layerData.Count} 个图层。图层里是切出的人物或对象，背景对应区域已变黑，移动或缩放图层时内容会一起跟随。");
        }
        finally
        {
            if (artifacts != null)
            {
                DestroyLayerTextures(artifacts.layers);
                DestroyRenderTexture(ref artifacts.maskedBackgroundPreview);
                DestroyRenderTexture(ref artifacts.maskedBackgroundHoleMask);
            }
            DestroyTexture(ref result.texture);
            DestroyTexture(ref result.mask);
            DestroyTexture(ref result.overlay);
            HideProgress();
        }
    }

    private void OnApplyDesign()
    {
        ApplyDesignAsync().Forget();
    }

    private void ClearDerivedDesignState()
    {
        ClearDetectedLayerState();
        if (_tipsLabel != null)
            _tipsLabel.text = L(
                "Select Detect layers to create draggable, resizable layers from YOLO Seg results. Detected people or objects are placed in layer boxes, and the removed background area is shown in black.",
                "点击“识别图层”后，会基于 YOLO Seg 结果生成可拖动、可缩放的图层。识别出的人物或对象会放进图层框里，被切走的背景区域会变黑。");
    }

    private void RebuildLayerBoxes()
    {
        if (_canvasOverlay == null || CompareView == null)
            return;

        _canvasOverlay.Clear();
        _layerElements.Clear();

        if (!CompareView.TryGetDisplayedImageRect(out var imageRect))
            return;

        foreach (var layer in _layerData)
        {
            var box = BuildLayerBox(layer, imageRect);
            _canvasOverlay.Add(box);
            _layerElements.Add(box);
        }
    }

    private VisualElement BuildLayerBox(LayerBoxData data, Rect imageRect)
    {
        var rect = new Rect(
            imageRect.xMin + data.normalizedRect.x * imageRect.width,
            imageRect.yMin + data.normalizedRect.y * imageRect.height,
            Mathf.Max(MinLayerSize, data.normalizedRect.width * imageRect.width),
            Mathf.Max(MinLayerSize, data.normalizedRect.height * imageRect.height));

        var box = new VisualElement();
        box.style.position = Position.Absolute;
        box.style.left = rect.x;
        box.style.top = rect.y;
        box.style.width = rect.width;
        box.style.height = rect.height;
        box.style.borderLeftWidth = 2;
        box.style.borderRightWidth = 2;
        box.style.borderTopWidth = 2;
        box.style.borderBottomWidth = 2;
        box.style.borderLeftColor = new StyleColor(data.color);
        box.style.borderRightColor = new StyleColor(data.color);
        box.style.borderTopColor = new StyleColor(data.color);
        box.style.borderBottomColor = new StyleColor(data.color);
        box.style.backgroundColor = new StyleColor(new Color(data.color.r, data.color.g, data.color.b, 0.08f));
        box.pickingMode = PickingMode.Position;

        var contentHost = new VisualElement();
        contentHost.style.position = Position.Absolute;
        contentHost.style.left = 0;
        contentHost.style.top = 0;
        contentHost.style.right = 0;
        contentHost.style.bottom = 0;
        contentHost.style.overflow = Overflow.Hidden;
        contentHost.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.14f));
        contentHost.pickingMode = PickingMode.Ignore;
        box.Add(contentHost);

        if (data.previewTexture != null)
        {
            var contentImage = new Image();
            contentImage.image = data.previewTexture;
            contentImage.scaleMode = ScaleMode.StretchToFill;
            contentImage.style.position = Position.Absolute;
            contentImage.style.left = 0;
            contentImage.style.top = 0;
            contentImage.style.right = 0;
            contentImage.style.bottom = 0;
            contentImage.pickingMode = PickingMode.Ignore;
            contentHost.Add(contentImage);
        }

        var header = new VisualElement();
        header.style.position = Position.Absolute;
        header.style.left = 0;
        header.style.right = 0;
        header.style.top = 0;
        header.style.height = 26;
        header.style.backgroundColor = new StyleColor(new Color(data.color.r, data.color.g, data.color.b, 0.82f));
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.paddingLeft = 8;
        header.style.paddingRight = 8;
        header.style.justifyContent = Justify.SpaceBetween;
        box.Add(header);

        var title = new Label(data.title);
        title.style.color = Color.black;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        header.Add(title);

        foreach (var handleName in new[] { "tl", "tr", "bl", "br" })
            box.Add(CreateHandle(handleName));

        SetupDragAndResize(box, header, data);
        return box;
    }

    private static VisualElement CreateHandle(string corner)
    {
        var handle = new VisualElement();
        handle.name = corner;
        handle.style.position = Position.Absolute;
        handle.style.width = 14;
        handle.style.height = 14;
        handle.style.backgroundColor = Color.white;
        handle.style.borderTopLeftRadius = 3;
        handle.style.borderTopRightRadius = 3;
        handle.style.borderBottomLeftRadius = 3;
        handle.style.borderBottomRightRadius = 3;

        switch (corner)
        {
            case "tl":
                handle.style.left = -7;
                handle.style.top = -7;
                break;
            case "tr":
                handle.style.right = -7;
                handle.style.top = -7;
                break;
            case "bl":
                handle.style.left = -7;
                handle.style.bottom = -7;
                break;
            case "br":
                handle.style.right = -7;
                handle.style.bottom = -7;
                break;
        }

        return handle;
    }

    private void SetupDragAndResize(VisualElement box, VisualElement header, LayerBoxData data)
    {
        var dragging = false;
        var dragPointerId = -1;
        var startPointer = Vector2.zero;
        var startRect = default(Rect);

        header.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (evt.button != 0)
                return;

            dragging = true;
            dragPointerId = evt.pointerId;
            startPointer = evt.position;
            startRect = GetCurrentBoxRect(box);
            header.CapturePointer(dragPointerId);
            evt.StopPropagation();
        });

        header.RegisterCallback<PointerMoveEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId || !header.HasPointerCapture(dragPointerId))
                return;

            var delta = (Vector2)evt.position - startPointer;
            ApplyBoxRect(box, data, new Rect(startRect.x + delta.x, startRect.y + delta.y, startRect.width, startRect.height));
            evt.StopPropagation();
        });

        header.RegisterCallback<PointerUpEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId)
                return;

            dragging = false;
            if (header.HasPointerCapture(dragPointerId))
                header.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });

        header.RegisterCallback<PointerCancelEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId)
                return;

            dragging = false;
            if (header.HasPointerCapture(dragPointerId))
                header.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });

        foreach (var handle in box.Children())
        {
            if (handle.name != "tl" && handle.name != "tr" && handle.name != "bl" && handle.name != "br")
                continue;

            var corner = handle.name;
            var resizing = false;
            var resizePointerId = -1;
            var resizeStart = Vector2.zero;
            var resizeStartRect = default(Rect);

            handle.RegisterCallback<PointerDownEvent>(evt =>
            {
                if (evt.button != 0)
                    return;

                resizing = true;
                resizePointerId = evt.pointerId;
                resizeStart = evt.position;
                resizeStartRect = GetCurrentBoxRect(box);
                handle.CapturePointer(resizePointerId);
                evt.StopPropagation();
            });

            handle.RegisterCallback<PointerMoveEvent>(evt =>
            {
                if (!resizing || resizePointerId != evt.pointerId || !handle.HasPointerCapture(resizePointerId))
                    return;

                var delta = (Vector2)evt.position - resizeStart;
                var rect = resizeStartRect;
                switch (corner)
                {
                    case "tl":
                        rect.x += delta.x;
                        rect.y += delta.y;
                        rect.width -= delta.x;
                        rect.height -= delta.y;
                        break;
                    case "tr":
                        rect.y += delta.y;
                        rect.width += delta.x;
                        rect.height -= delta.y;
                        break;
                    case "bl":
                        rect.x += delta.x;
                        rect.width -= delta.x;
                        rect.height += delta.y;
                        break;
                    case "br":
                        rect.width += delta.x;
                        rect.height += delta.y;
                        break;
                }

                ApplyBoxRect(box, data, rect);
                evt.StopPropagation();
            });

            handle.RegisterCallback<PointerUpEvent>(evt =>
            {
                if (!resizing || resizePointerId != evt.pointerId)
                    return;

                resizing = false;
                if (handle.HasPointerCapture(resizePointerId))
                    handle.ReleasePointer(resizePointerId);
                evt.StopPropagation();
            });

            handle.RegisterCallback<PointerCancelEvent>(evt =>
            {
                if (!resizing || resizePointerId != evt.pointerId)
                    return;

                resizing = false;
                if (handle.HasPointerCapture(resizePointerId))
                    handle.ReleasePointer(resizePointerId);
                evt.StopPropagation();
            });
        }
    }

    private void ApplyBoxRect(VisualElement box, LayerBoxData data, Rect rect)
    {
        if (!CompareView.TryGetDisplayedImageRect(out var imageRect))
            return;

        rect.width = Mathf.Max(MinLayerSize, rect.width);
        rect.height = Mathf.Max(MinLayerSize, rect.height);
        rect.x = Mathf.Clamp(rect.x, imageRect.xMin, imageRect.xMax - rect.width);
        rect.y = Mathf.Clamp(rect.y, imageRect.yMin, imageRect.yMax - rect.height);
        rect.width = Mathf.Min(rect.width, imageRect.xMax - rect.x);
        rect.height = Mathf.Min(rect.height, imageRect.yMax - rect.y);

        box.style.left = rect.x;
        box.style.top = rect.y;
        box.style.width = rect.width;
        box.style.height = rect.height;

        data.normalizedRect = new Rect(
            (rect.x - imageRect.xMin) / imageRect.width,
            (rect.y - imageRect.yMin) / imageRect.height,
            rect.width / imageRect.width,
            rect.height / imageRect.height);
    }

    private static Rect GetCurrentBoxRect(VisualElement box)
    {
        return new Rect(box.resolvedStyle.left, box.resolvedStyle.top, box.resolvedStyle.width, box.resolvedStyle.height);
    }

    private void ReplaceDetectedLayerState(List<LayerBoxData> layers, RenderTexture maskedBackgroundPreview, RenderTexture maskedBackgroundHoleMask)
    {
        ClearDetectedLayerState();

        if (layers != null && layers.Count > 0)
            _layerData.AddRange(layers);

        _maskedBackgroundPreview = maskedBackgroundPreview;
        _maskedBackgroundHoleMask = maskedBackgroundHoleMask;
        RefreshCompareSources();
        RebuildLayerBoxes();
    }

    private void ClearDetectedLayerState()
    {
        DestroyLayerTextures(_layerData);
        _layerData.Clear();

        CompareView?.SetPreview(null);
        DestroyRenderTexture(ref _maskedBackgroundPreview);
        DestroyRenderTexture(ref _maskedBackgroundHoleMask);

        _canvasOverlay?.Clear();
        _layerElements.Clear();
    }

    private static void DestroyLayerTextures(List<LayerBoxData> layers)
    {
        if (layers == null)
            return;

        for (var i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            if (layer == null)
                continue;

            DestroyRenderTexture(ref layer.contentRenderTexture);
            DestroyTexture(ref layer.previewTexture);
        }
    }

    private static void DestroyTexture(ref Texture texture)
    {
        if (texture == null)
            return;

        UnityEngine.Object.Destroy(texture);
        texture = null;
    }

    private static void DestroyTexture(ref Texture2D texture)
    {
        if (texture == null)
            return;

        UnityEngine.Object.Destroy(texture);
        texture = null;
    }

    private static void DestroyRenderTexture(ref RenderTexture texture)
    {
        if (texture == null)
            return;

        texture.Release();
        UnityEngine.Object.Destroy(texture);
        texture = null;
    }

    private async UniTask<DetectionBuildArtifacts> BuildDetectionArtifactsAsync(Texture2D source, YoloSegResult result)
    {
        if (source == null || result.mask == null || result.detections == null || result.detections.Length == 0)
            return null;

        var cs = Host?.ImageProcessingCS;
        if (cs == null)
            return null;

        var artifacts = new DetectionBuildArtifacts();
        RenderTexture maskedBackgroundRt = null;
        RenderTexture holeMaskRt = null;
        RenderTexture placedMaskRt = null;
        RenderTexture nextPlacedMaskRt = null;
        var tempRts = new List<RenderTexture>();
        try
        {
            placedMaskRt = CreateWorkingRenderTexture(source.width, source.height, "DesignViewPlacedLayerMaskSeed");
            if (placedMaskRt == null)
                return null;
            var prevActive = RenderTexture.active;
            RenderTexture.active = placedMaskRt;
            GL.Clear(false, true, Color.black);
            RenderTexture.active = prevActive;

            for (var i = 0; i < result.detections.Length; i++)
            {
                var detection = result.detections[i];
                var displayRect = ToDisplayPixelRect(detection.rect, source.width, source.height);
                if (displayRect.width < 8 || displayRect.height < 8)
                    continue;

                var normalized = new Rect(
                    displayRect.x / Mathf.Max(1f, source.width),
                    displayRect.y / Mathf.Max(1f, source.height),
                    displayRect.width / Mathf.Max(1f, source.width),
                    displayRect.height / Mathf.Max(1f, source.height));
                if (normalized.width < MinNormalizedLayerSize || normalized.height < MinNormalizedLayerSize)
                    continue;

                var textureRect = ToTexturePixelRect(displayRect, source.width, source.height);
                var cutoutRt = BuildLayerCutoutRenderTexture(source, result.mask, textureRect, cs);
                if (cutoutRt == null)
                    continue;

                tempRts.Add(cutoutRt);
                artifacts.layers.Add(new LayerBoxData
                {
                    title = $"Layer {i + 1}  {(detection.probability * 100f):0}%",
                    normalizedRect = normalized,
                    color = Color.HSVToRGB((i * 0.17f) % 1f, 0.68f, 1f),
                    previewTexture = cutoutRt,
                    contentRenderTexture = cutoutRt,
                    sourceTextureRect = textureRect
                });
                tempRts.Remove(cutoutRt);

                nextPlacedMaskRt = BuildPlacedLayerMaskRenderTexture(placedMaskRt, cutoutRt, displayRect, source.width, source.height, cs);
                if (nextPlacedMaskRt != null)
                {
                    DestroyRenderTexture(ref placedMaskRt);
                    placedMaskRt = nextPlacedMaskRt;
                    nextPlacedMaskRt = null;
                }
            }

            if (artifacts.layers.Count == 0 || placedMaskRt == null)
                return null;

            maskedBackgroundRt = await BuildMaskedBackgroundRenderTextureAsync(source, placedMaskRt, cs);
            holeMaskRt = await BuildHoleMaskRenderTextureAsync(placedMaskRt, cs);
            if (maskedBackgroundRt == null || holeMaskRt == null)
                return null;

            artifacts.maskedBackgroundPreview = maskedBackgroundRt;
            artifacts.maskedBackgroundHoleMask = holeMaskRt;
            maskedBackgroundRt = null;
            holeMaskRt = null;

            Debug.Log("[DesignView] BuildDetectionArtifactsAsync done | layers=" + artifacts.layers.Count.ToString(CultureInfo.InvariantCulture));
            return artifacts;
        }
        finally
        {
            DestroyRenderTexture(ref maskedBackgroundRt);
            DestroyRenderTexture(ref holeMaskRt);
            DestroyRenderTexture(ref placedMaskRt);
            DestroyRenderTexture(ref nextPlacedMaskRt);
            for (var i = 0; i < tempRts.Count; i++)
            {
                var rt = tempRts[i];
                DestroyRenderTexture(ref rt);
            }
        }
    }

    private static RenderTexture CreateRenderTextureFromTexture(Texture texture)
    {
        if (texture == null || texture.width <= 0 || texture.height <= 0)
            return null;

        var rt = new RenderTexture(texture.width, texture.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
        {
            enableRandomWrite = true,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear
        };
        rt.Create();
        Graphics.Blit(texture, rt);
        return rt;
    }

    private async UniTask<RenderTexture> BuildMaskedBackgroundRenderTextureAsync(Texture source, Texture mask, ComputeShader cs)
    {
        if (source == null || mask == null || cs == null)
            return null;

        int kernel;
        try { kernel = cs.FindKernel("BuildMaskedBackgroundFromMask"); }
        catch { return null; }

        var rt = CreateWorkingRenderTexture(source.width, source.height, "DesignViewMaskedBackgroundRt");
        if (rt == null)
            return null;

        cs.SetTexture(kernel, "_Source", source);
        cs.SetTexture(kernel, "_Overlay", mask);
        cs.SetTexture(kernel, "_Result", rt);
        cs.Dispatch(kernel, Mathf.Max(1, Mathf.CeilToInt(source.width / 8f)), Mathf.Max(1, Mathf.CeilToInt(source.height / 8f)), 1);
        await UniTask.Yield();
        return rt;
    }

    private async UniTask<RenderTexture> BuildHoleMaskRenderTextureAsync(Texture mask, ComputeShader cs)
    {
        if (mask == null || cs == null)
            return null;

        int kernel;
        try { kernel = cs.FindKernel("BuildHoleMaskFromMask"); }
        catch { return null; }

        var rt = CreateWorkingRenderTexture(mask.width, mask.height, "DesignViewHoleMaskRt");
        if (rt == null)
            return null;

        cs.SetTexture(kernel, "_Source", mask);
        cs.SetTexture(kernel, "_Overlay", mask);
        cs.SetTexture(kernel, "_Result", rt);
        cs.Dispatch(kernel, Mathf.Max(1, Mathf.CeilToInt(mask.width / 8f)), Mathf.Max(1, Mathf.CeilToInt(mask.height / 8f)), 1);
        await UniTask.Yield();
        return rt;
    }

    private RenderTexture BuildLayerCutoutRenderTexture(Texture source, Texture mask, RectInt pixelRect, ComputeShader cs)
    {
        if (source == null || mask == null || cs == null || pixelRect.width <= 0 || pixelRect.height <= 0)
            return null;

        int kernel;
        try { kernel = cs.FindKernel("BuildLayerCutoutFromMask"); }
        catch { return null; }

        var rt = CreateWorkingRenderTexture(pixelRect.width, pixelRect.height, "DesignViewLayerCutoutRt");
        if (rt == null)
            return null;

        cs.SetTexture(kernel, "_Source", source);
        cs.SetTexture(kernel, "_Overlay", mask);
        cs.SetTexture(kernel, "_Result", rt);
        cs.SetInts("_CropRect", pixelRect.x, pixelRect.y, pixelRect.width, pixelRect.height);
        cs.Dispatch(kernel, Mathf.Max(1, Mathf.CeilToInt(pixelRect.width / 8f)), Mathf.Max(1, Mathf.CeilToInt(pixelRect.height / 8f)), 1);
        return rt;
    }

    private RenderTexture BuildPlacedLayerMaskRenderTexture(RenderTexture existingMask, RenderTexture layerTexture, RectInt displayRect, int canvasWidth, int canvasHeight, ComputeShader cs)
    {
        if (existingMask == null || layerTexture == null || cs == null || displayRect.width <= 0 || displayRect.height <= 0 || canvasWidth <= 0 || canvasHeight <= 0)
            return null;

        int kernel;
        try { kernel = cs.FindKernel("DesignViewAccumulateLayerMask"); }
        catch { return null; }

        var rt = CreateWorkingRenderTexture(canvasWidth, canvasHeight, "DesignViewPlacedLayerMaskRt");
        if (rt == null)
            return null;

        cs.SetTexture(kernel, "_Source", existingMask);
        cs.SetTexture(kernel, "_Overlay", layerTexture);
        cs.SetTexture(kernel, "_Result", rt);
        cs.SetInts("_CropRect", displayRect.x, displayRect.y, displayRect.width, displayRect.height);
        cs.SetInts("_DesignViewCanvasSize", canvasWidth, canvasHeight);
        cs.Dispatch(kernel, Mathf.Max(1, Mathf.CeilToInt(canvasWidth / 8f)), Mathf.Max(1, Mathf.CeilToInt(canvasHeight / 8f)), 1);
        return rt;
    }

    private static bool IsMaskedPixel(Color32 pixel)
    {
        return pixel.r >= 128 || pixel.g >= 128 || pixel.b >= 128;
    }

    private static RectInt ToDisplayPixelRect(Rect rect, int width, int height)
    {
        if (width <= 0 || height <= 0)
            return default;

        var xMin = Mathf.Clamp(Mathf.FloorToInt(rect.xMin), 0, Mathf.Max(0, width - 1));
        var yMin = Mathf.Clamp(Mathf.FloorToInt(rect.yMin), 0, Mathf.Max(0, height - 1));
        var xMax = Mathf.Clamp(Mathf.CeilToInt(rect.xMax), xMin + 1, width);
        var yMax = Mathf.Clamp(Mathf.CeilToInt(rect.yMax), yMin + 1, height);
        return new RectInt(xMin, yMin, Mathf.Max(1, xMax - xMin), Mathf.Max(1, yMax - yMin));
    }

    private static RectInt ToTexturePixelRect(RectInt displayRect, int width, int height)
    {
        if (displayRect.width <= 0 || displayRect.height <= 0 || width <= 0 || height <= 0)
            return default;

        var textureY = height - displayRect.y - displayRect.height;
        textureY = Mathf.Clamp(textureY, 0, Mathf.Max(0, height - displayRect.height));
        return new RectInt(displayRect.x, textureY, displayRect.width, displayRect.height);
    }

    private async UniTaskVoid ApplyDesignAsync()
    {
        if (_layerData.Count == 0)
        {
            ShowToast("当前没有可应用的图层", 2200);
            return;
        }

        var originalHistory = GetOriginalHistoryTexture();
        if (originalHistory == null || Host?.DeepFillV2Runner == null)
        {
            ShowToast("找不到原始背景图", 2400);
            return;
        }

        if (_maskedBackgroundPreview == null || _maskedBackgroundHoleMask == null)
        {
            ShowToast("缺少背景或遮罩数据，请先重新识别图层", 2600);
            return;
        }

        if (!await Host.EnsureModelGroupsAvailableAsync(
                "Design background inpainting model download",
                System.Threading.CancellationToken.None,
                AIImageModelGroupId.DeepFillV2Case1Ncnn))
            return;

        ApplyCompositeResult applyResult = null;
        Texture2D inpaintedBackground = null;
        ShowProgress(L("Apply design", "应用生成"));
        try
        {
            var deepFillRunner = Host.DeepFillV2Runner;
            deepFillRunner.backend = DeepFillV2Backend.NcnnBin;
            deepFillRunner.enableDebugDump = _exportCompositeDebug;
            deepFillRunner.precisionMode = AexisPrecisionMode.Auto;
            deepFillRunner.useArgbFloatTensor = false;
            deepFillRunner.preserveUnmaskedPixels = true;
            deepFillRunner.enableGeneralTextureConvolution = true;
            deepFillRunner.enableDepthWiseTextureConvolution = true;
            deepFillRunner.enableConv1x1TextureConvolution = true;

            void OnInpaintProgress(float progress, string stage)
            {
                SetProgress(0.05f + 0.75f * progress, string.IsNullOrWhiteSpace(stage) ? "DeepFillV2" : "DeepFillV2 " + stage);
            }

            SetProgress(0.02f, L("Prepare background inpainting", "准备背景补全"));
            deepFillRunner.ProgressChanged -= OnInpaintProgress;
            deepFillRunner.ProgressChanged += OnInpaintProgress;
            DeepFillV2Result inpaintResult;
            try
            {
                inpaintResult = await deepFillRunner.ProcessAsync(
                    _maskedBackgroundPreview,
                    _maskedBackgroundHoleMask,
                    System.Threading.CancellationToken.None);
            }
            finally
            {
                deepFillRunner.ProgressChanged -= OnInpaintProgress;
                // The result is CPU-owned; release Pack4 intermediates before the composite dispatches.
                deepFillRunner.Release();
            }

            if (!string.IsNullOrWhiteSpace(inpaintResult.error) || inpaintResult.texture == null)
            {
                if (inpaintResult.texture != null)
                    DestroyTexture(ref inpaintResult.texture);
                ShowToast(string.IsNullOrWhiteSpace(inpaintResult.error) ? "DeepFillV2 背景补全失败" : inpaintResult.error, 3600);
                return;
            }

            inpaintedBackground = inpaintResult.texture;
            inpaintResult.texture = null;

            SetProgress(0.83f, L("Composite layers", "合成图层"));
            applyResult = await BuildAppliedCompositeAsync(
                inpaintedBackground,
                inpaintedBackground,
                _maskedBackgroundHoleMask,
                _layerData,
                _edgeCloseRadius,
                _edgeFeatherRadius,
                _edgePreserve);
            if (applyResult?.composedTexture == null)
            {
                ShowToast("图层合成失败", 2600);
                return;
            }

            AddHistory(applyResult.composedTexture, "设计合成");
            applyResult.composedTexture = null;

            ClearDetectedLayerState();

            var current = GetCurrentHistoryTexture();
            CompareView?.SetSources(current ?? originalHistory, originalHistory ?? current, GetCurrentHistoryLabel());
            CompareView?.FitToView();
            if (_exportCompositeDebug && !string.IsNullOrWhiteSpace(applyResult.debugDirectory))
                OpenFolderInShell(applyResult.debugDirectory);

            if (_tipsLabel != null)
            {
                _tipsLabel.text = applyResult.remainingMaskPixels > 0
                    ? L(
                        $"Created a new design result. DeepFillV2 restored {applyResult.remainingMaskPixels} background-mask pixels before the layers were composited.",
                        $"已生成新的设计结果。DeepFillV2 已在合成图层前补全 {applyResult.remainingMaskPixels} 个背景遮罩像素。")
                    : L(
                        "Created a new design result. DeepFillV2 restored the background and all current layers were composited.",
                        "已生成新的设计结果。DeepFillV2 已补全背景，当前图层已全部合成回背景。");
            }

            ShowToast(
                applyResult.remainingMaskPixels > 0
                    ? L(
                        "DeepFillV2 restored the masked background and the layers were composited.",
                        "DeepFillV2 已补全遮罩背景，并已合成图层。")
                    : L(
                        "Composited into the restored background and cleared all layers.",
                        "已合成到补全背景，并已清除所有图层。"),
                3200);
        }
        finally
        {
            if (applyResult?.composedTexture != null)
                DestroyTexture(ref applyResult.composedTexture);
            if (applyResult?.remainingMask != null)
                DestroyTexture(ref applyResult.remainingMask);
            DestroyTexture(ref inpaintedBackground);
            HideProgress();
            await UniTask.Yield();
        }
    }

    private async UniTask<ApplyCompositeResult> BuildAppliedCompositeAsync(
        Texture baseTexture,
        Texture backgroundReference,
        Texture holeMask,
        List<LayerBoxData> layers,
        int edgeCloseRadius,
        int edgeFeatherRadius,
        float edgePreserve)
    {
        if (baseTexture == null || backgroundReference == null || holeMask == null || layers == null || layers.Count == 0)
            return null;

        var cs = Host?.ImageProcessingCS;
        if (cs == null)
            return null;

        int kernel;
        try { kernel = cs.FindKernel("DesignViewCompositeLayer"); }
        catch { return null; }
        if (kernel < 0)
            return null;

        int buildClosedMaskKernel;
        try { buildClosedMaskKernel = cs.FindKernel("DesignViewBuildClosedMask"); }
        catch { return null; }
        if (buildClosedMaskKernel < 0)
            return null;

        int erodeMaskKernel;
        try { erodeMaskKernel = cs.FindKernel("DesignViewErodeMask1Px"); }
        catch { return null; }
        if (erodeMaskKernel < 0)
            return null;

        int accumulateFeatherRingKernel;
        try { accumulateFeatherRingKernel = cs.FindKernel("DesignViewAccumulateFeatherRing"); }
        catch { return null; }
        if (accumulateFeatherRingKernel < 0)
            return null;

        int finalizeFeatherMaskKernel;
        try { finalizeFeatherMaskKernel = cs.FindKernel("DesignViewFinalizeFeatherMask"); }
        catch { return null; }
        if (finalizeFeatherMaskKernel < 0)
            return null;

        int buildBlendMaskKernel = -1;
        if (_exportCompositeDebug)
        {
            try { buildBlendMaskKernel = cs.FindKernel("DesignViewBuildBlendMask"); }
            catch { buildBlendMaskKernel = -1; }
        }

        var width = baseTexture.width;
        var height = baseTexture.height;
        RenderTexture compositeRt = null;
        RenderTexture remainingMaskRt = null;
        RenderTexture tempRtA = null;
        RenderTexture tempRtB = null;
        RenderTexture debugBlendMaskRt = null;
        RenderTexture closedMaskRt = null;
        RenderTexture featherAccumRt = null;
        RenderTexture featherWorkRt = null;
        RenderTexture erodePingRt = null;
        RenderTexture erodePongRt = null;
        string debugDirectory = null;
        try
        {
            compositeRt = CreateWorkingRenderTexture(width, height, "DesignViewComposite");
            remainingMaskRt = CreateWorkingRenderTexture(width, height, "DesignViewRemainingMask");
            Graphics.Blit(baseTexture, compositeRt);
            Graphics.Blit(holeMask, remainingMaskRt);

            if (_exportCompositeDebug)
            {
                debugDirectory = CreateCompositeDebugDirectory();
                await DumpTextureAsync(_maskedBackgroundPreview != null ? (Texture)_maskedBackgroundPreview : baseTexture, debugDirectory, "00_base_background.png");
                await DumpTextureAsync(backgroundReference, debugDirectory, "01_background_reference.png");
                await DumpTextureAsync(holeMask, debugDirectory, "02_hole_mask.png");
                await DumpTextureAsync(baseTexture, debugDirectory, "03_composite_input_base.png");
            }

            var gx = Mathf.Max(1, Mathf.CeilToInt(width / 8f));
            var gy = Mathf.Max(1, Mathf.CeilToInt(height / 8f));
            for (var layerIndex = 0; layerIndex < layers.Count; layerIndex++)
            {
                var layer = layers[layerIndex];
                if (layer?.contentRenderTexture == null)
                    continue;
                if (layer.contentRenderTexture.width <= 0 || layer.contentRenderTexture.height <= 0)
                    continue;

                var targetRect = ToDisplayPixelRect(
                    new Rect(
                        layer.normalizedRect.x * width,
                        layer.normalizedRect.y * height,
                        layer.normalizedRect.width * width,
                        layer.normalizedRect.height * height),
                    width,
                    height);
                if (targetRect.width <= 0 || targetRect.height <= 0)
                    continue;

                tempRtA = CreateWorkingRenderTexture(width, height, "DesignViewCompositeTmpA");
                tempRtB = CreateWorkingRenderTexture(width, height, "DesignViewCompositeTmpB");
                closedMaskRt = CreateWorkingRenderTexture(targetRect.width, targetRect.height, "DesignViewClosedMask");
                featherAccumRt = CreateWorkingRenderTexture(targetRect.width, targetRect.height, "DesignViewFeatherAccum");
                featherWorkRt = CreateWorkingRenderTexture(targetRect.width, targetRect.height, "DesignViewFeatherWork");
                erodePingRt = CreateWorkingRenderTexture(targetRect.width, targetRect.height, "DesignViewFeatherErodeA");
                erodePongRt = CreateWorkingRenderTexture(targetRect.width, targetRect.height, "DesignViewFeatherErodeB");

                if (closedMaskRt == null || featherAccumRt == null || featherWorkRt == null || erodePingRt == null || erodePongRt == null)
                    continue;

                var prevActive = RenderTexture.active;
                RenderTexture.active = featherAccumRt;
                GL.Clear(false, true, Color.black);
                RenderTexture.active = prevActive;

                var layerGx = Mathf.Max(1, Mathf.CeilToInt(targetRect.width / 8f));
                var layerGy = Mathf.Max(1, Mathf.CeilToInt(targetRect.height / 8f));

                cs.SetTexture(buildClosedMaskKernel, "_Overlay", layer.contentRenderTexture);
                cs.SetTexture(buildClosedMaskKernel, "_Result", closedMaskRt);
                cs.SetInt("_DesignViewCloseRadius", Mathf.Clamp(edgeCloseRadius, 0, 8));
                cs.Dispatch(buildClosedMaskKernel, layerGx, layerGy, 1);

                Graphics.Blit(closedMaskRt, erodePingRt);

                var featherRadius = edgeFeatherRadius;
                if (featherRadius > 0)
                {
                    for (var ring = 1; ring <= featherRadius; ring++)
                    {
                        cs.SetTexture(erodeMaskKernel, "_Source", erodePingRt);
                        cs.SetTexture(erodeMaskKernel, "_Result", erodePongRt);
                        cs.Dispatch(erodeMaskKernel, layerGx, layerGy, 1);

                        cs.SetTexture(accumulateFeatherRingKernel, "_Source", erodePingRt);
                        cs.SetTexture(accumulateFeatherRingKernel, "_Overlay", erodePongRt);
                        cs.SetTexture(accumulateFeatherRingKernel, "_BackgroundRef", featherAccumRt);
                        cs.SetTexture(accumulateFeatherRingKernel, "_Result", featherWorkRt);
                        cs.SetFloat("_DesignViewRingWeight", ring / Mathf.Max(1f, featherRadius));
                        cs.Dispatch(accumulateFeatherRingKernel, layerGx, layerGy, 1);
                        Graphics.Blit(featherWorkRt, featherAccumRt);

                        Swap(ref erodePingRt, ref erodePongRt);
                    }
                }

                cs.SetTexture(finalizeFeatherMaskKernel, "_Source", featherAccumRt);
                cs.SetTexture(finalizeFeatherMaskKernel, "_Overlay", erodePingRt);
                cs.SetTexture(finalizeFeatherMaskKernel, "_Result", featherWorkRt);
                cs.Dispatch(finalizeFeatherMaskKernel, layerGx, layerGy, 1);
                Graphics.Blit(featherWorkRt, featherAccumRt);

                cs.SetTexture(kernel, "_Source", compositeRt);
                cs.SetTexture(kernel, "_BackgroundRef", backgroundReference);
                cs.SetTexture(kernel, "_Overlay", layer.contentRenderTexture);
                cs.SetTexture(kernel, "_DesignViewClosedMask", closedMaskRt);
                cs.SetTexture(kernel, "_DesignViewBlendMask", featherAccumRt);
                cs.SetTexture(kernel, "_FaceMaskIn", remainingMaskRt);
                cs.SetTexture(kernel, "_Result", tempRtA);
                cs.SetTexture(kernel, "_DesignViewRemainingMaskOut", tempRtB);
                cs.SetInts("_CropRect", targetRect.x, targetRect.y, targetRect.width, targetRect.height);
                cs.SetInts("_DesignViewLayerSize", layer.contentRenderTexture.width, layer.contentRenderTexture.height);
                cs.SetInts("_DesignViewCanvasSize", width, height);
                cs.SetInt("_DesignViewCloseRadius", Mathf.Clamp(edgeCloseRadius, 0, 8));
                cs.SetInt("_DesignViewFeatherRadius", edgeFeatherRadius);
                cs.SetFloat("_DesignViewPreserve", Mathf.Clamp01(edgePreserve));
                cs.Dispatch(kernel, gx, gy, 1);

                if (_exportCompositeDebug)
                {
                    if (buildBlendMaskKernel >= 0)
                    {
                        cs.SetTexture(buildBlendMaskKernel, "_Overlay", layer.contentRenderTexture);
                        cs.SetTexture(buildBlendMaskKernel, "_DesignViewClosedMask", closedMaskRt);
                        cs.SetTexture(buildBlendMaskKernel, "_DesignViewBlendMask", featherAccumRt);
                        cs.SetInts("_CropRect", targetRect.x, targetRect.y, targetRect.width, targetRect.height);
                        cs.SetInts("_DesignViewCanvasSize", width, height);
                        cs.SetInt("_DesignViewCloseRadius", Mathf.Clamp(edgeCloseRadius, 0, 8));
                        cs.SetInt("_DesignViewFeatherRadius", edgeFeatherRadius);
                        cs.SetFloat("_DesignViewPreserve", Mathf.Clamp01(edgePreserve));

                        debugBlendMaskRt = CreateWorkingRenderTexture(width, height, "DesignViewBlendMaskDebug");
                        cs.SetTexture(buildBlendMaskKernel, "_Result", debugBlendMaskRt);

                        cs.SetInt("_DesignViewDebugMaskMode", 0);
                        cs.Dispatch(buildBlendMaskKernel, gx, gy, 1);
                        await DumpTextureAsync(debugBlendMaskRt, debugDirectory, $"layer_{layerIndex:00}_mask_raw.png");

                        cs.SetInt("_DesignViewDebugMaskMode", 1);
                        cs.Dispatch(buildBlendMaskKernel, gx, gy, 1);
                        await DumpTextureAsync(debugBlendMaskRt, debugDirectory, $"layer_{layerIndex:00}_mask_closed.png");

                        cs.SetInt("_DesignViewDebugMaskMode", 2);
                        cs.Dispatch(buildBlendMaskKernel, gx, gy, 1);
                        await DumpTextureAsync(debugBlendMaskRt, debugDirectory, $"layer_{layerIndex:00}_mask_blurred.png");

                        cs.SetInt("_DesignViewDebugMaskMode", 3);
                        cs.Dispatch(buildBlendMaskKernel, gx, gy, 1);
                    }

                await DumpTextureAsync(layer.contentRenderTexture, debugDirectory, $"layer_{layerIndex:00}_foreground.png");
                await DumpTextureAsync(remainingMaskRt, debugDirectory, $"layer_{layerIndex:00}_mask_before.png");
                if (debugBlendMaskRt != null)
                    await DumpTextureAsync(debugBlendMaskRt, debugDirectory, $"layer_{layerIndex:00}_blend_mask.png");
                await DumpTextureAsync(tempRtA, debugDirectory, $"layer_{layerIndex:00}_composited.png");
                await DumpTextureAsync(tempRtB, debugDirectory, $"layer_{layerIndex:00}_mask_after.png");
                }

                Swap(ref compositeRt, ref tempRtA);
                Swap(ref remainingMaskRt, ref tempRtB);
                DestroyRenderTexture(ref tempRtA);
                DestroyRenderTexture(ref tempRtB);
                DestroyRenderTexture(ref debugBlendMaskRt);
                DestroyRenderTexture(ref closedMaskRt);
                DestroyRenderTexture(ref featherAccumRt);
                DestroyRenderTexture(ref featherWorkRt);
                DestroyRenderTexture(ref erodePingRt);
                DestroyRenderTexture(ref erodePongRt);
            }

            if (_exportCompositeDebug)
            {
                await DumpTextureAsync(compositeRt, debugDirectory, "90_after_deepfill_background_composite.png");
                await DumpTextureAsync(remainingMaskRt, debugDirectory, "91_remaining_mask_rt.png");
            }

            var composedTexture = await ReadbackTextureAsync(compositeRt, width, height);
            var remainingMaskTex = await ReadbackTextureAsync(remainingMaskRt, width, height);
            if (composedTexture == null)
                return null;

            var remainingMaskPixels = await CountMaskPixelsAsync(remainingMaskRt);
            if (composedTexture != null)
                composedTexture.name = "DesignViewAppliedComposite";
            if (remainingMaskTex != null)
                remainingMaskTex.name = "DesignViewRemainingHoleMask";

            return new ApplyCompositeResult
            {
                composedTexture = composedTexture,
                remainingMask = remainingMaskTex,
                remainingMaskPixels = remainingMaskPixels,
                debugDirectory = debugDirectory
            };
        }
        finally
        {
            DestroyRenderTexture(ref compositeRt);
            DestroyRenderTexture(ref remainingMaskRt);
            DestroyRenderTexture(ref tempRtA);
            DestroyRenderTexture(ref tempRtB);
            DestroyRenderTexture(ref debugBlendMaskRt);
            DestroyRenderTexture(ref closedMaskRt);
            DestroyRenderTexture(ref featherAccumRt);
            DestroyRenderTexture(ref featherWorkRt);
            DestroyRenderTexture(ref erodePingRt);
            DestroyRenderTexture(ref erodePongRt);
        }
    }

    private static RenderTexture CreateWorkingRenderTexture(int width, int height, string name)
    {
        if (width <= 0 || height <= 0)
            return null;

        var rt = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
        {
            enableRandomWrite = true,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
            name = name
        };
        rt.Create();
        return rt;
    }

    private static void Swap(ref RenderTexture a, ref RenderTexture b)
    {
        var t = a;
        a = b;
        b = t;
    }

    private static string CreateCompositeDebugDirectory()
    {
        var root = Path.Combine(Application.dataPath, "..", "Logs", "DesignViewCompositeDebug");
        Directory.CreateDirectory(root);
        var dir = Path.Combine(root, DateTime.Now.ToString("yyyyMMdd-HHmmss-fff", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private async UniTask DumpTextureAsync(Texture texture, string directory, string fileName)
    {
        if (texture == null || string.IsNullOrWhiteSpace(directory) || string.IsNullOrWhiteSpace(fileName))
            return;

        Texture2D output = null;
        var shouldDestroy = false;
        try
        {
            if (texture is Texture2D tex2D)
            {
                output = tex2D;
            }
            else if (texture is RenderTexture rt)
            {
                output = await ReadbackTextureAsync(rt, rt.width, rt.height);
                shouldDestroy = output != null;
            }

            if (output == null)
                return;

            Directory.CreateDirectory(directory);
            File.WriteAllBytes(Path.Combine(directory, fileName), output.EncodeToPNG());
        }
        catch
        {
        }
        finally
        {
            if (shouldDestroy && output != null)
                Destroy(output);
        }
    }

    private async UniTask<int> CountMaskPixelsAsync(RenderTexture maskTexture)
    {
        if (maskTexture == null)
            return 0;

        if (Application.isBatchMode || !SystemInfo.supportsAsyncGPUReadback)
        {
            var syncTexture = ReadbackTextureSync(maskTexture, maskTexture.width, maskTexture.height);
            if (syncTexture == null)
                return 0;

            try
            {
                var pixels = syncTexture.GetPixels32();
                var syncCount = 0;
                for (var i = 0; i < pixels.Length; i++)
                {
                    if (IsMaskedPixel(pixels[i]))
                        syncCount++;
                }

                return syncCount;
            }
            finally
            {
                Destroy(syncTexture);
            }
        }

        var tcs = new UniTaskCompletionSource<AsyncGPUReadbackRequest>();
        AsyncGPUReadback.Request(maskTexture, 0, TextureFormat.RGBA32, req => tcs.TrySetResult(req));
        var request = await tcs.Task;
        if (request.hasError)
            return 0;

        var data = request.GetData<Color32>();
        var count = 0;
        for (var i = 0; i < data.Length; i++)
        {
            if (IsMaskedPixel(data[i]))
                count++;
        }

        return count;
    }
}
