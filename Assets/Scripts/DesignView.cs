using System;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;
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
        public Texture2D previewTexture;
    }

    private sealed class ApplyCompositeResult
    {
        public Texture2D composedTexture;
        public Texture2D remainingMask;
        public int remainingMaskPixels;
    }

    public override AppPageId PageId => AppPageId.DesignView;

    private readonly List<LayerBoxData> _layerData = new List<LayerBoxData>();
    private readonly List<VisualElement> _layerElements = new List<VisualElement>();
    private VisualElement _canvasOverlay;
    private VisualElement _tipsPanel;
    private Label _tipsLabel;
    private Button _applyButton;
    private Button _detectButton;
    private Texture2D _maskedBackgroundPreview;
    private Texture2D _maskedBackgroundHoleMask;
    private int _edgeCloseRadius = DefaultEdgeCloseRadius;
    private int _edgeFeatherRadius = DefaultEdgeFeatherRadius;
    private float _edgePreserve = DefaultEdgePreserve;

    protected override AppPageId? ResolveSwipeTarget(SwipeDirection direction)
    {
        return direction == SwipeDirection.Left ? AppPageId.MainView2 : null;
    }

    protected override bool UseOverlaySwitchZone => true;

    protected override float GetSwitchPillAlignment01() => 1f;

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

    protected override void BuildPage(VisualElement contentRoot)
    {
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

        canvasHost.Add(CreateFloatingHistoryPanel(230f, "设计历史"));

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

        var tipsTitle = new Label("设计说明");
        tipsTitle.style.color = Color.white;
        tipsTitle.style.unityFontStyleAndWeight = FontStyle.Bold;
        _tipsPanel.Add(tipsTitle);

        _tipsLabel = new Label("点击“识别图层”后，会基于 YOLO Seg 结果生成可拖动、可缩放的图层。识别出的人物或对象会放进图层框里，被切走的背景区域会变黑。");
        _tipsLabel.style.color = new Color(0.82f, 0.86f, 0.92f, 1f);
        _tipsLabel.style.whiteSpace = WhiteSpace.Normal;
        _tipsLabel.style.marginTop = 4;
        _tipsPanel.Add(_tipsLabel);

        _tipsPanel.Add(BuildBlendControls());

        BuildStandardOverlays();
    }

    protected override void OnLayoutChanged(bool isPortrait, Rect layoutRect)
    {
        RebuildLayerBoxes();
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
            CompareView.SetSources(
                _maskedBackgroundPreview,
                original ?? current ?? _maskedBackgroundPreview,
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

        var title = new Label("设计");
        title.style.fontSize = 18;
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        title.style.marginRight = 12;
        bar.Add(title);

        _detectButton = CreateActionButton("识别图层", OnDetectLayers);
        bar.Add(_detectButton);

        _applyButton = CreateActionButton("应用生成", OnApplyDesign);
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

        var title = new Label("边缘融合");
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        host.Add(title);

        var desc = new Label("默认会对人物边缘做轻量形态学闭运算和 alpha 羽化，适合发丝、半透明边缘和轻微锯齿。");
        desc.style.color = new Color(0.76f, 0.82f, 0.9f, 1f);
        desc.style.whiteSpace = WhiteSpace.Normal;
        desc.style.marginTop = 4;
        host.Add(desc);

        host.Add(CreateBlendSliderRow("闭运算", 0, 4, _edgeCloseRadius, "px", v => _edgeCloseRadius = Mathf.RoundToInt(v)));
        host.Add(CreateBlendSliderRow("羽化", 0, 8, _edgeFeatherRadius, "px", v => _edgeFeatherRadius = Mathf.RoundToInt(v)));
        host.Add(CreateBlendSliderRow("边缘保真", 0f, 1f, _edgePreserve, "", v => _edgePreserve = Mathf.Clamp01(v), "0.00"));

        return host;
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

        ShowProgress("识别图层");
        YoloSegResult result = default;
        List<LayerBoxData> detectedLayers = null;
        Texture2D maskedBackgroundPreview = null;
        Texture2D maskedBackgroundHoleMask = null;
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

            detectedLayers = BuildLayerDataFromDetection(src, result, out maskedBackgroundPreview, out maskedBackgroundHoleMask);
            if (detectedLayers.Count == 0)
            {
                ShowToast("未检测到可用的人物或对象图层", 2600);
                return;
            }

            ReplaceDetectedLayerState(detectedLayers, maskedBackgroundPreview, maskedBackgroundHoleMask);
            detectedLayers = null;
            maskedBackgroundPreview = null;
            maskedBackgroundHoleMask = null;

            if (_tipsLabel != null)
                _tipsLabel.text = $"已生成 {_layerData.Count} 个图层。图层里是切出的人物或对象，背景对应区域已变黑，移动或缩放图层时内容会一起跟随。";
        }
        finally
        {
            DestroyLayerTextures(detectedLayers);
            DestroyTexture(ref maskedBackgroundPreview);
            DestroyTexture(ref maskedBackgroundHoleMask);
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
            _tipsLabel.text = "点击“识别图层”后，会基于 YOLO Seg 结果生成可拖动、可缩放的图层。识别出的人物或对象会放进图层框里，被切走的背景区域会变黑。";
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
            contentImage.scaleMode = ScaleMode.ScaleToFit;
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

    private void ReplaceDetectedLayerState(List<LayerBoxData> layers, Texture2D maskedBackgroundPreview, Texture2D maskedBackgroundHoleMask)
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
        DestroyTexture(ref _maskedBackgroundPreview);
        DestroyTexture(ref _maskedBackgroundHoleMask);

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

    private static List<LayerBoxData> BuildLayerDataFromDetection(
        Texture2D source,
        YoloSegResult result,
        out Texture2D maskedBackgroundPreview,
        out Texture2D maskedBackgroundHoleMask)
    {
        maskedBackgroundPreview = BuildMaskedBackgroundTexture(source, result.mask);
        maskedBackgroundHoleMask = BuildHoleMaskTexture(result.mask);
        var layers = new List<LayerBoxData>();
        if (source == null || result.mask == null || result.detections == null || result.detections.Length == 0)
            return layers;

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
            var contentTexture = BuildLayerCutoutTexture(source, result.mask, textureRect, i + 1);
            var contentRt = CreateRenderTextureFromTexture(contentTexture);

            layers.Add(new LayerBoxData
            {
                title = $"Layer {i + 1}  {(detection.probability * 100f):0}%",
                normalizedRect = normalized,
                color = Color.HSVToRGB((i * 0.17f) % 1f, 0.68f, 1f),
                previewTexture = contentTexture,
                contentRenderTexture = contentRt
            });
        }

        return layers;
    }

    private static Texture2D BuildMaskedBackgroundTexture(Texture2D source, Texture2D mask)
    {
        if (!TryGetPixels(source, out var sourcePixels) || !TryGetPixels(mask, out var maskPixels) || sourcePixels.Length != maskPixels.Length)
            return null;

        var pixels = new Color32[sourcePixels.Length];
        for (var i = 0; i < pixels.Length; i++)
        {
            pixels[i] = IsMaskedPixel(maskPixels[i])
                ? new Color32(0, 0, 0, 255)
                : new Color32(sourcePixels[i].r, sourcePixels[i].g, sourcePixels[i].b, 255);
        }

        return CreateTextureFromPixels(pixels, source.width, source.height, "DesignViewMaskedBackground");
    }

    private static Texture2D BuildHoleMaskTexture(Texture2D mask)
    {
        if (!TryGetPixels(mask, out var maskPixels))
            return null;

        var pixels = new Color32[maskPixels.Length];
        for (var i = 0; i < pixels.Length; i++)
        {
            pixels[i] = IsMaskedPixel(maskPixels[i])
                ? new Color32(255, 255, 255, 255)
                : new Color32(0, 0, 0, 255);
        }

        return CreateTextureFromPixels(pixels, mask.width, mask.height, "DesignViewHoleMask");
    }

    private static Texture2D BuildLayerCutoutTexture(Texture2D source, Texture2D mask, RectInt pixelRect, int layerIndex)
    {
        if (pixelRect.width <= 0 || pixelRect.height <= 0)
            return null;
        if (!TryGetPixels(source, out var sourcePixels) || !TryGetPixels(mask, out var maskPixels) || sourcePixels.Length != maskPixels.Length)
            return null;

        var pixels = new Color32[pixelRect.width * pixelRect.height];
        var visiblePixels = 0;
        for (var y = 0; y < pixelRect.height; y++)
        {
            var srcRow = (pixelRect.y + y) * source.width;
            var dstRow = y * pixelRect.width;
            for (var x = 0; x < pixelRect.width; x++)
            {
                var srcIndex = srcRow + pixelRect.x + x;
                if (IsMaskedPixel(maskPixels[srcIndex]))
                {
                    var p = sourcePixels[srcIndex];
                    p.a = 255;
                    pixels[dstRow + x] = p;
                    visiblePixels++;
                }
                else
                {
                    pixels[dstRow + x] = new Color32(0, 0, 0, 0);
                }
            }
        }

        if (visiblePixels == 0)
            return null;

        return CreateTextureFromPixels(pixels, pixelRect.width, pixelRect.height, $"DesignViewLayerCutout{layerIndex}");
    }

    private static Texture2D CreateTextureFromPixels(Color32[] pixels, int width, int height, string name)
    {
        var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
        texture.SetPixels32(pixels);
        texture.Apply(false, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Bilinear;
        texture.name = name;
        return texture;
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

    private static bool TryGetPixels(Texture2D texture, out Color32[] pixels)
    {
        pixels = null;
        if (texture == null)
            return false;

        try
        {
            pixels = texture.GetPixels32();
        }
        catch
        {
            pixels = null;
            return false;
        }

        return pixels != null && pixels.Length > 0;
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
        if (originalHistory == null)
        {
            ShowToast("找不到原始背景图", 2400);
            return;
        }

        ShowBusy("应用生成");
        if (_maskedBackgroundPreview == null || _maskedBackgroundHoleMask == null)
        {
            ShowToast("缺少背景或遮罩数据，请先重新识别图层", 2600);
            return;
        }

        ApplyCompositeResult applyResult = null;
        try
        {
            applyResult = await BuildAppliedCompositeAsync(
                _maskedBackgroundPreview,
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

            if (_tipsLabel != null)
            {
                _tipsLabel.text = applyResult.remainingMaskPixels > 0
                    ? $"已生成新的设计结果，剩余 {applyResult.remainingMaskPixels} 个待补区域像素，SD inpainting 接口已预留但暂未调用。"
                    : "已生成新的设计结果，当前图层已全部合成回背景。";
            }

            ShowToast(
                applyResult.remainingMaskPixels > 0
                    ? "已合成到背景，仍有黑色缺口，后续可从预留的 SD inpainting 接口补洞。"
                    : "已合成到背景，并已清除所有图层。",
                3200);
        }
        finally
        {
            if (applyResult?.composedTexture != null)
                DestroyTexture(ref applyResult.composedTexture);
            if (applyResult?.remainingMask != null)
                DestroyTexture(ref applyResult.remainingMask);
            HideBusy();
            await UniTask.Yield();
        }
    }

    private async UniTask<ApplyCompositeResult> BuildAppliedCompositeAsync(
        Texture2D maskedBackground,
        Texture2D holeMask,
        List<LayerBoxData> layers,
        int edgeCloseRadius,
        int edgeFeatherRadius,
        float edgePreserve)
    {
        if (maskedBackground == null || holeMask == null || layers == null || layers.Count == 0)
            return null;

        var cs = Host?.ImageProcessingCS;
        if (cs == null)
            return null;

        int kernel;
        try { kernel = cs.FindKernel("DesignViewCompositeLayer"); }
        catch { return null; }
        if (kernel < 0)
            return null;

        var width = maskedBackground.width;
        var height = maskedBackground.height;
        RenderTexture compositeRt = null;
        RenderTexture remainingMaskRt = null;
        RenderTexture tempRtA = null;
        RenderTexture tempRtB = null;
        try
        {
            compositeRt = CreateWorkingRenderTexture(width, height, "DesignViewComposite");
            remainingMaskRt = CreateWorkingRenderTexture(width, height, "DesignViewRemainingMask");
            Graphics.Blit(maskedBackground, compositeRt);
            Graphics.Blit(holeMask, remainingMaskRt);

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

                cs.SetTexture(kernel, "_Source", compositeRt);
                cs.SetTexture(kernel, "_Overlay", layer.contentRenderTexture);
                cs.SetTexture(kernel, "_FaceMaskIn", remainingMaskRt);
                cs.SetTexture(kernel, "_Result", tempRtA);
                cs.SetTexture(kernel, "_DesignViewRemainingMaskOut", tempRtB);
                cs.SetInts("_CropRect", targetRect.x, targetRect.y, targetRect.width, targetRect.height);
                cs.SetInts("_DesignViewLayerSize", layer.contentRenderTexture.width, layer.contentRenderTexture.height);
                cs.SetInts("_DesignViewCanvasSize", width, height);
                cs.SetInt("_DesignViewCloseRadius", Mathf.Clamp(edgeCloseRadius, 0, 8));
                cs.SetInt("_DesignViewFeatherRadius", Mathf.Clamp(edgeFeatherRadius, 0, 24));
                cs.SetFloat("_DesignViewPreserve", Mathf.Clamp01(edgePreserve));
                cs.Dispatch(kernel, gx, gy, 1);

                Swap(ref compositeRt, ref tempRtA);
                Swap(ref remainingMaskRt, ref tempRtB);
                DestroyRenderTexture(ref tempRtA);
                DestroyRenderTexture(ref tempRtB);
            }

            var composedTexture = await ReadbackTextureAsync(compositeRt, width, height);
            var remainingMaskTex = await ReadbackTextureAsync(remainingMaskRt, width, height);
            if (composedTexture == null)
                return null;

            var remainingMaskPixels = CountMaskPixels(remainingMaskTex);
            if (composedTexture != null)
                composedTexture.name = "DesignViewAppliedComposite";
            if (remainingMaskTex != null)
                remainingMaskTex.name = "DesignViewRemainingHoleMask";

            return new ApplyCompositeResult
            {
                composedTexture = composedTexture,
                remainingMask = remainingMaskTex,
                remainingMaskPixels = remainingMaskPixels
            };
        }
        finally
        {
            DestroyRenderTexture(ref compositeRt);
            DestroyRenderTexture(ref remainingMaskRt);
            DestroyRenderTexture(ref tempRtA);
            DestroyRenderTexture(ref tempRtB);
        }
    }

    private static Texture2D BuildRemainingHoleMaskTexture(bool[] remainingHole, int width, int height, out int remainingMaskPixels)
    {
        remainingMaskPixels = 0;
        if (remainingHole == null || remainingHole.Length != width * height)
            return null;

        var pixels = new Color32[remainingHole.Length];
        for (var i = 0; i < remainingHole.Length; i++)
        {
            if (remainingHole[i])
            {
                pixels[i] = new Color32(255, 255, 255, 255);
                remainingMaskPixels++;
            }
            else
            {
                pixels[i] = new Color32(0, 0, 0, 255);
            }
        }

        return CreateTextureFromPixels(pixels, width, height, "DesignViewRemainingHoleMask");
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

    private static int CountMaskPixels(Texture2D maskTexture)
    {
        if (!TryGetPixels(maskTexture, out var pixels))
            return 0;

        var count = 0;
        for (var i = 0; i < pixels.Length; i++)
        {
            if (IsMaskedPixel(pixels[i]))
                count++;
        }

        return count;
    }
}
