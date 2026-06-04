using System;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

public sealed class DesignView : BasePageView
{
    private sealed class LayerBoxData
    {
        public string title;
        public Rect normalizedRect;
        public Color color;
    }

    public override AppPageId PageId => AppPageId.DesignView;

    private readonly List<LayerBoxData> _layerData = new List<LayerBoxData>();
    private readonly List<VisualElement> _layerElements = new List<VisualElement>();
    private VisualElement _canvasOverlay;
    private VisualElement _tipsPanel;
    private Label _tipsLabel;
    private Button _applyButton;
    private Button _detectButton;

    protected override AppPageId? ResolveSwipeTarget(SwipeDirection direction)
    {
        return direction == SwipeDirection.Left ? AppPageId.MainView2 : null;
    }

    protected override void OnShown()
    {
        if (CompareView != null)
        {
            CompareView.ViewTransformChanged -= OnCompareViewTransformChanged;
            CompareView.ViewTransformChanged += OnCompareViewTransformChanged;
        }

        var current = GetCurrentHistoryTexture();
        var original = GetOriginalHistoryTexture();
        if (current != null || original != null)
        {
            CompareView?.SetSources(current ?? original, original ?? current, GetCurrentHistoryLabel());
            CompareView?.FitToView();
            RebuildLayerBoxes();
        }
    }

    protected override void OnBeforeDetach()
    {
        if (CompareView != null)
            CompareView.ViewTransformChanged -= OnCompareViewTransformChanged;
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
        _canvasOverlay.pickingMode = PickingMode.Position;
        canvasHost.Add(_canvasOverlay);

        canvasHost.Add(CreateFloatingHistoryPanel(230f, "设计历史"));

        _tipsPanel = new VisualElement();
        _tipsPanel.style.position = Position.Absolute;
        _tipsPanel.style.right = 18;
        _tipsPanel.style.top = 18;
        _tipsPanel.style.width = 260;
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

        _tipsLabel = new Label("点击“识别图层”后，会基于 YOLO Seg 结果生成可拖动、可缩放的图层框。应用按钮先保留接口，后续接入 SD inpainting。");
        _tipsLabel.style.color = new Color(0.82f, 0.86f, 0.92f, 1f);
        _tipsLabel.style.whiteSpace = WhiteSpace.Normal;
        _tipsLabel.style.marginTop = 4;
        _tipsPanel.Add(_tipsLabel);

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
            _tipsPanel.style.width = 260;
        }
    }

    public void SyncFromMainView(string path, Texture2D currentTexture, Texture2D originalTexture, string label)
    {
        if (currentTexture == null && originalTexture == null)
            return;

        SetHistoryFromSharedTextures(originalTexture, currentTexture, label, path);
        CompareView?.SetSources(currentTexture ?? originalTexture, originalTexture ?? currentTexture, label);
        CompareView?.FitToView();
        RebuildLayerBoxes();
    }

    private void OnCompareViewTransformChanged()
    {
        RebuildLayerBoxes();
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
        try
        {
            var oldTargetPersonOnly = Host.YoloSegRunner.targetPersonOnly;
            Host.YoloSegRunner.targetPersonOnly = false;
            void OnProgress(float p, string t) => SetProgress(p, t);
            Host.YoloSegRunner.ProgressChanged -= OnProgress;
            Host.YoloSegRunner.ProgressChanged += OnProgress;
            var result = await Host.YoloSegRunner.ProcessAsync(src, default);
            Host.YoloSegRunner.ProgressChanged -= OnProgress;
            Host.YoloSegRunner.targetPersonOnly = oldTargetPersonOnly;

            if (!string.IsNullOrWhiteSpace(result.error))
            {
                ShowToast(result.error, 3000);
                return;
            }

            _layerData.Clear();
            if (result.detections != null && result.detections.Length > 0)
            {
                for (var i = 0; i < result.detections.Length; i++)
                {
                    var detection = result.detections[i];
                    var rect = detection.rect;
                    var normalized = new Rect(
                        Mathf.Clamp01(rect.x / Mathf.Max(1f, src.width)),
                        Mathf.Clamp01(rect.y / Mathf.Max(1f, src.height)),
                        Mathf.Clamp01(rect.width / Mathf.Max(1f, src.width)),
                        Mathf.Clamp01(rect.height / Mathf.Max(1f, src.height)));

                    if (normalized.width < 0.03f || normalized.height < 0.03f)
                        continue;

                    _layerData.Add(new LayerBoxData
                    {
                        title = $"Layer {i + 1}  {(detection.probability * 100f):0}%",
                        normalizedRect = normalized,
                        color = Color.HSVToRGB((i * 0.17f) % 1f, 0.68f, 1f)
                    });
                }
            }

            if (_layerData.Count == 0)
            {
                _layerData.Add(new LayerBoxData
                {
                    title = "Layer 1",
                    normalizedRect = new Rect(0.26f, 0.20f, 0.30f, 0.36f),
                    color = new Color(0.31f, 0.78f, 1f, 1f)
                });
            }

            RebuildLayerBoxes();
            _tipsLabel.text = $"已生成 {_layerData.Count} 个图层框，可拖动、缩放并保留位置。后续会在这里接入背景重算和 inpainting。";
        }
        finally
        {
            HideProgress();
        }
    }

    private void OnApplyDesign()
    {
        ShowToast("应用接口已预留：后续会在这里重新计算背景 mask 并调用 SD inpainting 生成整图。", 3600);
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
            Mathf.Max(48f, data.normalizedRect.width * imageRect.width),
            Mathf.Max(48f, data.normalizedRect.height * imageRect.height));

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

        var header = new VisualElement();
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

        rect.width = Mathf.Max(48f, rect.width);
        rect.height = Mathf.Max(48f, rect.height);
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
}
