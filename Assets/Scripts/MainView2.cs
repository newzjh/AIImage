using System;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

/// <summary>
/// MainView2 - 主视图页面，支持横竖版布局
/// 横版：带右侧调节面板
/// 竖版：调节面板折叠
/// </summary>
public class MainView2 : BasePageView
{
    private VisualElement _contentContainer;
    private VisualElement _imageViewerContainer;
    private VisualElement _adjustmentPanel;
    private VisualElement _topButtonBar;
    private VisualElement _bottomPresetBar;
    private VisualElement _toastOverlay;
    private Button _collapsePanelBtn;

    private SplitCompareView _imageViewer; // 使用完整的SplitCompareView
    private List<HistoryEntry> _historyEntries = new List<HistoryEntry>();
    private ListView _historyList;
    private long _historyOpSeq;

    private Texture2D _currentImage;
    private string _currentImagePath;

    public override void BuildPage()
    {
        _pageContainer = new VisualElement();
        _pageContainer.style.width = Length.Percent(100);
        _pageContainer.style.height = Length.Percent(100);
        _pageContainer.style.position = Position.Relative;
        _pageContainer.style.flexDirection = FlexDirection.Column;

        // 顶部按钮栏
        BuildTopButtonBar();

        // 内容区域
        _contentContainer = new VisualElement();
        _contentContainer.style.flexGrow = 1;
        _contentContainer.style.flexDirection = FlexDirection.Row;
        _contentContainer.style.minHeight = 0;
        _pageContainer.Add(_contentContainer);

        // 图像查看器容器
        _imageViewerContainer = new VisualElement();
        _imageViewerContainer.style.flexGrow = 1;
        _imageViewerContainer.style.position = Position.Relative;
        _contentContainer.Add(_imageViewerContainer);

        // 创建ImageViewer（带分割线的before/after视图）
        BuildImageViewer();

        // 底部预设按钮栏
        BuildBottomPresetBar();

        // 根据横竖屏决定是否显示右侧调节面板
        if (IsLandscape())
        {
            BuildAdjustmentPanel(false); // 横屏，面板在右侧
        }
        else
        {
            BuildAdjustmentPanel(true); // 竖屏，浮动面板默认折叠
        }

        // Toast提示
        _toastOverlay = BuildToast();
        _pageContainer.Add(_toastOverlay);

        // 页面切换指示器
        BuildPageIndicator(PageType.MainView2);
    }

    private void BuildTopButtonBar()
    {
        _topButtonBar = new VisualElement();
        _topButtonBar.style.flexDirection = FlexDirection.Row;
        _topButtonBar.style.flexWrap = Wrap.Wrap;
        _topButtonBar.style.alignItems = Align.Center;
        _topButtonBar.style.paddingLeft = 8;
        _topButtonBar.style.paddingRight = 8;
        _topButtonBar.style.paddingTop = 6;
        _topButtonBar.style.paddingBottom = 6;
        _topButtonBar.style.backgroundColor = new StyleColor(new Color(0.15f, 0.15f, 0.15f, 1f));
        _pageContainer.Add(_topButtonBar);

        // 参考MainView.cs中的按钮，创建图标按钮
        AddIconButton(_topButtonBar, "换脸", OnFaceSwap);
        AddIconButton(_topButtonBar, "清晰", OnSharpen);
        AddIconButton(_topButtonBar, "美白", OnWhiten);
        AddIconButton(_topButtonBar, "清晰&美白", OnSharpenWhiten);
        AddIconButton(_topButtonBar, "换背景", OnChangeBackground);
        AddIconButton(_topButtonBar, "去霾&调色", OnDehazeColorGrade);
        AddIconButton(_topButtonBar, "调色", OnColorGrade);
        AddIconButton(_topButtonBar, "去霾", OnDehaze);
        AddIconButton(_topButtonBar, "GPU清晰", OnGpuSharpen);
        AddIconButton(_topButtonBar, "ESRGAN", OnRealEsrgan);
        AddIconButton(_topButtonBar, "保存", OnSave);
    }

    private void AddIconButton(VisualElement parent, string text, Action onClick)
    {
        var btn = new Button(onClick) { text = text };
        btn.style.marginRight = 4;
        btn.style.marginBottom = 4;
        btn.style.paddingLeft = 12;
        btn.style.paddingRight = 12;
        btn.style.paddingTop = 6;
        btn.style.paddingBottom = 6;
        btn.style.backgroundColor = new StyleColor(new Color(0.25f, 0.25f, 0.25f, 1f));
        btn.style.color = Color.white;
        btn.style.borderTopLeftRadius = 4;
        btn.style.borderTopRightRadius = 4;
        btn.style.borderBottomLeftRadius = 4;
        btn.style.borderBottomRightRadius = 4;
        parent.Add(btn);
    }

    private void BuildImageViewer()
    {
        _imageViewer = new SplitCompareView(); // 使用完整的SplitCompareView
        _imageViewer.style.flexGrow = 1;
        _imageViewer.style.minHeight = 0;
        _imageViewerContainer.Add(_imageViewer);

        // 历史记录面板（浮动在图像查看器上方）
        BuildHistoryPanel();
    }

    private void BuildHistoryPanel()
    {
        var historyPanel = new VisualElement();
        historyPanel.name = "history-panel";
        historyPanel.style.position = Position.Absolute;
        historyPanel.style.left = 10;
        historyPanel.style.top = 10;
        historyPanel.style.width = 200;
        historyPanel.style.maxHeight = 300;
        historyPanel.style.backgroundColor = new StyleColor(new Color(0.1f, 0.1f, 0.1f, 0.9f));
        historyPanel.style.borderTopLeftRadius = 8;
        historyPanel.style.borderTopRightRadius = 8;
        historyPanel.style.borderBottomLeftRadius = 8;
        historyPanel.style.borderBottomRightRadius = 8;
        historyPanel.style.paddingLeft = 8;
        historyPanel.style.paddingRight = 8;
        historyPanel.style.paddingTop = 8;
        historyPanel.style.paddingBottom = 8;
        historyPanel.style.flexDirection = FlexDirection.Column;

        var header = new VisualElement();
        header.name = "drag-header";
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.marginBottom = 6;
        header.style.backgroundColor = new Color(0.2f, 0.2f, 0.2f);
        header.style.paddingLeft = 4;
        header.style.paddingRight = 4;
        header.style.height = 24;
        historyPanel.Add(header);

        var title = new Label("历史记录");
        title.style.flexGrow = 1;
        title.style.color = Color.white;
        header.Add(title);

        Button collapseBtn = null;
        collapseBtn = new Button(() =>
        {
            _historyList.style.display = _historyList.style.display == DisplayStyle.None ? DisplayStyle.Flex : DisplayStyle.None;
            if (collapseBtn != null)
                collapseBtn.text = _historyList.style.display == DisplayStyle.None ? "+" : "-";
        })
        { text = "-" };
        collapseBtn.style.width = 24;
        collapseBtn.style.height = 24;
        header.Add(collapseBtn);

        _historyList = new ListView();
        _historyList.style.flexGrow = 1;
        _historyList.style.minHeight = 0;
        _historyList.virtualizationMethod = CollectionVirtualizationMethod.FixedHeight;
        _historyList.fixedItemHeight = 24;
        _historyList.showBorder = true;
        _historyList.selectionType = SelectionType.Single;
        _historyList.itemsSource = _historyEntries;
        _historyList.makeItem = () =>
        {
            var label = new Label();
            label.style.color = Color.white;
            label.style.unityTextAlign = TextAnchor.MiddleLeft;
            label.style.whiteSpace = WhiteSpace.NoWrap;
            label.style.overflow = Overflow.Hidden;
            label.style.textOverflow = TextOverflow.Ellipsis;
            return label;
        };
        _historyList.bindItem = (element, index) =>
        {
            var label = (Label)element;
            label.text = _historyEntries[index].label;
        };
        _historyList.selectionChanged += OnHistorySelectionChanged;
        historyPanel.Add(_historyList);

        // 添加拖拽功能
        SetupPanelDragging(historyPanel, header);

        _imageViewerContainer.Add(historyPanel);
    }

    private void SetupPanelDragging(VisualElement panel, VisualElement dragHandle)
    {
        bool dragging = false;
        int dragPointerId = -1;
        Vector2 startPointer = default;
        Vector2 startPos = default;

        dragHandle.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (evt.button != 0) return;
            dragging = true;
            dragPointerId = evt.pointerId;
            startPointer = evt.position;
            startPos = new Vector2(panel.resolvedStyle.left, panel.resolvedStyle.top);
            dragHandle.CapturePointer(dragPointerId);
            evt.StopPropagation();
        });

        dragHandle.RegisterCallback<PointerMoveEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId || !dragHandle.HasPointerCapture(dragPointerId))
                return;

            var delta = (Vector2)evt.position - startPointer;
            var newPos = startPos + delta;

            var bounds = _imageViewerContainer.contentRect;
            var pw = panel.resolvedStyle.width;
            var ph = panel.resolvedStyle.height;

            var maxX = Mathf.Max(0f, bounds.width - pw);
            var maxY = Mathf.Max(0f, bounds.height - 30); // 至少保留标题栏可见

            newPos.x = Mathf.Clamp(newPos.x, 0f, maxX);
            newPos.y = Mathf.Clamp(newPos.y, 0f, maxY);

            panel.style.left = newPos.x;
            panel.style.top = newPos.y;
            evt.StopPropagation();
        });

        dragHandle.RegisterCallback<PointerUpEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId) return;
            dragging = false;
            if (dragHandle.HasPointerCapture(dragPointerId))
                dragHandle.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });

        dragHandle.RegisterCallback<PointerCancelEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId) return;
            dragging = false;
            if (dragHandle.HasPointerCapture(dragPointerId))
                dragHandle.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });
    }

    private void BuildBottomPresetBar()
    {
        _bottomPresetBar = new VisualElement();
        _bottomPresetBar.style.flexDirection = FlexDirection.Row;
        _bottomPresetBar.style.flexWrap = Wrap.Wrap;
        _bottomPresetBar.style.alignItems = Align.Center;
        _bottomPresetBar.style.justifyContent = Justify.Center;
        _bottomPresetBar.style.paddingLeft = 8;
        _bottomPresetBar.style.paddingRight = 8;
        _bottomPresetBar.style.paddingTop = 6;
        _bottomPresetBar.style.paddingBottom = 6; // 减少底部padding，为滑块留空间
        _bottomPresetBar.style.backgroundColor = new StyleColor(new Color(0.15f, 0.15f, 0.15f, 0.8f));
        _pageContainer.Add(_bottomPresetBar);

        // 预设风格化按钮
        AddPresetButton(_bottomPresetBar, "鲜艳", () => ApplyPreset("鲜艳"));
        AddPresetButton(_bottomPresetBar, "柔和", () => ApplyPreset("柔和"));
        AddPresetButton(_bottomPresetBar, "黑白", () => ApplyPreset("黑白"));
        AddPresetButton(_bottomPresetBar, "复古", () => ApplyPreset("复古"));
        AddPresetButton(_bottomPresetBar, "冷色", () => ApplyPreset("冷色"));
        AddPresetButton(_bottomPresetBar, "暖色", () => ApplyPreset("暖色"));
    }

    private void AddPresetButton(VisualElement parent, string text, Action onClick)
    {
        var btn = new Button(onClick) { text = text };
        btn.style.width = 70;
        btn.style.height = 70;
        btn.style.marginLeft = 6;
        btn.style.marginRight = 6;
        btn.style.backgroundColor = new StyleColor(new Color(0.3f, 0.3f, 0.3f, 1f));
        btn.style.color = Color.white;
        btn.style.borderTopLeftRadius = 8;
        btn.style.borderTopRightRadius = 8;
        btn.style.borderBottomLeftRadius = 8;
        btn.style.borderBottomRightRadius = 8;
        parent.Add(btn);
    }

    private void BuildAdjustmentPanel(bool isFloating)
    {
        _adjustmentPanel = new VisualElement();

        if (isFloating)
        {
            // 浮动面板
            _adjustmentPanel.style.position = Position.Absolute;
            _adjustmentPanel.style.right = 10;
            _adjustmentPanel.style.top = 60;
            _adjustmentPanel.style.width = 320;
        }
        else
        {
            // 固定在右侧
            _adjustmentPanel.style.width = 360;
        }

        _adjustmentPanel.style.backgroundColor = new StyleColor(new Color(0.1f, 0.1f, 0.1f, 0.92f));
        _adjustmentPanel.style.borderTopLeftRadius = 8;
        _adjustmentPanel.style.borderTopRightRadius = 8;
        _adjustmentPanel.style.borderBottomLeftRadius = 8;
        _adjustmentPanel.style.borderBottomRightRadius = 8;
        _adjustmentPanel.style.paddingLeft = 8;
        _adjustmentPanel.style.paddingRight = 8;
        _adjustmentPanel.style.paddingTop = 8;
        _adjustmentPanel.style.paddingBottom = 8;
        _adjustmentPanel.style.flexDirection = FlexDirection.Column;

        var header = new VisualElement();
        header.name = "drag-header";
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.height = 30;
        header.style.marginBottom = 6;
        header.style.backgroundColor = new Color(0.2f, 0.2f, 0.2f);
        _adjustmentPanel.Add(header);

        var title = new Label("图像调节");
        title.style.flexGrow = 1;
        title.style.color = Color.yellow;
        title.style.unityTextAlign = TextAnchor.MiddleLeft;
        title.style.paddingLeft = 8;
        header.Add(title);

        var body = new VisualElement();
        body.style.flexDirection = FlexDirection.Column;
        _adjustmentPanel.Add(body);

        _collapsePanelBtn = new Button(() =>
        {
            body.style.display = body.style.display == DisplayStyle.None ? DisplayStyle.Flex : DisplayStyle.None;
            _collapsePanelBtn.text = body.style.display == DisplayStyle.None ? "+" : "-";
        })
        { text = isFloating ? "+" : "-" };
        _collapsePanelBtn.style.height = 28;
        _collapsePanelBtn.style.fontSize = 28;
        header.Add(_collapsePanelBtn);

        if (isFloating)
        {
            body.style.display = DisplayStyle.None; // 竖屏时默认折叠
            // 为浮动面板添加拖拽功能
            SetupPanelDragging(_adjustmentPanel, header);
        }

        _collapsePanelBtn = new Button(() =>
        {
            body.style.display = body.style.display == DisplayStyle.None ? DisplayStyle.Flex : DisplayStyle.None;
            _collapsePanelBtn.text = body.style.display == DisplayStyle.None ? "+" : "-";
        })
        { text = isFloating ? "+" : "-" };
        _collapsePanelBtn.style.height = 28;
        _collapsePanelBtn.style.fontSize = 28;
        header.Add(_collapsePanelBtn);

        if (isFloating)
        {
            body.style.display = DisplayStyle.None; // 竖屏时默认折叠
        }

        // 调节滑块（复刻MainView中的浮动调节面板）
        AddAdjustmentSlider(body, "对比度", -0.5f, 0.5f, 0f);
        AddAdjustmentSlider(body, "亮度", -0.5f, 0.5f, 0f);
        AddAdjustmentSlider(body, "自然饱和度", -1f, 1f, 0f);
        AddAdjustmentSlider(body, "去阴影", 0f, 0.5f, 0f);
        AddAdjustmentSlider(body, "去高光", 0f, 0.5f, 0f);
        AddAdjustmentSlider(body, "加温滤镜", 0f, 1f, 0f);
        AddAdjustmentSlider(body, "冷却滤镜", 0f, 1f, 0f);
        AddAdjustmentSlider(body, "锐化", 0f, 4f, 0f);
        AddAdjustmentSlider(body, "模糊", 0f, 4f, 0f);

        if (isFloating)
        {
            _imageViewerContainer.Add(_adjustmentPanel);
        }
        else
        {
            _contentContainer.Add(_adjustmentPanel);
        }
    }

    private void AddAdjustmentSlider(VisualElement parent, string name, float min, float max, float defaultValue)
    {
        var row = new VisualElement();
        row.style.flexDirection = FlexDirection.Row;
        row.style.alignItems = Align.Center;
        row.style.marginBottom = 6;

        var label = new Label(name);
        label.style.width = 100;
        label.style.color = Color.white;
        row.Add(label);

        var slider = new Slider(min, max);
        slider.value = defaultValue;
        slider.style.flexGrow = 1;
        slider.style.marginRight = 8;
        row.Add(slider);

        var btn = new Button(() =>
        {
            ApplyAdjustment(name, slider.value);
        })
        { text = "应用" };
        btn.style.height = 28;
        row.Add(btn);

        parent.Add(row);
    }

    // 占位方法 - 这些将连接到实际的处理逻辑
    private void OnFaceSwap() { ShowToast(_toastOverlay, "换脸功能"); }
    private void OnSharpen() { ShowToast(_toastOverlay, "清晰功能"); }
    private void OnWhiten() { ShowToast(_toastOverlay, "美白功能"); }
    private void OnSharpenWhiten() { ShowToast(_toastOverlay, "清晰&美白功能"); }
    private void OnChangeBackground() { ShowToast(_toastOverlay, "换背景功能"); }
    private void OnDehazeColorGrade() { ShowToast(_toastOverlay, "去霾&调色功能"); }
    private void OnColorGrade() { ShowToast(_toastOverlay, "调色功能"); }
    private void OnDehaze() { ShowToast(_toastOverlay, "去霾功能"); }
    private void OnGpuSharpen() { ShowToast(_toastOverlay, "GPU清晰功能"); }
    private void OnRealEsrgan() { ShowToast(_toastOverlay, "ESRGAN功能"); }
    private void OnSave() { ShowToast(_toastOverlay, "保存功能"); }

    private void ApplyPreset(string presetName)
    {
        ShowToast(_toastOverlay, $"应用预设: {presetName}");
    }

    private void ApplyAdjustment(string name, float value)
    {
        ShowToast(_toastOverlay, $"{name}: {value:F2}");
    }

    private void OnHistorySelectionChanged(IEnumerable<object> selectedItems)
    {
        var first = System.Linq.Enumerable.FirstOrDefault(selectedItems);
        if (first is not HistoryEntry entry) return;

        var original = _historyEntries.Count > 0 ? _historyEntries[0].texture : null;
        _imageViewer?.SetSources(entry.texture, original, entry.label);
    }

    public void SetCurrentImage(Texture2D texture, string path)
    {
        _currentImage = texture;
        _currentImagePath = path;

        // 重置历史记录
        _historyEntries.Clear();
        _historyOpSeq = 0;

        if (texture != null)
        {
            _historyEntries.Add(new HistoryEntry
            {
                label = "原图: " + (path != null ? System.IO.Path.GetFileName(path) : texture.name),
                texture = texture,
                owned = false,
                sourcePath = path,
                opSeq = 0
            });

            _historyList?.RefreshItems();
            _historyList?.SetSelection(0);
            _imageViewer?.SetSources(texture, texture, "原图");
            _imageViewer?.FitToView();
        }
    }

    [Serializable]
    private struct HistoryEntry
    {
        public string label;
        public Texture2D texture;
        public bool owned;
        public string sourcePath;
        public long opSeq;
    }
}
