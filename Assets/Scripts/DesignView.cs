using System;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

/// <summary>
/// DesignView - 照片设计页面
/// 支持YOLO分割，每个对象作为可拖动、可缩放的图层
/// </summary>
public class DesignView : BasePageView
{
    private VisualElement _imageViewerContainer;
    private VisualElement _layerPanel;
    private VisualElement _toastOverlay;
    private SplitCompareImageView _imageViewer;
    private ScrollView _layerListScroll;
    private VisualElement _layerListContainer;
    private Button _applyButton;

    private List<HistoryEntry> _historyEntries = new List<HistoryEntry>();
    private ListView _historyList;
    private long _historyOpSeq;

    private Texture2D _currentImage;
    private string _currentImagePath;
    private List<LayerData> _layers = new List<LayerData>();
    private int _nextLayerId = 1;

    public override void BuildPage()
    {
        _pageContainer = new VisualElement();
        _pageContainer.style.width = Length.Percent(100);
        _pageContainer.style.height = Length.Percent(100);
        _pageContainer.style.position = Position.Relative;
        _pageContainer.style.flexDirection = FlexDirection.Column;

        // 顶部工具栏
        BuildTopToolbar();

        // 主内容区域
        var contentContainer = new VisualElement();
        contentContainer.style.flexGrow = 1;
        contentContainer.style.flexDirection = FlexDirection.Row;
        contentContainer.style.minHeight = 0;
        _pageContainer.Add(contentContainer);

        // 图像查看器容器
        _imageViewerContainer = new VisualElement();
        _imageViewerContainer.style.flexGrow = 1;
        _imageViewerContainer.style.position = Position.Relative;
        contentContainer.Add(_imageViewerContainer);

        // 创建ImageViewer
        BuildImageViewer();

        // 右侧图层面板
        BuildLayerPanel();
        contentContainer.Add(_layerPanel);

        // 历史记录面板（浮动）
        BuildHistoryPanel();

        // Toast提示
        _toastOverlay = BuildToast();
        _pageContainer.Add(_toastOverlay);

        // 页面切换指示器
        BuildPageIndicator(PageType.DesignView);
    }

    private void BuildTopToolbar()
    {
        var toolbar = new VisualElement();
        toolbar.style.flexDirection = FlexDirection.Row;
        toolbar.style.alignItems = Align.Center;
        toolbar.style.paddingLeft = 8;
        toolbar.style.paddingRight = 8;
        toolbar.style.paddingTop = 6;
        toolbar.style.paddingBottom = 6;
        toolbar.style.backgroundColor = new StyleColor(new Color(0.15f, 0.15f, 0.15f, 1f));
        _pageContainer.Add(toolbar);

        var title = new Label("设计模式");
        title.style.flexGrow = 1;
        title.style.color = Color.white;
        title.style.fontSize = 16;
        toolbar.Add(title);

        var detectButton = new Button(OnDetectObjects) { text = "识别对象 (YOLO)" };
        detectButton.style.backgroundColor = new StyleColor(new Color(0.3f, 0.6f, 0.9f, 1f));
        detectButton.style.color = Color.white;
        toolbar.Add(detectButton);

        _applyButton = new Button(OnApplyChanges) { text = "应用生成 (SD-Inpainting)" };
        _applyButton.style.backgroundColor = new StyleColor(new Color(0.3f, 0.8f, 0.3f, 1f));
        _applyButton.style.color = Color.white;
        _applyButton.style.marginLeft = 8;
        toolbar.Add(_applyButton);
    }

    private void BuildImageViewer()
    {
        _imageViewer = new SplitCompareImageView();
        _imageViewer.style.flexGrow = 1;
        _imageViewer.style.minHeight = 0;
        _imageViewerContainer.Add(_imageViewer);

        // 在ImageViewer上叠加图层框
        BuildLayerOverlay();
    }

    private void BuildLayerOverlay()
    {
        // 图层框将在这里动态创建
        // 每个图层框可拖动、可缩放
    }

    private void BuildLayerPanel()
    {
        _layerPanel = new VisualElement();
        _layerPanel.style.width = 280;
        _layerPanel.style.backgroundColor = new StyleColor(new Color(0.12f, 0.12f, 0.12f, 1f));
        _layerPanel.style.flexDirection = FlexDirection.Column;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.paddingLeft = 8;
        header.style.paddingRight = 8;
        header.style.paddingTop = 8;
        header.style.paddingBottom = 8;
        header.style.backgroundColor = new StyleColor(new Color(0.15f, 0.15f, 0.15f, 1f));
        _layerPanel.Add(header);

        var title = new Label("图层列表");
        title.style.flexGrow = 1;
        title.style.color = Color.white;
        title.style.fontSize = 14;
        header.Add(title);

        var clearButton = new Button(OnClearLayers) { text = "清空" };
        clearButton.style.fontSize = 12;
        header.Add(clearButton);

        _layerListScroll = new ScrollView(ScrollViewMode.Vertical);
        _layerListScroll.style.flexGrow = 1;
        _layerListScroll.style.minHeight = 0;
        _layerPanel.Add(_layerListScroll);

        _layerListContainer = new VisualElement();
        _layerListContainer.style.flexDirection = FlexDirection.Column;
        _layerListContainer.style.paddingLeft = 8;
        _layerListContainer.style.paddingRight = 8;
        _layerListContainer.style.paddingTop = 8;
        _layerListScroll.Add(_layerListContainer);
    }

    private void BuildHistoryPanel()
    {
        var historyPanel = new VisualElement();
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
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.marginBottom = 6;
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

        _imageViewerContainer.Add(historyPanel);
    }

    private void OnDetectObjects()
    {
        DetectObjectsAsync().Forget();
    }

    private async UniTaskVoid DetectObjectsAsync()
    {
        if (_currentImage == null)
        {
            ShowToast(_toastOverlay, "请先加载图片");
            return;
        }

        ShowToast(_toastOverlay, "正在识别对象（YOLO-Seg）...");

        // 模拟YOLO检测 - 实际应该调用YoloSegNcnnReproRunner
        await UniTask.Delay(1000);

        // 创建模拟图层
        CreateMockLayers();

        ShowToast(_toastOverlay, $"检测到 {_layers.Count} 个对象");
    }

    private void CreateMockLayers()
    {
        _layers.Clear();
        _layerListContainer.Clear();

        // 模拟创建几个图层
        var mockObjects = new[]
        {
            new { name = "人物1", x = 0.2f, y = 0.3f, w = 0.3f, h = 0.5f },
            new { name = "人物2", x = 0.5f, y = 0.3f, w = 0.25f, h = 0.45f },
            new { name = "背景物体", x = 0.1f, y = 0.1f, w = 0.2f, h = 0.2f }
        };

        foreach (var obj in mockObjects)
        {
            var layer = new LayerData
            {
                id = _nextLayerId++,
                name = obj.name,
                x = obj.x,
                y = obj.y,
                width = obj.w,
                height = obj.h,
                visible = true,
                locked = false
            };

            _layers.Add(layer);
            CreateLayerUI(layer);
            CreateLayerFrame(layer);
        }
    }

    private void CreateLayerUI(LayerData layer)
    {
        var layerItem = new VisualElement();
        layerItem.style.flexDirection = FlexDirection.Column;
        layerItem.style.marginBottom = 8;
        layerItem.style.paddingLeft = 8;
        layerItem.style.paddingRight = 8;
        layerItem.style.paddingTop = 6;
        layerItem.style.paddingBottom = 6;
        layerItem.style.backgroundColor = new StyleColor(new Color(0.2f, 0.2f, 0.2f, 1f));
        layerItem.style.borderTopLeftRadius = 4;
        layerItem.style.borderTopRightRadius = 4;
        layerItem.style.borderBottomLeftRadius = 4;
        layerItem.style.borderBottomRightRadius = 4;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        layerItem.Add(header);

        var nameLabel = new Label(layer.name);
        nameLabel.style.flexGrow = 1;
        nameLabel.style.color = Color.white;
        header.Add(nameLabel);

        var visibleToggle = new Toggle();
        visibleToggle.value = layer.visible;
        visibleToggle.RegisterValueChangedCallback(evt =>
        {
            layer.visible = evt.newValue;
            UpdateLayerFrameVisibility(layer);
        });
        header.Add(visibleToggle);

        var deleteButton = new Button(() => OnDeleteLayer(layer)) { text = "X" };
        deleteButton.style.width = 24;
        deleteButton.style.height = 24;
        deleteButton.style.marginLeft = 4;
        header.Add(deleteButton);

        _layerListContainer.Add(layerItem);
    }

    private void CreateLayerFrame(LayerData layer)
    {
        var frame = new VisualElement();
        frame.name = $"layer-frame-{layer.id}";
        frame.style.position = Position.Absolute;
        frame.style.left = Length.Percent(layer.x * 100);
        frame.style.top = Length.Percent(layer.y * 100);
        frame.style.width = Length.Percent(layer.width * 100);
        frame.style.height = Length.Percent(layer.height * 100);
        frame.style.borderLeftWidth = 2;
        frame.style.borderRightWidth = 2;
        frame.style.borderTopWidth = 2;
        frame.style.borderBottomWidth = 2;
        frame.style.borderLeftColor = new StyleColor(Color.cyan);
        frame.style.borderRightColor = new StyleColor(Color.cyan);
        frame.style.borderTopColor = new StyleColor(Color.cyan);
        frame.style.borderBottomColor = new StyleColor(Color.cyan);

        // 标题栏（可拖动）
        var titleBar = new VisualElement();
        titleBar.style.height = 24;
        titleBar.style.backgroundColor = new StyleColor(new Color(0f, 0.8f, 0.8f, 0.8f));
        titleBar.style.flexDirection = FlexDirection.Row;
        titleBar.style.alignItems = Align.Center;
        frame.Add(titleBar);

        var titleLabel = new Label(layer.name);
        titleLabel.style.flexGrow = 1;
        titleLabel.style.color = Color.white;
        titleLabel.style.paddingLeft = 4;
        titleLabel.style.fontSize = 12;
        titleBar.Add(titleLabel);

        // 拖动逻辑
        bool dragging = false;
        int dragPointerId = -1;
        Vector2 dragStartPointer = default;
        Vector2 dragStartPos = default;

        titleBar.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (evt.button != 0) return;
            dragging = true;
            dragPointerId = evt.pointerId;
            dragStartPointer = evt.position;
            dragStartPos = new Vector2(frame.resolvedStyle.left, frame.resolvedStyle.top);
            titleBar.CapturePointer(dragPointerId);
            evt.StopPropagation();
        });

        titleBar.RegisterCallback<PointerMoveEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId) return;

            var delta = (Vector2)evt.position - dragStartPointer;
            var newPos = dragStartPos + delta;

            frame.style.left = newPos.x;
            frame.style.top = newPos.y;

            // 更新图层数据
            var parentSize = _imageViewerContainer.contentRect.size;
            if (parentSize.x > 0 && parentSize.y > 0)
            {
                layer.x = newPos.x / parentSize.x;
                layer.y = newPos.y / parentSize.y;
            }

            evt.StopPropagation();
        });

        titleBar.RegisterCallback<PointerUpEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId) return;
            dragging = false;
            if (titleBar.HasPointerCapture(dragPointerId))
                titleBar.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });

        // 四角缩放handle
        CreateResizeHandles(frame, layer);

        _imageViewerContainer.Add(frame);
        layer.frameElement = frame;
    }

    private void CreateResizeHandles(VisualElement frame, LayerData layer)
    {
        var handleSize = 12;
        var positions = new[]
        {
            new { name = "top-left", left = -6f, top = -6f, right = 0f, bottom = 0f },
            new { name = "top-right", left = 0f, top = -6f, right = -6f, bottom = 0f },
            new { name = "bottom-left", left = -6f, top = 0f, right = 0f, bottom = -6f },
            new { name = "bottom-right", left = 0f, top = 0f, right = -6f, bottom = -6f }
        };

        foreach (var pos in positions)
        {
            var handle = new VisualElement();
            handle.name = $"handle-{pos.name}";
            handle.style.position = Position.Absolute;
            handle.style.width = handleSize;
            handle.style.height = handleSize;
            handle.style.backgroundColor = new StyleColor(Color.white);
            handle.style.borderTopLeftRadius = handleSize / 2;
            handle.style.borderTopRightRadius = handleSize / 2;
            handle.style.borderBottomLeftRadius = handleSize / 2;
            handle.style.borderBottomRightRadius = handleSize / 2;

            if (pos.name.Contains("left"))
                handle.style.left = pos.left;
            else
                handle.style.right = pos.right;

            if (pos.name.Contains("top"))
                handle.style.top = pos.top;
            else
                handle.style.bottom = pos.bottom;

            // 缩放逻辑（简化版，完整实现需要更复杂的计算）
            bool resizing = false;
            int resizePointerId = -1;

            handle.RegisterCallback<PointerDownEvent>(evt =>
            {
                if (evt.button != 0) return;
                resizing = true;
                resizePointerId = evt.pointerId;
                handle.CapturePointer(resizePointerId);
                evt.StopPropagation();
            });

            handle.RegisterCallback<PointerMoveEvent>(evt =>
            {
                if (!resizing || resizePointerId != evt.pointerId) return;
                // 缩放逻辑
                evt.StopPropagation();
            });

            handle.RegisterCallback<PointerUpEvent>(evt =>
            {
                if (!resizing || resizePointerId != evt.pointerId) return;
                resizing = false;
                if (handle.HasPointerCapture(resizePointerId))
                    handle.ReleasePointer(resizePointerId);
                evt.StopPropagation();
            });

            frame.Add(handle);
        }
    }

    private void UpdateLayerFrameVisibility(LayerData layer)
    {
        if (layer.frameElement != null)
        {
            layer.frameElement.style.display = layer.visible ? DisplayStyle.Flex : DisplayStyle.None;
        }
    }

    private void OnDeleteLayer(LayerData layer)
    {
        _layers.Remove(layer);
        if (layer.frameElement != null && _imageViewerContainer.Contains(layer.frameElement))
        {
            _imageViewerContainer.Remove(layer.frameElement);
        }
        RefreshLayerList();
        ShowToast(_toastOverlay, $"已删除图层: {layer.name}");
    }

    private void OnClearLayers()
    {
        foreach (var layer in _layers)
        {
            if (layer.frameElement != null && _imageViewerContainer.Contains(layer.frameElement))
            {
                _imageViewerContainer.Remove(layer.frameElement);
            }
        }
        _layers.Clear();
        RefreshLayerList();
        ShowToast(_toastOverlay, "已清空所有图层");
    }

    private void RefreshLayerList()
    {
        _layerListContainer.Clear();
        foreach (var layer in _layers)
        {
            CreateLayerUI(layer);
        }
    }

    private void OnApplyChanges()
    {
        ApplyChangesAsync().Forget();
    }

    private async UniTaskVoid ApplyChangesAsync()
    {
        if (_layers.Count == 0)
        {
            ShowToast(_toastOverlay, "没有图层可应用");
            return;
        }

        ShowToast(_toastOverlay, "正在应用更改（SD-Inpainting）...");

        // 占位 - 实际应该：
        // 1. 重新计算背景mask区域
        // 2. 调用SD-Inpainting生成新图
        await UniTask.Delay(2000);

        ShowToast(_toastOverlay, "生成完成（功能未实现，留接口）");
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

        // 清空图层
        OnClearLayers();
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

    private class LayerData
    {
        public int id;
        public string name;
        public float x, y, width, height;
        public bool visible;
        public bool locked;
        public VisualElement frameElement;
    }

    // 简化的SplitCompareImageView
    private sealed class SplitCompareImageView : VisualElement
    {
        private readonly Label _info;
        private Texture _texA;
        private Texture _texB;
        private float _zoom = 1f;
        private Vector2 _pan;

        public SplitCompareImageView()
        {
            style.flexGrow = 1;
            style.backgroundColor = new StyleColor(new Color(0.12f, 0.12f, 0.12f, 1f));

            _info = new Label();
            _info.style.paddingLeft = 8;
            _info.style.paddingTop = 6;
            _info.style.paddingBottom = 6;
            _info.style.color = Color.white;
            Add(_info);

            generateVisualContent += OnGenerateVisualContent;
            RegisterCallback<GeometryChangedEvent>(_ => { if (_texA != null || _texB != null) FitToView(); });
        }

        public void SetSources(Texture2D current, Texture2D original, string label)
        {
            _texA = current;
            _texB = original;
            _info.text = label ?? "";
            MarkDirtyRepaint();
        }

        public void FitToView()
        {
            var refTex = _texA ?? _texB;
            if (refTex == null) return;

            var viewRect = contentRect;
            if (viewRect.width <= 1f || viewRect.height <= 1f) return;

            var scaleX = viewRect.width / refTex.width;
            var scaleY = viewRect.height / refTex.height;
            _zoom = Mathf.Clamp(Mathf.Min(scaleX, scaleY), 0.01f, 20f);

            var scaledSize = new Vector2(refTex.width * _zoom, refTex.height * _zoom);
            _pan = (viewRect.size - scaledSize) * 0.5f;
            MarkDirtyRepaint();
        }

        private void OnGenerateVisualContent(MeshGenerationContext mgc)
        {
            var refTex = _texA ?? _texB;
            if (refTex == null) return;

            var viewRect = contentRect;
            if (viewRect.width <= 1f || viewRect.height <= 1f) return;

            var imageRect = new Rect(_pan.x, _pan.y, refTex.width * _zoom, refTex.height * _zoom);
            var drawRect = IntersectRect(viewRect, imageRect);
            if (drawRect.width <= 1f || drawRect.height <= 1f) return;

            if (_texA != null)
            {
                var mesh = mgc.Allocate(4, 6, _texA);
                var p0 = new Vector2(drawRect.xMin, drawRect.yMin);
                var p1 = new Vector2(drawRect.xMax, drawRect.yMin);
                var p2 = new Vector2(drawRect.xMax, drawRect.yMax);
                var p3 = new Vector2(drawRect.xMin, drawRect.yMax);

                mesh.SetNextVertex(new Vertex { position = p0, uv = new Vector2(0, 0), tint = Color.white });
                mesh.SetNextVertex(new Vertex { position = p1, uv = new Vector2(1, 0), tint = Color.white });
                mesh.SetNextVertex(new Vertex { position = p2, uv = new Vector2(1, 1), tint = Color.white });
                mesh.SetNextVertex(new Vertex { position = p3, uv = new Vector2(0, 1), tint = Color.white });

                mesh.SetNextIndex(0);
                mesh.SetNextIndex(1);
                mesh.SetNextIndex(2);
                mesh.SetNextIndex(0);
                mesh.SetNextIndex(2);
                mesh.SetNextIndex(3);
            }
        }

        private static Rect IntersectRect(Rect a, Rect b)
        {
            var xMin = Mathf.Max(a.xMin, b.xMin);
            var yMin = Mathf.Max(a.yMin, b.yMin);
            var xMax = Mathf.Min(a.xMax, b.xMax);
            var yMax = Mathf.Min(a.yMax, b.yMax);
            if (xMax <= xMin || yMax <= yMin)
                return new Rect(0, 0, 0, 0);
            return Rect.MinMaxRect(xMin, yMin, xMax, yMax);
        }
    }
}
