using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UIElements;
using Process = System.Diagnostics.Process;
using ProcessStartInfo = System.Diagnostics.ProcessStartInfo;

public enum AppPageId
{
    MainView2,
    LibraryView,
    DesignView
}

public enum SwipeDirection
{
    Left = -1,
    Right = 1
}

public abstract class BasePageView : MonoBehaviour
{
    protected sealed class HistoryEntry
    {
        public string label;
        public Texture2D texture;
        public bool owned;
        public string sourcePath;
        public long opSeq;
    }

    protected AIImagePageHost Host { get; private set; }
    protected UIDocument Document { get; private set; }
    protected VisualElement PageRoot => _pageRoot;
    protected VisualElement ContentRoot => _contentRoot;
    protected BeforeAfterCompareView CompareView => _compareView;
    protected string CurrentImagePath => _currentImagePath;
    protected string CurrentImageLabel => _currentImageLabel;
    protected bool IsPortraitLayout => _isPortraitLayout;

    public abstract AppPageId PageId { get; }

    private VisualElement _pageRoot;
    private VisualElement _contentRoot;
    private VisualElement _switchZone;
    private VisualElement _switchTrack;
    private VisualElement _switchPill;
    private Vector2 _switchDragStart;
    private bool _switchDragging;
    private int _switchPointerId = -1;
    private bool _isPortraitLayout;

    private BeforeAfterCompareView _compareView;
    private readonly List<HistoryEntry> _historyEntries = new List<HistoryEntry>();
    private ListView _historyList;
    private long _historyOpSeq;
    private string _currentImagePath;
    private string _currentImageLabel;

    private VisualElement _busyOverlay;
    private VisualElement _busyBarTrack;
    private VisualElement _busyBar;
    private Label _busyText;
    private IVisualElementScheduledItem _busyAnim;
    private float _busyPhase;

    private VisualElement _progressOverlay;
    private ProgressBar _progressBar;
    private Label _progressTitle;
    private Label _progressDetail;
    private IVisualElementScheduledItem _progressTick;
    private readonly object _progressLock = new object();
    private float _progressValue01;
    private string _progressText;

    private VisualElement _choiceOverlay;
    private UniTaskCompletionSource<int> _choiceTcs;

    private VisualElement _toastOverlay;
    private Label _toastText;
    private IVisualElementScheduledItem _toastHide;

    private bool _previewRunning;
    private RenderTexture _previewRt;
    private Texture2D _previewSource;
    private string _previewKernelName;
    private Action<ComputeShader, float> _previewParamSetter;
    private float _previewValue;
    private int _previewPointerId = -1;
    private VisualElement _previewCaptureElement;

    internal void Initialize(AIImagePageHost host, UIDocument document)
    {
        Host = host;
        Document = document;
        OnInitialized();
    }

    internal void AttachTo(VisualElement parent)
    {
        if (parent == null)
            return;

        BuildRoot();
        parent.Add(_pageRoot);
        _pageRoot.BringToFront();
        OnShown();
        OnLayoutChanged(_isPortraitLayout, _pageRoot.contentRect);
    }

    internal void Detach()
    {
        StopPreview();
        HideBusy();
        HideProgress();
        HideChoice();
        _busyAnim?.Pause();
        _busyAnim = null;
        _progressTick?.Pause();
        _progressTick = null;
        _toastHide?.Pause();
        _toastHide = null;
        if (_toastOverlay != null)
            _toastOverlay.style.display = DisplayStyle.None;
        OnBeforeDetach();
        if (_pageRoot != null && _pageRoot.parent != null)
            _pageRoot.parent.Remove(_pageRoot);
        _pageRoot = null;
        _contentRoot = null;
        _switchZone = null;
        _switchTrack = null;
        _switchPill = null;
        _historyList = null;
        _busyOverlay = null;
        _busyBarTrack = null;
        _busyBar = null;
        _busyText = null;
        _progressOverlay = null;
        _progressBar = null;
        _progressTitle = null;
        _progressDetail = null;
        _choiceOverlay = null;
        _choiceTcs = null;
        _toastOverlay = null;
        _toastText = null;
        _compareView = null;
    }

    internal void SetPageOffset(float x, float opacity = 1f)
    {
        if (_pageRoot == null)
            return;
        _pageRoot.style.translate = new Translate(new Length(x, LengthUnit.Pixel), new Length(0f, LengthUnit.Pixel));
        _pageRoot.style.opacity = opacity;
    }

    internal float ResolveWidth()
    {
        if (_pageRoot == null)
            return 0f;
        return Mathf.Max(1f, _pageRoot.resolvedStyle.width);
    }

    protected virtual void OnInitialized() { }
    protected virtual void OnShown() { }
    protected virtual void OnBeforeDetach() { }
    protected virtual void OnLayoutChanged(bool isPortrait, Rect layoutRect) { }
    protected virtual AppPageId? ResolveSwipeTarget(SwipeDirection direction) => null;
    protected virtual bool UseOverlaySwitchZone => false;
    protected virtual float GetSwitchPillAlignment01() => 0.5f;
    protected abstract void BuildPage(VisualElement contentRoot);

    protected virtual void OnDestroy()
    {
        ClearHistory();
        StopPreview();
    }

    protected BeforeAfterCompareView CreateCompareView(VisualElement parent, bool allowInteraction = true)
    {
        _compareView = new BeforeAfterCompareView
        {
            InteractionEnabled = allowInteraction
        };
        _compareView.style.flexGrow = 1;
        _compareView.style.minHeight = 0;
        parent.Add(_compareView);
        return _compareView;
    }

    protected VisualElement CreateFloatingHistoryPanel(float width = 220f, string title = "历史记录")
    {
        var panel = new VisualElement();
        panel.style.position = Position.Absolute;
        panel.style.left = 16;
        panel.style.top = 16;
        panel.style.width = width;
        panel.style.maxHeight = 360;
        panel.style.backgroundColor = new StyleColor(new Color(0.10f, 0.11f, 0.14f, 0.86f));
        panel.style.borderTopLeftRadius = 18;
        panel.style.borderTopRightRadius = 18;
        panel.style.borderBottomLeftRadius = 18;
        panel.style.borderBottomRightRadius = 18;
        panel.style.borderLeftWidth = 1;
        panel.style.borderTopWidth = 1;
        panel.style.borderRightWidth = 1;
        panel.style.borderBottomWidth = 1;
        panel.style.borderLeftColor = new Color(1f, 1f, 1f, 0.10f);
        panel.style.borderRightColor = new Color(1f, 1f, 1f, 0.10f);
        panel.style.borderTopColor = new Color(1f, 1f, 1f, 0.10f);
        panel.style.borderBottomColor = new Color(1f, 1f, 1f, 0.10f);
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

        var headerTitle = new Label(title);
        headerTitle.style.flexGrow = 1;
        headerTitle.style.color = Color.white;
        headerTitle.style.unityFontStyleAndWeight = FontStyle.Bold;
        header.Add(headerTitle);

        var undoButton = CreateMiniActionButton("↺", UndoLastOperation, "撤销");
        header.Add(undoButton);

        var deleteButton = CreateMiniActionButton("⌫", DeleteSelectedHistoryEntry, "删除当前");
        deleteButton.style.marginLeft = 6;
        header.Add(deleteButton);

        _historyList = new ListView();
        _historyList.style.flexGrow = 1;
        _historyList.style.flexBasis = 0;
        _historyList.style.minHeight = 120;
        _historyList.fixedItemHeight = 28;
        _historyList.showAlternatingRowBackgrounds = AlternatingRowBackground.ContentOnly;
        _historyList.selectionType = SelectionType.Single;
        _historyList.itemsSource = _historyEntries;
        _historyList.makeItem = () =>
        {
            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.style.paddingLeft = 8;
            row.style.paddingRight = 8;
            row.style.alignItems = Align.Center;
            row.style.height = 28;

            var label = new Label();
            label.style.flexGrow = 1;
            label.style.color = Color.white;
            label.style.whiteSpace = WhiteSpace.NoWrap;
            label.style.overflow = Overflow.Hidden;
            label.style.textOverflow = TextOverflow.Ellipsis;
            row.Add(label);
            return row;
        };
        _historyList.bindItem = (element, index) =>
        {
            var label = element.Q<Label>();
            label.text = _historyEntries[index].label;
        };
        _historyList.selectionChanged += OnHistorySelectionChanged;
        panel.Add(_historyList);

        EnableFloatingPanelDrag(panel, header);
        RefreshHistoryUi();
        return panel;
    }

    protected void EnableFloatingPanelDrag(VisualElement panel, VisualElement dragHandle, float margin = 10f)
    {
        if (panel == null || dragHandle == null)
            return;

        var dragging = false;
        var dragPointerId = -1;
        var startPointer = Vector2.zero;
        var startPos = Vector2.zero;

        dragHandle.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (evt.button != 0 || panel.parent == null)
                return;
            if (IsDragBlockedByInteractiveChild(evt.target, dragHandle))
                return;

            dragging = true;
            dragPointerId = evt.pointerId;
            startPointer = evt.position;

            var panelRect = panel.worldBound;
            var parentRect = panel.parent.worldBound;
            startPos = new Vector2(panelRect.xMin - parentRect.xMin, panelRect.yMin - parentRect.yMin);

            panel.style.left = startPos.x;
            panel.style.top = startPos.y;
            panel.style.right = new StyleLength(StyleKeyword.Auto);
            panel.style.bottom = new StyleLength(StyleKeyword.Auto);
            panel.style.width = panel.resolvedStyle.width;
            panel.style.height = panel.resolvedStyle.height;

            dragHandle.CapturePointer(dragPointerId);
            evt.StopPropagation();
        });

        dragHandle.RegisterCallback<PointerMoveEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId || panel.parent == null || !dragHandle.HasPointerCapture(dragPointerId))
                return;

            var delta = (Vector2)evt.position - startPointer;
            var newPos = startPos + delta;
            var bounds = panel.parent.contentRect;
            var panelWidth = Mathf.Max(1f, panel.resolvedStyle.width);
            var panelHeight = Mathf.Max(1f, panel.resolvedStyle.height);
            var headerHeight = Mathf.Max(32f, dragHandle.resolvedStyle.height);

            var minX = margin - panelWidth;
            var maxX = Mathf.Max(margin, bounds.width - margin);
            var minY = margin;
            var maxY = Mathf.Max(margin, bounds.height - headerHeight - margin);

            newPos.x = Mathf.Clamp(newPos.x, minX, maxX);
            newPos.y = Mathf.Clamp(newPos.y, minY, maxY);

            panel.style.left = newPos.x;
            panel.style.top = newPos.y;
            evt.StopPropagation();
        });

        dragHandle.RegisterCallback<PointerUpEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId)
                return;

            dragging = false;
            if (dragHandle.HasPointerCapture(dragPointerId))
                dragHandle.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });

        dragHandle.RegisterCallback<PointerCancelEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId)
                return;

            dragging = false;
            if (dragHandle.HasPointerCapture(dragPointerId))
                dragHandle.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });
    }

    private static bool IsDragBlockedByInteractiveChild(IEventHandler target, VisualElement dragHandle)
    {
        if (target is not VisualElement ve)
            return false;

        var current = ve;
        while (current != null && !ReferenceEquals(current, dragHandle))
        {
            if (current is Button || current is Toggle || current is Slider || current is TextField || current is Scroller)
                return true;
            current = current.parent;
        }

        return false;
    }

    protected void BuildStandardOverlays()
    {
        if (_pageRoot == null)
            return;
        BuildBusyOverlay(_pageRoot);
        BuildProgressOverlay(_pageRoot);
        BuildToast(_pageRoot);
    }

    protected void SetHistoryFromSharedTextures(Texture2D originalTexture, Texture2D currentTexture, string label, string fullPath)
    {
        ClearHistory();
        _historyOpSeq = 0;
        _currentImagePath = fullPath;
        _currentImageLabel = label;

        if (originalTexture != null)
        {
            _historyEntries.Add(new HistoryEntry
            {
                label = "原图: " + (label ?? originalTexture.name),
                texture = originalTexture,
                owned = false,
                sourcePath = fullPath,
                opSeq = 0
            });
        }

        if (currentTexture != null && !ReferenceEquals(currentTexture, originalTexture))
        {
            _historyEntries.Insert(0, new HistoryEntry
            {
                label = label ?? currentTexture.name,
                texture = currentTexture,
                owned = false,
                sourcePath = fullPath,
                opSeq = ++_historyOpSeq
            });
        }

        RefreshHistoryUi();
        if (_historyEntries.Count > 0)
            SelectHistoryIndex(0);
    }

    protected void ResetHistoryWithOriginal(Texture2D originalTexture, string label, string fullPath)
    {
        ClearHistory();
        _historyOpSeq = 0;
        _currentImagePath = fullPath;
        _currentImageLabel = label ?? (originalTexture != null ? originalTexture.name : null);

        if (originalTexture == null)
        {
            RefreshHistoryUi();
            return;
        }

        _historyEntries.Add(new HistoryEntry
        {
            label = "原图: " + (label ?? originalTexture.name),
            texture = originalTexture,
            owned = false,
            sourcePath = fullPath,
            opSeq = 0
        });

        RefreshHistoryUi();
        SelectHistoryIndex(0);
    }

    protected void AddHistory(Texture2D texture, string label)
    {
        if (texture == null || _historyEntries.Count == 0)
            return;

        var entry = new HistoryEntry
        {
            label = label ?? texture.name,
            texture = texture,
            owned = true,
            sourcePath = null,
            opSeq = ++_historyOpSeq
        };

        _historyEntries.Insert(0, entry);
        RefreshHistoryUi();
        SelectHistoryIndex(0);
    }

    protected void AddSharedHistory(Texture2D texture, string label)
    {
        if (texture == null || _historyEntries.Count == 0)
            return;

        _historyEntries.Insert(0, new HistoryEntry
        {
            label = label ?? texture.name,
            texture = texture,
            owned = false,
            sourcePath = _currentImagePath,
            opSeq = ++_historyOpSeq
        });
        RefreshHistoryUi();
        SelectHistoryIndex(0);
    }

    protected Texture2D GetCurrentHistoryTexture()
    {
        if (_historyEntries.Count == 0)
            return null;
        if (_historyList == null)
            return _historyEntries[0].texture;
        var index = _historyList.selectedIndex;
        if (index < 0 || index >= _historyEntries.Count)
            index = 0;
        return _historyEntries[index].texture;
    }

    protected string GetCurrentHistoryLabel()
    {
        if (_historyEntries.Count == 0)
            return _currentImageLabel;
        if (_historyList == null)
            return _historyEntries[0].label;
        var index = _historyList.selectedIndex;
        if (index < 0 || index >= _historyEntries.Count)
            index = 0;
        return _historyEntries[index].label;
    }

    protected Texture2D GetOriginalHistoryTexture()
    {
        if (_historyEntries.Count == 0)
            return null;
        return _historyEntries[_historyEntries.Count - 1].texture;
    }

    protected void SelectHistoryIndex(int index)
    {
        if (_historyEntries.Count == 0)
        {
            _compareView?.Clear();
            return;
        }

        index = Mathf.Clamp(index, 0, _historyEntries.Count - 1);
        _historyList?.SetSelection(index);
        _historyList?.ScrollToItem(index);
        var current = _historyEntries[index].texture;
        var original = GetOriginalHistoryTexture();
        _compareView?.SetSources(current, original, _historyEntries[index].label);
        _compareView?.FitToView();
    }

    protected void ClearHistory()
    {
        for (var i = 0; i < _historyEntries.Count; i++)
        {
            if (_historyEntries[i].owned && _historyEntries[i].texture != null)
                Destroy(_historyEntries[i].texture);
        }
        _historyEntries.Clear();
        RefreshHistoryUi();
        _compareView?.Clear();
    }

    protected void UndoLastOperation()
    {
        if (_historyEntries.Count <= 1)
            return;
        _historyEntries.RemoveAt(0);
        RefreshHistoryUi();
        SelectHistoryIndex(0);
    }

    protected void DeleteSelectedHistoryEntry()
    {
        if (_historyEntries.Count <= 1 || _historyList == null)
            return;

        var index = _historyList.selectedIndex;
        if (index < 0 || index >= _historyEntries.Count)
            return;

        var entry = _historyEntries[index];
        _historyEntries.RemoveAt(index);
        if (entry.owned && entry.texture != null)
            Destroy(entry.texture);
        RefreshHistoryUi();
        SelectHistoryIndex(Mathf.Clamp(index, 0, _historyEntries.Count - 1));
    }

    protected async UniTask ApplyComputeAdjustmentAsync(string kernelName, Action<ComputeShader> setParams, string historyLabel)
    {
        var src = GetCurrentHistoryTexture();
        if (src == null)
            src = GetOriginalHistoryTexture();
        if (src == null)
            return;

        var cs = Host != null ? Host.ImageProcessingCS : null;
        if (cs == null)
        {
            ShowToast("找不到 ImageProcessing.compute", 2400);
            return;
        }

        var kernel = GetKernelId(cs, kernelName);
        if (kernel < 0)
        {
            ShowToast("无效的调节内核: " + kernelName, 2400);
            return;
        }

        ShowBusy(historyLabel);
        RenderTexture rt = null;
        try
        {
            rt = new RenderTexture(src.width, src.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
            {
                enableRandomWrite = true
            };
            rt.Create();

            cs.SetTexture(kernel, "_Source", src);
            cs.SetTexture(kernel, "_Result", rt);
            setParams?.Invoke(cs);

            var gx = Mathf.CeilToInt(src.width / 8f);
            var gy = Mathf.CeilToInt(src.height / 8f);
            cs.Dispatch(kernel, gx, gy, 1);

            var tex = await ReadbackTextureAsync(rt, src.width, src.height);
            if (tex != null)
                AddHistory(tex, historyLabel);
        }
        finally
        {
            if (rt != null)
            {
                rt.Release();
                Destroy(rt);
            }
            HideBusy();
        }
    }

    protected VisualElement CreateAdjustRow(string name, float min, float max, float defaultValue, string kernelName, Action<ComputeShader, float> paramSetter, Func<float, string> historyLabelFactory)
    {
        var card = new VisualElement();
        card.style.flexDirection = FlexDirection.Column;
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

        var row = new VisualElement();
        row.style.flexDirection = FlexDirection.Row;
        row.style.alignItems = Align.Center;
        card.Add(row);

        var label = new Label(name);
        label.style.flexGrow = 1;
        label.style.color = Color.white;
        label.style.unityFontStyleAndWeight = FontStyle.Bold;
        row.Add(label);

        var valueLabel = new Label(defaultValue.ToString("0.00"));
        valueLabel.style.minWidth = 48;
        valueLabel.style.unityTextAlign = TextAnchor.MiddleRight;
        valueLabel.style.color = new Color(0.78f, 0.88f, 1f, 1f);
        row.Add(valueLabel);

        var slider = new Slider(min, max)
        {
            value = defaultValue
        };
        slider.style.marginTop = 8;
        slider.RegisterValueChangedCallback(evt =>
        {
            valueLabel.text = evt.newValue.ToString("0.00");
            if (_previewRunning && ReferenceEquals(_previewCaptureElement, slider))
                _previewValue = evt.newValue;
        });
        slider.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (evt.button != 0)
                return;
            StartPreview(slider, evt.pointerId, kernelName, paramSetter, slider.value);
        }, TrickleDown.TrickleDown);
        slider.RegisterCallback<PointerUpEvent>(evt =>
        {
            if (_previewRunning && ReferenceEquals(_previewCaptureElement, slider) && _previewPointerId == evt.pointerId)
                StopPreview();
        }, TrickleDown.TrickleDown);
        slider.RegisterCallback<PointerCancelEvent>(evt =>
        {
            if (_previewRunning && ReferenceEquals(_previewCaptureElement, slider) && _previewPointerId == evt.pointerId)
                StopPreview();
        }, TrickleDown.TrickleDown);
        card.Add(slider);

        var actionRow = new VisualElement();
        actionRow.style.flexDirection = FlexDirection.Row;
        actionRow.style.justifyContent = Justify.FlexEnd;
        actionRow.style.marginTop = 8;
        card.Add(actionRow);

        var applyButton = new Button(() =>
        {
            StopPreview();
            var value = slider.value;
            ApplyComputeAdjustmentAsync(
                kernelName,
                cs => paramSetter?.Invoke(cs, value),
                historyLabelFactory != null ? historyLabelFactory(value) : name).Forget();
        })
        {
            text = "应用"
        };
        applyButton.style.height = 30;
        applyButton.style.paddingLeft = 16;
        applyButton.style.paddingRight = 16;
        applyButton.style.backgroundColor = new StyleColor(new Color(0.22f, 0.58f, 0.96f, 1f));
        applyButton.style.color = Color.white;
        actionRow.Add(applyButton);

        return card;
    }

    protected async UniTask<int> ShowChoiceAsync(IReadOnlyList<Texture2D> options)
    {
        if (options == null || options.Count == 0)
            return 0;

        HideChoice();
        _choiceTcs = new UniTaskCompletionSource<int>();

        _choiceOverlay = new VisualElement();
        _choiceOverlay.style.position = Position.Absolute;
        _choiceOverlay.style.left = 0;
        _choiceOverlay.style.top = 0;
        _choiceOverlay.style.right = 0;
        _choiceOverlay.style.bottom = 0;
        _choiceOverlay.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.62f));
        _choiceOverlay.style.alignItems = Align.Center;
        _choiceOverlay.style.justifyContent = Justify.Center;

        var panel = new VisualElement();
        panel.style.width = 720;
        panel.style.maxWidth = Length.Percent(92);
        panel.style.maxHeight = 560;
        panel.style.backgroundColor = new StyleColor(new Color(0.12f, 0.13f, 0.16f, 0.98f));
        panel.style.borderTopLeftRadius = 18;
        panel.style.borderTopRightRadius = 18;
        panel.style.borderBottomLeftRadius = 18;
        panel.style.borderBottomRightRadius = 18;
        panel.style.paddingLeft = 16;
        panel.style.paddingRight = 16;
        panel.style.paddingTop = 14;
        panel.style.paddingBottom = 16;
        panel.style.flexDirection = FlexDirection.Column;
        _choiceOverlay.Add(panel);

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.marginBottom = 10;
        panel.Add(header);

        var title = new Label("请选择结果");
        title.style.flexGrow = 1;
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        header.Add(title);

        var cancelButton = CreateMiniActionButton("取消", () => ResolveChoice(0), "取消");
        cancelButton.style.height = 30;
        cancelButton.style.paddingLeft = 12;
        cancelButton.style.paddingRight = 12;
        header.Add(cancelButton);

        var scroll = new ScrollView(ScrollViewMode.Vertical);
        scroll.style.flexGrow = 1;
        panel.Add(scroll);

        var grid = new VisualElement();
        grid.style.flexDirection = FlexDirection.Row;
        grid.style.flexWrap = Wrap.Wrap;
        scroll.Add(grid);

        for (var i = 0; i < options.Count; i++)
        {
            var tex = options[i];
            var index = i;
            var card = new Button(() => ResolveChoice(index));
            card.style.width = 170;
            card.style.height = 190;
            card.style.marginLeft = 6;
            card.style.marginRight = 6;
            card.style.marginTop = 6;
            card.style.marginBottom = 6;
            card.style.paddingLeft = 6;
            card.style.paddingRight = 6;
            card.style.paddingTop = 6;
            card.style.paddingBottom = 8;
            card.style.flexDirection = FlexDirection.Column;
            card.style.backgroundColor = new StyleColor(new Color(0.18f, 0.18f, 0.22f, 1f));
            card.style.borderTopLeftRadius = 14;
            card.style.borderTopRightRadius = 14;
            card.style.borderBottomLeftRadius = 14;
            card.style.borderBottomRightRadius = 14;

            var image = new Image
            {
                image = tex,
                scaleMode = ScaleMode.ScaleToFit
            };
            image.style.flexGrow = 1;
            card.Add(image);

            var label = new Label("结果 " + (i + 1));
            label.style.marginTop = 6;
            label.style.unityTextAlign = TextAnchor.MiddleCenter;
            label.style.color = Color.white;
            card.Add(label);
            grid.Add(card);
        }

        _pageRoot.Add(_choiceOverlay);
        _choiceOverlay.BringToFront();
        return await _choiceTcs.Task;
    }

    protected void ShowToast(string text, int milliseconds = 2200)
    {
        if (_toastOverlay == null)
            return;

        _toastText.text = text ?? string.Empty;
        _toastOverlay.style.display = DisplayStyle.Flex;
        _toastOverlay.BringToFront();

        _toastHide?.Pause();
        _toastHide = _toastOverlay.schedule.Execute(() =>
        {
            if (_toastOverlay != null)
                _toastOverlay.style.display = DisplayStyle.None;
        }).StartingIn(Mathf.Max(400, milliseconds));
    }

    protected void ShowBusy(string text)
    {
        if (_busyOverlay == null)
            return;

        _busyText.text = string.IsNullOrWhiteSpace(text) ? "处理中" : text;
        _busyOverlay.style.display = DisplayStyle.Flex;
        _busyOverlay.BringToFront();
        _busyPhase = 0f;

        _busyAnim ??= _busyOverlay.schedule.Execute(() =>
        {
            if (_busyOverlay.resolvedStyle.display == DisplayStyle.None)
                return;

            var width = _busyBarTrack.resolvedStyle.width;
            if (width <= 1f)
                return;

            _busyPhase += 0.10f;
            var t = (Mathf.Sin(_busyPhase) + 1f) * 0.5f;
            var barWidth = Mathf.Clamp(width * (0.25f + 0.20f * (Mathf.Sin(_busyPhase * 1.7f) * 0.5f + 0.5f)), 50f, width);
            var x = (width - barWidth) * t;
            var alpha = 0.55f + 0.35f * (Mathf.Sin(_busyPhase * 2.3f) * 0.5f + 0.5f);

            _busyBar.style.width = barWidth;
            _busyBar.style.left = x;
            _busyBar.style.opacity = alpha;
        }).Every(16);
        _busyAnim.Resume();
    }

    protected void HideBusy()
    {
        if (_busyOverlay == null)
            return;
        _busyOverlay.style.display = DisplayStyle.None;
        _busyAnim?.Pause();
    }

    protected void ShowProgress(string title)
    {
        if (_progressOverlay == null)
            return;

        lock (_progressLock)
        {
            _progressValue01 = 0f;
            _progressText = string.Empty;
        }

        _progressTitle.text = string.IsNullOrWhiteSpace(title) ? "处理中" : title;
        _progressBar.value = 0f;
        _progressBar.title = "0%";
        _progressDetail.text = string.Empty;
        _progressOverlay.style.display = DisplayStyle.Flex;
        _progressOverlay.BringToFront();

        _progressTick ??= _progressOverlay.schedule.Execute(() =>
        {
            if (_progressOverlay.resolvedStyle.display == DisplayStyle.None)
                return;

            float progress;
            string detail;
            lock (_progressLock)
            {
                progress = _progressValue01;
                detail = _progressText;
            }

            progress = Mathf.Clamp01(progress);
            _progressBar.value = progress * 100f;
            _progressBar.title = Mathf.RoundToInt(progress * 100f) + "%";
            _progressDetail.text = detail ?? string.Empty;
        }).Every(50);
        _progressTick.Resume();
    }

    protected void HideProgress()
    {
        if (_progressOverlay == null)
            return;
        _progressOverlay.style.display = DisplayStyle.None;
        _progressTick?.Pause();
    }

    protected void SetProgress(float progress01, string text)
    {
        lock (_progressLock)
        {
            _progressValue01 = progress01;
            _progressText = text;
        }
    }

    protected static Texture2D LoadTextureFromFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return null;

        byte[] data;
        try
        {
            data = File.ReadAllBytes(filePath);
        }
        catch
        {
            return null;
        }

        var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        if (!tex.LoadImage(data, false))
        {
            UnityEngine.Object.Destroy(tex);
            return null;
        }

        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        tex.name = Path.GetFileName(filePath);
        return tex;
    }

    protected static string FormatBytes(long bytes)
    {
        if (bytes < 1024)
            return bytes + " B";
        if (bytes < 1024 * 1024)
            return (bytes / 1024f).ToString("0.0") + " KB";
        if (bytes < 1024L * 1024L * 1024L)
            return (bytes / (1024f * 1024f)).ToString("0.0") + " MB";
        return (bytes / (1024f * 1024f * 1024f)).ToString("0.0") + " GB";
    }

    protected static void OpenFolderInShell(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            return;

        try
        {
            try { Directory.CreateDirectory(directoryPath); } catch { }
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            Process.Start(new ProcessStartInfo { FileName = directoryPath, UseShellExecute = true });
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
            Process.Start(new ProcessStartInfo("open", directoryPath) { UseShellExecute = false });
#elif UNITY_STANDALONE_LINUX
            Process.Start(new ProcessStartInfo("xdg-open", directoryPath) { UseShellExecute = false });
#else
            var url = "file://" + directoryPath.Replace('\\', '/');
            Application.OpenURL(url);
#endif
        }
        catch
        {
        }
    }

    protected async UniTask<Texture2D> ReadbackTextureAsync(RenderTexture rt, int width, int height)
    {
        var tcs = new UniTaskCompletionSource<AsyncGPUReadbackRequest>();
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, req => tcs.TrySetResult(req));
        var request = await tcs.Task;
        if (request.hasError)
            return null;

        var data = request.GetData<byte>();
        var tex = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
        tex.LoadRawTextureData(data);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
    }

    private void BuildRoot()
    {
        _pageRoot = new VisualElement();
        _pageRoot.style.position = Position.Absolute;
        _pageRoot.style.left = 0;
        _pageRoot.style.top = 0;
        _pageRoot.style.right = 0;
        _pageRoot.style.bottom = 0;
        _pageRoot.style.flexDirection = FlexDirection.Column;
        _pageRoot.style.backgroundColor = new StyleColor(new Color(0.07f, 0.08f, 0.10f, 1f));
        _pageRoot.RegisterCallback<GeometryChangedEvent>(OnGeometryChanged);
        _pageRoot.RegisterCallback<KeyDownEvent>(OnRootKeyDown, TrickleDown.TrickleDown);
        _pageRoot.RegisterCallback<PointerUpEvent>(OnAnyPointerUp, TrickleDown.TrickleDown);
        _pageRoot.RegisterCallback<PointerCancelEvent>(OnAnyPointerCancel, TrickleDown.TrickleDown);

        _contentRoot = new VisualElement();
        _contentRoot.style.flexGrow = 1;
        _contentRoot.style.minHeight = 0;
        _contentRoot.style.position = Position.Relative;
        _pageRoot.Add(_contentRoot);

        BuildPage(_contentRoot);
        BuildSwitchZone(_pageRoot);
    }

    private void BuildSwitchZone(VisualElement root)
    {
        _switchZone = new VisualElement();
        _switchZone.style.height = 72;
        _switchZone.style.alignItems = Align.Center;
        _switchZone.style.justifyContent = Justify.Center;
        _switchZone.style.paddingBottom = 12;
        _switchZone.style.paddingTop = 12;
        _switchZone.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0f));
        if (UseOverlaySwitchZone)
        {
            _switchZone.style.position = Position.Absolute;
            _switchZone.style.left = 0;
            _switchZone.style.right = 0;
            _switchZone.style.bottom = 0;
        }
        else
        {
            _switchZone.style.flexShrink = 0;
        }

        _switchTrack = new VisualElement();
        _switchTrack.style.position = Position.Relative;
        _switchTrack.style.width = 160;
        _switchTrack.style.height = 28;
        _switchTrack.style.alignItems = Align.Center;
        _switchTrack.style.justifyContent = Justify.Center;
        _switchTrack.style.borderTopLeftRadius = 14;
        _switchTrack.style.borderTopRightRadius = 14;
        _switchTrack.style.borderBottomLeftRadius = 14;
        _switchTrack.style.borderBottomRightRadius = 14;
        _switchTrack.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.08f));
        _switchZone.Add(_switchTrack);

        _switchPill = new VisualElement();
        _switchPill.style.position = Position.Absolute;
        _switchPill.style.width = 118;
        _switchPill.style.height = 8;
        _switchPill.style.top = 10;
        _switchPill.style.borderTopLeftRadius = 4;
        _switchPill.style.borderTopRightRadius = 4;
        _switchPill.style.borderBottomLeftRadius = 4;
        _switchPill.style.borderBottomRightRadius = 4;
        _switchPill.style.backgroundColor = Color.white;
        _switchPill.style.opacity = 0.95f;
        _switchTrack.Add(_switchPill);

        _switchZone.RegisterCallback<PointerDownEvent>(OnSwitchPointerDown);
        _switchZone.RegisterCallback<PointerMoveEvent>(OnSwitchPointerMove);
        _switchZone.RegisterCallback<PointerUpEvent>(OnSwitchPointerUp);
        _switchZone.RegisterCallback<PointerCancelEvent>(OnSwitchPointerCancel);
        root.Add(_switchZone);
        _switchZone.schedule.Execute(ApplySwitchPillOffset);
    }

    private void OnGeometryChanged(GeometryChangedEvent evt)
    {
        var rect = evt.newRect;
        _isPortraitLayout = rect.height > rect.width;
        OnLayoutChanged(_isPortraitLayout, rect);
        ApplySwitchPillOffset();
    }

    private void OnHistorySelectionChanged(IEnumerable<object> selectedItems)
    {
        var first = selectedItems.FirstOrDefault();
        if (first is not HistoryEntry entry)
            return;
        var original = GetOriginalHistoryTexture();
        _compareView?.SetSources(entry.texture, original, entry.label);
    }

    private void RefreshHistoryUi()
    {
        _historyList?.RefreshItems();
    }

    private void ResolveChoice(int index)
    {
        if (_choiceTcs == null)
            return;
        var tcs = _choiceTcs;
        _choiceTcs = null;
        HideChoice();
        tcs.TrySetResult(index);
    }

    private void HideChoice()
    {
        if (_choiceOverlay == null)
            return;
        if (_choiceOverlay.parent != null)
            _choiceOverlay.parent.Remove(_choiceOverlay);
        _choiceOverlay = null;
        _choiceTcs = null;
    }

    private void BuildBusyOverlay(VisualElement root)
    {
        _busyOverlay = new VisualElement();
        _busyOverlay.style.position = Position.Absolute;
        _busyOverlay.style.left = 0;
        _busyOverlay.style.top = 0;
        _busyOverlay.style.right = 0;
        _busyOverlay.style.bottom = 0;
        _busyOverlay.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.38f));
        _busyOverlay.style.alignItems = Align.Center;
        _busyOverlay.style.justifyContent = Justify.Center;
        _busyOverlay.style.display = DisplayStyle.None;

        var panel = new VisualElement();
        panel.style.width = 360;
        panel.style.maxWidth = Length.Percent(84);
        panel.style.backgroundColor = new StyleColor(new Color(0.12f, 0.13f, 0.16f, 0.96f));
        panel.style.borderTopLeftRadius = 18;
        panel.style.borderTopRightRadius = 18;
        panel.style.borderBottomLeftRadius = 18;
        panel.style.borderBottomRightRadius = 18;
        panel.style.paddingLeft = 16;
        panel.style.paddingRight = 16;
        panel.style.paddingTop = 14;
        panel.style.paddingBottom = 14;
        _busyOverlay.Add(panel);

        _busyText = new Label("处理中");
        _busyText.style.color = Color.white;
        _busyText.style.marginBottom = 10;
        panel.Add(_busyText);

        _busyBarTrack = new VisualElement();
        _busyBarTrack.style.height = 12;
        _busyBarTrack.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.08f));
        _busyBarTrack.style.borderTopLeftRadius = 6;
        _busyBarTrack.style.borderTopRightRadius = 6;
        _busyBarTrack.style.borderBottomLeftRadius = 6;
        _busyBarTrack.style.borderBottomRightRadius = 6;
        _busyBarTrack.style.overflow = Overflow.Hidden;
        _busyBarTrack.style.position = Position.Relative;
        panel.Add(_busyBarTrack);

        _busyBar = new VisualElement();
        _busyBar.style.position = Position.Absolute;
        _busyBar.style.left = 0;
        _busyBar.style.top = 0;
        _busyBar.style.height = Length.Percent(100);
        _busyBar.style.width = 110;
        _busyBar.style.backgroundColor = new StyleColor(new Color(0.28f, 0.64f, 1f, 0.92f));
        _busyBarTrack.Add(_busyBar);

        root.Add(_busyOverlay);
    }

    private void BuildProgressOverlay(VisualElement root)
    {
        _progressTick?.Pause();
        _progressTick = null;
        _progressOverlay = new VisualElement();
        _progressOverlay.style.position = Position.Absolute;
        _progressOverlay.style.left = 0;
        _progressOverlay.style.top = 0;
        _progressOverlay.style.right = 0;
        _progressOverlay.style.bottom = 0;
        _progressOverlay.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.38f));
        _progressOverlay.style.alignItems = Align.Center;
        _progressOverlay.style.justifyContent = Justify.Center;
        _progressOverlay.style.display = DisplayStyle.None;

        var panel = new VisualElement();
        panel.style.width = 420;
        panel.style.maxWidth = Length.Percent(88);
        panel.style.backgroundColor = new StyleColor(new Color(0.12f, 0.13f, 0.16f, 0.96f));
        panel.style.borderTopLeftRadius = 18;
        panel.style.borderTopRightRadius = 18;
        panel.style.borderBottomLeftRadius = 18;
        panel.style.borderBottomRightRadius = 18;
        panel.style.paddingLeft = 16;
        panel.style.paddingRight = 16;
        panel.style.paddingTop = 14;
        panel.style.paddingBottom = 14;
        _progressOverlay.Add(panel);

        _progressTitle = new Label("处理中");
        _progressTitle.style.color = Color.white;
        _progressTitle.style.marginBottom = 8;
        panel.Add(_progressTitle);

        _progressBar = new ProgressBar
        {
            lowValue = 0,
            highValue = 100,
            value = 0,
            title = "0%"
        };
        _progressBar.style.height = 18;
        panel.Add(_progressBar);

        _progressDetail = new Label();
        _progressDetail.style.color = new Color(0.86f, 0.88f, 0.92f, 1f);
        _progressDetail.style.marginTop = 8;
        panel.Add(_progressDetail);

        root.Add(_progressOverlay);
    }

    private void BuildToast(VisualElement root)
    {
        _toastOverlay = new VisualElement();
        _toastOverlay.style.position = Position.Absolute;
        _toastOverlay.style.left = 0;
        _toastOverlay.style.right = 0;
        _toastOverlay.style.top = 114;
        _toastOverlay.style.alignItems = Align.Center;
        _toastOverlay.style.justifyContent = Justify.FlexStart;
        _toastOverlay.style.display = DisplayStyle.None;
        _toastOverlay.pickingMode = PickingMode.Ignore;

        var bubble = new VisualElement();
        bubble.style.backgroundColor = new StyleColor(new Color(0.08f, 0.09f, 0.12f, 0.94f));
        bubble.style.borderTopLeftRadius = 12;
        bubble.style.borderTopRightRadius = 12;
        bubble.style.borderBottomLeftRadius = 12;
        bubble.style.borderBottomRightRadius = 12;
        bubble.style.paddingLeft = 14;
        bubble.style.paddingRight = 14;
        bubble.style.paddingTop = 8;
        bubble.style.paddingBottom = 8;
        _toastOverlay.Add(bubble);

        _toastText = new Label();
        _toastText.style.color = Color.white;
        _toastText.style.whiteSpace = WhiteSpace.NoWrap;
        _toastText.style.overflow = Overflow.Hidden;
        _toastText.style.textOverflow = TextOverflow.Ellipsis;
        bubble.Add(_toastText);

        root.Add(_toastOverlay);
    }

    private void StartPreview(VisualElement captureElement, int pointerId, string kernelName, Action<ComputeShader, float> paramSetter, float initialValue)
    {
        var src = GetCurrentHistoryTexture();
        if (src == null)
            src = GetOriginalHistoryTexture();
        if (src == null || Host == null || Host.ImageProcessingCS == null)
            return;

        StopPreview();
        _previewRunning = true;
        _previewKernelName = kernelName;
        _previewParamSetter = paramSetter;
        _previewValue = initialValue;
        _previewSource = src;
        _previewPointerId = pointerId;
        _previewCaptureElement = captureElement;
        _previewRt = new RenderTexture(src.width, src.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
        {
            enableRandomWrite = true
        };
        _previewRt.Create();
        _previewCaptureElement?.CapturePointer(pointerId);
        _compareView?.SetPreview(_previewRt);
    }

    private void StopPreview()
    {
        if (!_previewRunning)
            return;

        if (_previewCaptureElement != null && _previewPointerId >= 0 && _previewCaptureElement.HasPointerCapture(_previewPointerId))
            _previewCaptureElement.ReleasePointer(_previewPointerId);

        _previewRunning = false;
        _previewKernelName = null;
        _previewParamSetter = null;
        _previewValue = 0f;
        _previewPointerId = -1;
        _previewCaptureElement = null;
        _previewSource = null;
        _compareView?.SetPreview(null);

        if (_previewRt != null)
        {
            _previewRt.Release();
            Destroy(_previewRt);
            _previewRt = null;
        }
    }

    private void OnAnyPointerUp(PointerUpEvent evt)
    {
        if (_previewRunning && evt.pointerId == _previewPointerId)
            StopPreview();
    }

    private void OnAnyPointerCancel(PointerCancelEvent evt)
    {
        if (_previewRunning && evt.pointerId == _previewPointerId)
            StopPreview();
    }

    private void OnRootKeyDown(KeyDownEvent evt)
    {
        if (_pageRoot == null)
            return;
        if (_pageRoot.focusController?.focusedElement is TextField)
            return;

        var ctrlOrCmd = evt.ctrlKey || evt.commandKey;
        if (ctrlOrCmd && !evt.shiftKey && evt.keyCode == KeyCode.Z)
        {
            UndoLastOperation();
            evt.StopPropagation();
            evt.PreventDefault();
            return;
        }

        if (evt.keyCode == KeyCode.Delete || evt.keyCode == KeyCode.Backspace)
        {
            DeleteSelectedHistoryEntry();
            evt.StopPropagation();
            evt.PreventDefault();
        }
    }

    private void Update()
    {
        if (!_previewRunning || _previewRt == null || _previewSource == null || Host == null)
            return;
        if (!Input.GetMouseButton(0))
        {
            StopPreview();
            return;
        }

        var cs = Host.ImageProcessingCS;
        if (cs == null)
            return;
        var kernel = GetKernelId(cs, _previewKernelName);
        if (kernel < 0)
            return;

        cs.SetTexture(kernel, "_Source", _previewSource);
        cs.SetTexture(kernel, "_Result", _previewRt);
        _previewParamSetter?.Invoke(cs, _previewValue);
        var gx = Mathf.CeilToInt(_previewSource.width / 8f);
        var gy = Mathf.CeilToInt(_previewSource.height / 8f);
        cs.Dispatch(kernel, gx, gy, 1);
        _compareView?.MarkDirtyRepaint();
    }

    private void OnSwitchPointerDown(PointerDownEvent evt)
    {
        if (evt.button != 0 || _switchZone == null)
            return;
        _switchDragging = true;
        _switchPointerId = evt.pointerId;
        _switchDragStart = evt.position;
        _switchZone.CapturePointer(evt.pointerId);
        evt.StopPropagation();
    }

    private void OnSwitchPointerMove(PointerMoveEvent evt)
    {
        if (!_switchDragging || _switchPointerId != evt.pointerId || _switchPill == null)
            return;

        var delta = evt.position.x - _switchDragStart.x;
        var max = Mathf.Max(24f, (_switchTrack?.resolvedStyle.width ?? 160f) * 0.18f);
        delta = Mathf.Clamp(delta, -max, max);
        ApplySwitchPillOffset(delta);
        evt.StopPropagation();
    }

    private void OnSwitchPointerUp(PointerUpEvent evt)
    {
        if (!_switchDragging || _switchPointerId != evt.pointerId)
            return;

        var delta = evt.position.x - _switchDragStart.x;
        FinishSwitchGesture(delta);
        evt.StopPropagation();
    }

    private void OnSwitchPointerCancel(PointerCancelEvent evt)
    {
        if (!_switchDragging || _switchPointerId != evt.pointerId)
            return;
        ResetSwitchPill();
        ReleaseSwitchPointer();
        evt.StopPropagation();
    }

    private void FinishSwitchGesture(float delta)
    {
        var threshold = Mathf.Max(36f, (_switchTrack?.resolvedStyle.width ?? 160f) * 0.18f);
        var direction = delta <= -threshold ? SwipeDirection.Left : (delta >= threshold ? SwipeDirection.Right : (SwipeDirection?)null);
        ResetSwitchPill();
        ReleaseSwitchPointer();

        if (!direction.HasValue)
            return;

        var target = ResolveSwipeTarget(direction.Value);
        if (target.HasValue)
            Host?.RequestPageSwitch(this, target.Value, direction.Value);
    }

    private void ReleaseSwitchPointer()
    {
        if (_switchZone != null && _switchPointerId >= 0 && _switchZone.HasPointerCapture(_switchPointerId))
            _switchZone.ReleasePointer(_switchPointerId);
        _switchDragging = false;
        _switchPointerId = -1;
    }

    private void ResetSwitchPill()
    {
        ApplySwitchPillOffset();
    }

    private void ApplySwitchPillOffset()
    {
        ApplySwitchPillOffset(0f);
    }

    private void ApplySwitchPillOffset(float dragDelta)
    {
        if (_switchPill == null)
            return;

        var trackWidth = _switchTrack != null && _switchTrack.resolvedStyle.width > 1f
            ? _switchTrack.resolvedStyle.width
            : 160f;
        var pillWidth = _switchPill.resolvedStyle.width > 1f
            ? _switchPill.resolvedStyle.width
            : 118f;
        var maxLeft = Mathf.Max(0f, trackWidth - pillWidth);
        var baseLeft = Mathf.Lerp(0f, maxLeft, Mathf.Clamp01(GetSwitchPillAlignment01()));
        _switchPill.style.left = Mathf.Clamp(baseLeft + dragDelta, 0f, maxLeft);
    }

    private static int GetKernelId(ComputeShader shader, string kernelName)
    {
        if (shader == null || string.IsNullOrWhiteSpace(kernelName))
            return -1;
        try
        {
            return shader.FindKernel(kernelName);
        }
        catch
        {
            return -1;
        }
    }

    private static Button CreateMiniActionButton(string text, Action onClick, string tooltip)
    {
        var button = new Button(onClick) { text = text };
        button.tooltip = tooltip;
        button.style.height = 28;
        button.style.minWidth = 28;
        button.style.paddingLeft = 8;
        button.style.paddingRight = 8;
        button.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.08f));
        button.style.color = Color.white;
        button.style.borderTopLeftRadius = 10;
        button.style.borderTopRightRadius = 10;
        button.style.borderBottomLeftRadius = 10;
        button.style.borderBottomRightRadius = 10;
        return button;
    }
}
