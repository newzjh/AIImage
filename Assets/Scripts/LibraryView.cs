using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

public sealed class LibraryView : BasePageView
{
    private const string PendingText = "待提取";
    private const string PendingClipText = "待接入";
    private const string EmptyText = "无";
    private const int ThumbnailMaxEdge = 640;
    private static readonly ExplorerStringComparer ExplorerComparer = new ExplorerStringComparer();

    private enum LibraryImageType
    {
        Original,
        Edited,
        Unknown
    }

    private sealed class DirectoryEntryData
    {
        public string path;
        public string displayName;
        public bool isPlaceholder;
    }

    private sealed class ThumbnailEntry
    {
        public string fullPath;
        public string fileName;
        public DateTime modifiedTime;
        public DateTime? captureTime;
        public long fileSize;
        public Texture2D thumbnail;
        public bool thumbnailLoading;
        public bool thumbnailFailed;
        public LibraryImageType type;
        public string locationText = PendingText;
        public string faceText = PendingClipText;
        public string clipText = PendingClipText;
        public string cameraText = PendingText;
        public string apertureText = PendingText;
        public bool favorite;

        public DateTime DisplayTime => captureTime ?? modifiedTime;
    }

    private sealed class ThumbnailPayload
    {
        public byte[] thumbnailBytes;
        public DateTime? captureTime;
        public string locationText;
        public string cameraText;
        public string apertureText;
    }

    private static readonly HashSet<string> ImageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".psd", ".tiff", ".tif", ".exr", ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng"
    };

    private const float LandscapeCardWidth = 294f;
    private const float LandscapeCardHeight = 387f;
    private const float LandscapeImageHeight = 294f;
    private const float PortraitCardWidth = 246f;
    private const float PortraitCardHeight = 330f;
    private const float PortraitImageHeight = 246f;

    public override AppPageId PageId => AppPageId.LibraryView;

    private PopupField<string> _drivePopup;
    private TreeView _directoryTree;
    private ScrollView _thumbnailScroll;
    private VisualElement _thumbnailGrid;
    private Label _directorySummary;
    private Toggle _showOriginalToggle;
    private Toggle _showEditedToggle;
    private Toggle _showUnknownToggle;
    private Toggle _favoritesOnlyToggle;
    private Toggle _sortTimeToggle;
    private Toggle _sortFaceToggle;
    private Toggle _sortLocationToggle;
    private Label _selectionTipsTitle;
    private Label _selectionTipsDetail;

    private readonly List<ThumbnailEntry> _thumbnailEntries = new List<ThumbnailEntry>();
    private readonly List<ThumbnailEntry> _visibleEntries = new List<ThumbnailEntry>();
    private readonly HashSet<int> _loadedDirectoryIds = new HashSet<int>();
    private readonly Dictionary<string, ThumbnailEntry> _entryByPath = new Dictionary<string, ThumbnailEntry>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Image> _imageByPath = new Dictionary<string, Image>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Label> _statusByPath = new Dictionary<string, Label>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Label> _timeLabelByPath = new Dictionary<string, Label>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, VisualElement> _cardByPath = new Dictionary<string, VisualElement>(StringComparer.OrdinalIgnoreCase);
    private bool _didInitialPathSync;
    private string _currentDriveRoot;
    private string _selectedDirectoryPath;
    private string _selectedThumbnailPath;
    private string _materializedDirectoryPath;
    private long _lastClickTicks;
    private string _lastClickPath;
    private CancellationTokenSource _thumbnailLoadCts;
    private CancellationTokenSource _directoryScanCts;
    private int _thumbnailLoadGeneration;
    private int _directoryScanGeneration;

    protected override AppPageId? ResolveSwipeTarget(SwipeDirection direction)
    {
        return direction == SwipeDirection.Right ? AppPageId.MainView2 : null;
    }

    protected override float GetSwitchPillAlignment01() => 0f;

    protected override void BuildPage(VisualElement contentRoot)
    {
        contentRoot.style.flexDirection = FlexDirection.Column;
        contentRoot.style.flexGrow = 1;
        contentRoot.style.minHeight = 0;

        contentRoot.Add(BuildTopBar());

        var body = new VisualElement();
        body.style.flexGrow = 1;
        body.style.minHeight = 0;
        body.style.flexDirection = FlexDirection.Row;
        body.style.paddingLeft = 12;
        body.style.paddingRight = 12;
        body.style.paddingTop = 8;
        contentRoot.Add(body);

        body.Add(BuildLeftPane());
        body.Add(BuildRightPane());

        BuildStandardOverlays();
    }

    protected override void OnShown()
    {
        SyncInitialSelectionFromCurrentImagePath();
        PopulateDrives();
        RestoreSelectionState();
        if (!string.IsNullOrWhiteSpace(_materializedDirectoryPath) &&
            string.Equals(_materializedDirectoryPath, _selectedDirectoryPath, StringComparison.OrdinalIgnoreCase))
        {
            ApplyFilters();
            RestoreSelectedThumbnailTips();
            ScrollToSelectedThumbnailSoon();
        }
    }

    protected override void OnBeforeDetach()
    {
        CancelDirectoryScan();
        CancelThumbnailRefresh();
    }

    protected override void OnLayoutChanged(bool isPortrait, Rect layoutRect)
    {
        if (ContentRoot == null || ContentRoot.childCount < 2)
            return;

        var body = ContentRoot[1];
        body.style.flexDirection = isPortrait ? FlexDirection.Column : FlexDirection.Row;
        ApplyFilters();
    }

    protected override void OnDestroy()
    {
        CancelDirectoryScan();
        CancelThumbnailRefresh();
        ClearThumbnailEntries(true);
        base.OnDestroy();
    }

    private void SyncInitialSelectionFromCurrentImagePath()
    {
        if (_didInitialPathSync)
            return;

        _didInitialPathSync = true;

        var currentPath = Host?.MainPage?.CurrentSourcePathForSync;
        if (string.IsNullOrWhiteSpace(currentPath) || !File.Exists(currentPath))
            return;

        try
        {
            var directory = Path.GetDirectoryName(currentPath);
            if (!string.IsNullOrWhiteSpace(directory) && Directory.Exists(directory))
                _selectedDirectoryPath = directory;
            _selectedThumbnailPath = currentPath;
        }
        catch
        {
        }
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
        bar.style.flexWrap = Wrap.Wrap;

        var title = new Label("图库");
        title.style.fontSize = 18;
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        title.style.marginRight = 14;
        bar.Add(title);

        bar.Add(CreateFilterToggle("名称", true, out _sortTimeToggle, OnSortToggleChanged));
        bar.Add(CreateFilterToggle("人脸", false, out _sortFaceToggle, OnSortToggleChanged));
        bar.Add(CreateFilterToggle("地点", false, out _sortLocationToggle, OnSortToggleChanged));
        bar.Add(CreateFilterToggle("原图", true, out _showOriginalToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("修图", true, out _showEditedToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("未知", true, out _showUnknownToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("收藏", false, out _favoritesOnlyToggle, ApplyFilters));
        return bar;
    }

    private VisualElement BuildLeftPane()
    {
        var pane = CreatePaneContainer();
        pane.style.width = 310;
        pane.style.minWidth = 260;
        pane.style.maxWidth = 360;
        pane.style.flexShrink = 0;
        pane.style.marginRight = 12;

        var driveRow = new VisualElement();
        driveRow.style.flexDirection = FlexDirection.Row;
        driveRow.style.alignItems = Align.Center;
        pane.Add(driveRow);

        var driveLabel = new Label("盘符");
        driveLabel.style.color = Color.white;
        driveLabel.style.minWidth = 42;
        driveRow.Add(driveLabel);

        _drivePopup = new PopupField<string>(new List<string> { string.Empty }, 0);
        _drivePopup.style.flexGrow = 1;
        _drivePopup.RegisterValueChangedCallback(evt => SetDrive(evt.newValue, true));
        driveRow.Add(_drivePopup);

        _directorySummary = new Label("请选择目录");
        _directorySummary.style.marginTop = 10;
        _directorySummary.style.marginBottom = 8;
        _directorySummary.style.color = new Color(0.78f, 0.84f, 0.92f, 1f);
        pane.Add(_directorySummary);

        _directoryTree = new TreeView();
        _directoryTree.style.flexGrow = 1;
        _directoryTree.style.minHeight = 0;
        _directoryTree.virtualizationMethod = CollectionVirtualizationMethod.DynamicHeight;
        _directoryTree.fixedItemHeight = 28;
        _directoryTree.selectionType = SelectionType.Single;
        _directoryTree.makeItem = () => new Label();
        _directoryTree.bindItem = (element, index) =>
        {
            var label = (Label)element;
            var data = _directoryTree.GetItemDataForIndex<DirectoryEntryData>(index);
            label.text = data.displayName;
            label.style.color = Color.white;
        };
        _directoryTree.itemExpandedChanged += OnDirectoryExpandedChanged;
        _directoryTree.selectionChanged += OnDirectorySelectionChanged;
        pane.Add(_directoryTree);

        return pane;
    }

    private VisualElement BuildRightPane()
    {
        var pane = CreatePaneContainer();
        pane.style.flexGrow = 1;

        var selectionTips = new VisualElement();
        selectionTips.style.flexShrink = 0;
        selectionTips.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.06f));
        selectionTips.style.borderTopLeftRadius = 18;
        selectionTips.style.borderTopRightRadius = 18;
        selectionTips.style.borderBottomLeftRadius = 18;
        selectionTips.style.borderBottomRightRadius = 18;
        selectionTips.style.paddingLeft = 14;
        selectionTips.style.paddingRight = 14;
        selectionTips.style.paddingTop = 10;
        selectionTips.style.paddingBottom = 10;
        selectionTips.style.marginBottom = 10;
        pane.Add(selectionTips);

        _selectionTipsTitle = new Label("缩略图信息");
        _selectionTipsTitle.style.color = Color.white;
        _selectionTipsTitle.style.unityFontStyleAndWeight = FontStyle.Bold;
        selectionTips.Add(_selectionTipsTitle);

        _selectionTipsDetail = new Label("单击缩略图查看信息，双击直接进入主编辑页。");
        _selectionTipsDetail.style.color = new Color(0.82f, 0.86f, 0.92f, 1f);
        _selectionTipsDetail.style.whiteSpace = WhiteSpace.Normal;
        _selectionTipsDetail.style.marginTop = 4;
        selectionTips.Add(_selectionTipsDetail);

        _thumbnailScroll = new ScrollView(ScrollViewMode.Vertical);
        _thumbnailScroll.style.flexGrow = 1;
        _thumbnailScroll.style.minHeight = 0;
        pane.Add(_thumbnailScroll);

        _thumbnailGrid = new VisualElement();
        _thumbnailGrid.style.flexDirection = FlexDirection.Row;
        _thumbnailGrid.style.flexWrap = Wrap.Wrap;
        _thumbnailGrid.style.alignContent = Align.FlexStart;
        _thumbnailGrid.style.paddingBottom = 8;
        _thumbnailScroll.Add(_thumbnailGrid);
        return pane;
    }

    private static VisualElement CreatePaneContainer()
    {
        var pane = new VisualElement();
        pane.style.backgroundColor = new StyleColor(new Color(0.10f, 0.11f, 0.14f, 0.95f));
        pane.style.borderTopLeftRadius = 24;
        pane.style.borderTopRightRadius = 24;
        pane.style.borderBottomLeftRadius = 24;
        pane.style.borderBottomRightRadius = 24;
        pane.style.paddingLeft = 12;
        pane.style.paddingRight = 12;
        pane.style.paddingTop = 12;
        pane.style.paddingBottom = 12;
        pane.style.flexDirection = FlexDirection.Column;
        pane.style.minHeight = 0;
        return pane;
    }

    private VisualElement CreateFilterToggle(string text, bool defaultValue, out Toggle toggle, Action onChanged)
    {
        toggle = new Toggle(text);
        var localToggle = toggle;
        localToggle.value = defaultValue;
        localToggle.style.height = 34;
        localToggle.style.marginRight = 8;
        localToggle.style.marginBottom = 6;
        localToggle.style.paddingLeft = 10;
        localToggle.style.paddingRight = 10;
        localToggle.style.paddingTop = 6;
        localToggle.style.paddingBottom = 6;
        localToggle.style.borderTopLeftRadius = 16;
        localToggle.style.borderTopRightRadius = 16;
        localToggle.style.borderBottomLeftRadius = 16;
        localToggle.style.borderBottomRightRadius = 16;
        localToggle.style.borderLeftWidth = 1;
        localToggle.style.borderRightWidth = 1;
        localToggle.style.borderTopWidth = 1;
        localToggle.style.borderBottomWidth = 1;
        localToggle.style.color = Color.white;
        ApplyToggleVisual(localToggle);

        var checkmark = localToggle.Q(className: "unity-checkmark");
        if (checkmark != null)
            checkmark.style.display = DisplayStyle.None;

        var label = localToggle.Q<Label>();
        if (label != null)
        {
            label.style.flexGrow = 1;
            label.style.unityTextAlign = TextAnchor.MiddleCenter;
            label.style.marginLeft = 0;
        }

        localToggle.RegisterValueChangedCallback(_ =>
        {
            ApplyToggleVisual(localToggle);
            onChanged?.Invoke();
        });
        return localToggle;
    }

    private static void ApplyToggleVisual(Toggle toggle)
    {
        var on = toggle.value;
        var background = on ? new Color(0.24f, 0.49f, 0.97f, 0.9f) : new Color(1f, 1f, 1f, 0.08f);
        var border = on ? new Color(0.46f, 0.68f, 1f, 1f) : new Color(1f, 1f, 1f, 0.16f);
        toggle.style.backgroundColor = new StyleColor(background);
        toggle.style.borderLeftColor = new StyleColor(border);
        toggle.style.borderRightColor = new StyleColor(border);
        toggle.style.borderTopColor = new StyleColor(border);
        toggle.style.borderBottomColor = new StyleColor(border);
    }

    private void OnSortToggleChanged()
    {
        if (_sortTimeToggle.value)
        {
            _sortFaceToggle.SetValueWithoutNotify(false);
            _sortLocationToggle.SetValueWithoutNotify(false);
            ApplyToggleVisual(_sortFaceToggle);
            ApplyToggleVisual(_sortLocationToggle);
        }
        else if (_sortFaceToggle.value)
        {
            _sortTimeToggle.SetValueWithoutNotify(false);
            _sortLocationToggle.SetValueWithoutNotify(false);
            ApplyToggleVisual(_sortTimeToggle);
            ApplyToggleVisual(_sortLocationToggle);
        }
        else if (_sortLocationToggle.value)
        {
            _sortTimeToggle.SetValueWithoutNotify(false);
            _sortFaceToggle.SetValueWithoutNotify(false);
            ApplyToggleVisual(_sortTimeToggle);
            ApplyToggleVisual(_sortFaceToggle);
        }
        else
        {
            _sortTimeToggle.SetValueWithoutNotify(true);
            ApplyToggleVisual(_sortTimeToggle);
        }

        ApplyFilters();
    }

    private void PopulateDrives()
    {
        List<string> drives;
        try
        {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            drives = Environment.GetLogicalDrives()
                .Select(Path.GetPathRoot)
                .Where(d => !string.IsNullOrWhiteSpace(d))
                .Select(d => d.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar)
                .Where(Directory.Exists)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(s => s, StringComparer.OrdinalIgnoreCase)
                .ToList();
#else
            drives = DriveInfo.GetDrives().Where(d => d.IsReady).Select(d => d.RootDirectory.FullName).ToList();
#endif
        }
        catch
        {
            drives = new List<string>();
        }

        if (drives.Count == 0)
            drives.Add(Path.GetPathRoot(Application.persistentDataPath));

        _drivePopup.choices = drives;

        var preferred = drives[0];
        if (!string.IsNullOrWhiteSpace(_selectedDirectoryPath))
        {
            var selectedRoot = Path.GetPathRoot(_selectedDirectoryPath);
            var selectedMatch = drives.FirstOrDefault(d => string.Equals(d, selectedRoot, StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrWhiteSpace(selectedMatch))
                preferred = selectedMatch;
        }
        else
        {
            var lastPath = Host?.GetLastImagePath();
            if (!string.IsNullOrWhiteSpace(lastPath))
            {
                var root = Path.GetPathRoot(lastPath);
                var match = drives.FirstOrDefault(d => string.Equals(d, root, StringComparison.OrdinalIgnoreCase));
                if (!string.IsNullOrWhiteSpace(match))
                    preferred = match;
            }
        }

        _drivePopup.SetValueWithoutNotify(preferred);
        SetDrive(preferred, string.IsNullOrWhiteSpace(_selectedDirectoryPath));
    }

    private void SetDrive(string driveRoot, bool autoSelectRoot)
    {
        if (string.IsNullOrWhiteSpace(driveRoot) || _directoryTree == null)
            return;

        _currentDriveRoot = driveRoot;
        _loadedDirectoryIds.Clear();

        var rootDisplayName = driveRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var rootItem = BuildDirectoryItem(driveRoot, rootDisplayName, 0);
        _directoryTree.SetRootItems(new[] { rootItem });
        _directoryTree.Rebuild();
        _directoryTree.schedule.Execute(() =>
        {
            _directoryTree.ExpandItem(rootItem.id);
            EnsureDirectoryChildrenLoaded(rootItem.id, driveRoot);
            if (autoSelectRoot)
                _directoryTree.SetSelectionById(rootItem.id);
        });
    }

    private TreeViewItemData<DirectoryEntryData> BuildDirectoryItem(string path, string displayName, int depth)
    {
        var children = new List<TreeViewItemData<DirectoryEntryData>>();
        if (depth < 4 && HasSubDirectoriesSafe(path))
            children.Add(BuildPlaceholderItem(path));

        return new TreeViewItemData<DirectoryEntryData>(StableId(path), new DirectoryEntryData
        {
            path = path,
            displayName = displayName,
            isPlaceholder = false
        }, children);
    }

    private TreeViewItemData<DirectoryEntryData> BuildPlaceholderItem(string path)
    {
        return new TreeViewItemData<DirectoryEntryData>(StableId(path + "|placeholder"), new DirectoryEntryData
        {
            path = path,
            displayName = "...",
            isPlaceholder = true
        });
    }

    private void OnDirectoryExpandedChanged(TreeViewExpansionChangedArgs args)
    {
        if (!_directoryTree.IsExpanded(args.id))
            return;

        var data = _directoryTree.GetItemDataForId<DirectoryEntryData>(args.id);
        if (data == null || string.IsNullOrWhiteSpace(data.path))
            return;

        EnsureDirectoryChildrenLoaded(args.id, data.path);
    }

    private void EnsureDirectoryChildrenLoaded(int parentId, string directoryPath)
    {
        if (_loadedDirectoryIds.Contains(parentId))
            return;

        _loadedDirectoryIds.Add(parentId);
        var children = EnumerateDirectoriesSafe(directoryPath, 150)
            .Select(path => BuildDirectoryItem(path, DirectoryNameFromPath(path), GetDepthForPath(path)))
            .ToList();

        RemovePlaceholderChildren(directoryPath);
        foreach (var child in children)
            _directoryTree.AddItem(child, parentId, -1, false);

        _directoryTree.RefreshItems();
    }

    private void RemovePlaceholderChildren(string directoryPath)
    {
        try
        {
            var method = _directoryTree.GetType().GetMethod("RemoveItem", new[] { typeof(int), typeof(bool) });
            method?.Invoke(_directoryTree, new object[] { StableId(directoryPath + "|placeholder"), true });
        }
        catch
        {
        }
    }

    private void OnDirectorySelectionChanged(IEnumerable<object> selectedItems)
    {
        var first = selectedItems.FirstOrDefault();
        if (first is not DirectoryEntryData entry || entry.isPlaceholder || string.IsNullOrWhiteSpace(entry.path))
            return;

        _selectedDirectoryPath = entry.path;
        _directorySummary.text = entry.path;
        RefreshThumbnailGrid(entry.path, !string.Equals(_materializedDirectoryPath, entry.path, StringComparison.OrdinalIgnoreCase));
    }

    private void RefreshThumbnailGrid(string directoryPath, bool forceRescan)
    {
        RefreshThumbnailGridAsync(directoryPath, forceRescan).Forget();
    }

    private async UniTaskVoid RefreshThumbnailGridAsync(string directoryPath, bool forceRescan)
    {
        CancelDirectoryScan();
        CancelThumbnailRefresh();
        _imageByPath.Clear();

        if (forceRescan)
        {
            ClearThumbnailEntries(true);
            _materializedDirectoryPath = directoryPath;
            ShowGridStatus(string.IsNullOrWhiteSpace(directoryPath) || !Directory.Exists(directoryPath) ? "该目录不存在。" : "正在扫描目录...");
            if (string.IsNullOrWhiteSpace(directoryPath) || !Directory.Exists(directoryPath))
            {
                _selectionTipsDetail.text = "该目录不存在。";
                return;
            }

            _directoryScanGeneration++;
            var generation = _directoryScanGeneration;
            var scanCts = new CancellationTokenSource();
            _directoryScanCts = scanCts;
            var cancellationToken = scanCts.Token;

            try
            {
                var files = await UniTask.RunOnThreadPool(
                    () => ScanDirectoryEntries(directoryPath),
                    cancellationToken: cancellationToken);

                if (cancellationToken.IsCancellationRequested ||
                    generation != _directoryScanGeneration ||
                    !string.Equals(_selectedDirectoryPath, directoryPath, StringComparison.OrdinalIgnoreCase))
                    return;

                _thumbnailEntries.AddRange(files);
                foreach (var entry in files)
                    _entryByPath[entry.fullPath] = entry;
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch
            {
                ShowGridStatus("目录扫描失败。");
                return;
            }
            finally
            {
                if (ReferenceEquals(_directoryScanCts, scanCts))
                {
                    try { _directoryScanCts.Dispose(); } catch { }
                    _directoryScanCts = null;
                }
            }
        }

        ApplyFilters();
    }

    private void ApplyFilters()
    {
        CancelThumbnailRefresh();
        if (_thumbnailGrid == null)
            return;

        _imageByPath.Clear();
        _statusByPath.Clear();
        _timeLabelByPath.Clear();
        _cardByPath.Clear();
        _visibleEntries.Clear();
        _thumbnailGrid.Clear();

        IEnumerable<ThumbnailEntry> filtered = _thumbnailEntries;
        filtered = filtered.Where(entry =>
            ((_showOriginalToggle?.value ?? true) || entry.type != LibraryImageType.Original) &&
            ((_showEditedToggle?.value ?? true) || entry.type != LibraryImageType.Edited) &&
            ((_showUnknownToggle?.value ?? true) || entry.type != LibraryImageType.Unknown));

        if (_favoritesOnlyToggle?.value == true)
            filtered = filtered.Where(entry => entry.favorite);

        if (_sortFaceToggle?.value == true)
            filtered = filtered.OrderByDescending(entry => entry.faceText, ExplorerComparer)
                .ThenBy(entry => entry.fileName, ExplorerComparer);
        else if (_sortLocationToggle?.value == true)
            filtered = filtered.OrderBy(entry => entry.locationText, ExplorerComparer)
                .ThenBy(entry => entry.fileName, ExplorerComparer);
        else
            filtered = filtered.OrderBy(entry => entry.fileName, ExplorerComparer)
                .ThenByDescending(entry => entry.DisplayTime);

        _visibleEntries.AddRange(filtered);
        if (_visibleEntries.Count == 0)
        {
            ShowGridStatus(_directoryScanCts != null ? "正在扫描目录..." : "当前筛选下没有图片。");
            RestoreSelectedThumbnailTips();
            return;
        }

        foreach (var entry in _visibleEntries)
            _thumbnailGrid.Add(BuildThumbnailCard(entry));

        StartThumbnailRefresh();
        RestoreSelectedThumbnailTips();
        ScrollToSelectedThumbnailSoon();
    }

    private VisualElement BuildThumbnailCard(ThumbnailEntry entry)
    {
        var card = new VisualElement();
        card.style.width = IsPortraitLayout ? PortraitCardWidth : LandscapeCardWidth;
        card.style.height = IsPortraitLayout ? PortraitCardHeight : LandscapeCardHeight;
        card.style.marginRight = 10;
        card.style.marginBottom = 10;
        card.style.backgroundColor = new StyleColor(new Color(0.14f, 0.15f, 0.18f, 1f));
        card.style.borderTopLeftRadius = 18;
        card.style.borderTopRightRadius = 18;
        card.style.borderBottomLeftRadius = 18;
        card.style.borderBottomRightRadius = 18;
        card.style.paddingLeft = 6;
        card.style.paddingRight = 6;
        card.style.paddingTop = 6;
        card.style.paddingBottom = 8;
        _cardByPath[entry.fullPath] = card;

        var imageHost = new VisualElement();
        imageHost.style.height = IsPortraitLayout ? PortraitImageHeight : LandscapeImageHeight;
        imageHost.style.width = Length.Percent(100);
        imageHost.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.06f));
        imageHost.style.borderTopLeftRadius = 14;
        imageHost.style.borderTopRightRadius = 14;
        imageHost.style.borderBottomLeftRadius = 14;
        imageHost.style.borderBottomRightRadius = 14;
        imageHost.style.overflow = Overflow.Hidden;
        card.Add(imageHost);

        var image = new Image();
        image.style.width = Length.Percent(100);
        image.style.height = Length.Percent(100);
        image.scaleMode = ScaleMode.ScaleAndCrop;
        image.image = entry.thumbnail;
        imageHost.Add(image);
        _imageByPath[entry.fullPath] = image;

        var statusLabel = new Label();
        statusLabel.style.position = Position.Absolute;
        statusLabel.style.left = 0;
        statusLabel.style.right = 0;
        statusLabel.style.top = 0;
        statusLabel.style.bottom = 0;
        statusLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
        statusLabel.style.color = new Color(0.82f, 0.86f, 0.92f, 1f);
        statusLabel.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.14f));
        imageHost.Add(statusLabel);
        _statusByPath[entry.fullPath] = statusLabel;
        UpdateThumbnailVisuals(entry);

        var badgeRow = new VisualElement();
        badgeRow.style.position = Position.Absolute;
        badgeRow.style.left = 12;
        badgeRow.style.top = 12;
        badgeRow.style.flexDirection = FlexDirection.Row;
        card.Add(badgeRow);

        badgeRow.Add(CreateBadge(entry.type switch
        {
            LibraryImageType.Original => "RAW",
            LibraryImageType.Edited => "EDIT",
            _ => "?"
        }, entry.type switch
        {
            LibraryImageType.Original => new Color(1f, 1f, 1f, 0.92f),
            LibraryImageType.Edited => new Color(0.18f, 0.72f, 1f, 0.92f),
            _ => new Color(0.58f, 0.58f, 0.64f, 0.92f)
        }));

        if (entry.favorite)
        {
            var favorite = CreateBadge("★", new Color(0.96f, 0.28f, 0.31f, 0.96f));
            favorite.style.marginLeft = 6;
            badgeRow.Add(favorite);
        }

        var label = new Label(entry.fileName);
        label.style.color = Color.white;
        label.style.marginTop = 8;
        label.style.whiteSpace = WhiteSpace.NoWrap;
        label.style.overflow = Overflow.Hidden;
        label.style.textOverflow = TextOverflow.Ellipsis;
        card.Add(label);

        var sub = new Label(FormatThumbnailTime(entry));
        sub.style.color = new Color(0.80f, 0.84f, 0.90f, 1f);
        sub.style.fontSize = 10;
        card.Add(sub);
        _timeLabelByPath[entry.fullPath] = sub;

        card.RegisterCallback<PointerUpEvent>(evt =>
        {
            if (evt.button != 0)
                return;
            OnThumbnailClicked(entry);
        });

        if (string.Equals(_selectedThumbnailPath, entry.fullPath, StringComparison.OrdinalIgnoreCase))
            ApplySelectedCardBorder(card, true);

        return card;
    }

    private static void ApplySelectedCardBorder(VisualElement card, bool selected)
    {
        var width = selected ? 2 : 0;
        var color = new StyleColor(new Color(0.22f, 0.60f, 1f, 1f));
        card.style.borderLeftWidth = width;
        card.style.borderRightWidth = width;
        card.style.borderTopWidth = width;
        card.style.borderBottomWidth = width;
        card.style.borderLeftColor = color;
        card.style.borderRightColor = color;
        card.style.borderTopColor = color;
        card.style.borderBottomColor = color;
    }

    private void OnThumbnailClicked(ThumbnailEntry entry)
    {
        var nowTicks = DateTime.UtcNow.Ticks;
        var isDoubleClick = string.Equals(_lastClickPath, entry.fullPath, StringComparison.OrdinalIgnoreCase) &&
                            TimeSpan.FromTicks(nowTicks - _lastClickTicks).TotalMilliseconds < 420;

        _lastClickTicks = nowTicks;
        _lastClickPath = entry.fullPath;
        _selectedThumbnailPath = entry.fullPath;
        ApplyFilters();
        UpdateSelectionTips(entry);

        if (isDoubleClick)
            Host?.OpenLibraryImageInMain(entry.fullPath);
    }

    private void RestoreSelectedThumbnailTips()
    {
        if (string.IsNullOrWhiteSpace(_selectedThumbnailPath))
            return;

        if (_entryByPath.TryGetValue(_selectedThumbnailPath, out var entry))
            UpdateSelectionTips(entry);
    }

    private void UpdateSelectionTips(ThumbnailEntry entry)
    {
        if (_selectionTipsTitle == null || _selectionTipsDetail == null)
            return;

        _selectionTipsTitle.text = entry.fileName;
        _selectionTipsDetail.text =
            $"拍摄时间: {entry.DisplayTime:yyyy-MM-dd HH:mm:ss}\n" +
            $"文件大小: {FormatBytes(entry.fileSize)}\n" +
            $"地点: {NormalizeDisplay(entry.locationText)}\n" +
            $"相机: {NormalizeDisplay(entry.cameraText)}\n" +
            $"光圈: {NormalizeDisplay(entry.apertureText)}\n" +
            $"人脸: {NormalizeDisplay(entry.faceText)}\n" +
            $"CLIP: {NormalizeDisplay(entry.clipText)}";
    }

    private void StartThumbnailRefresh()
    {
        CancelThumbnailRefresh();
        if (_visibleEntries.Count == 0)
            return;

        _thumbnailLoadGeneration++;
        _thumbnailLoadCts = new CancellationTokenSource();
        RefreshVisibleThumbnailsAsync(_thumbnailLoadGeneration, _thumbnailLoadCts.Token).Forget();
    }

    private void CancelThumbnailRefresh()
    {
        if (_thumbnailLoadCts == null)
            return;

        try { _thumbnailLoadCts.Cancel(); } catch { }
        try { _thumbnailLoadCts.Dispose(); } catch { }
        _thumbnailLoadCts = null;
    }

    private void CancelDirectoryScan()
    {
        if (_directoryScanCts == null)
            return;

        try { _directoryScanCts.Cancel(); } catch { }
        try { _directoryScanCts.Dispose(); } catch { }
        _directoryScanCts = null;
    }

    private async UniTaskVoid RefreshVisibleThumbnailsAsync(int generation, CancellationToken cancellationToken)
    {
        foreach (var entry in _visibleEntries.ToArray())
        {
            if (cancellationToken.IsCancellationRequested || generation != _thumbnailLoadGeneration)
                return;

            if (entry.thumbnail != null || entry.thumbnailLoading || entry.thumbnailFailed)
            {
                UpdateThumbnailVisuals(entry);
                continue;
            }

            entry.thumbnailLoading = true;
            UpdateThumbnailVisuals(entry);
            try
            {
                var payload = await UniTask.RunOnThreadPool(
                    () => LoadThumbnailPayload(entry.fullPath, ThumbnailMaxEdge),
                    cancellationToken: cancellationToken);

                if (cancellationToken.IsCancellationRequested || generation != _thumbnailLoadGeneration)
                    return;

                if (!string.IsNullOrWhiteSpace(payload.locationText))
                    entry.locationText = payload.locationText;
                if (!string.IsNullOrWhiteSpace(payload.cameraText))
                    entry.cameraText = payload.cameraText;
                if (!string.IsNullOrWhiteSpace(payload.apertureText))
                    entry.apertureText = payload.apertureText;
                if (payload.captureTime.HasValue)
                    entry.captureTime = payload.captureTime.Value;

                if (payload.thumbnailBytes == null || payload.thumbnailBytes.Length == 0)
                {
                    entry.thumbnailFailed = true;
                    UpdateThumbnailVisuals(entry);
                    continue;
                }

                var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                if (!texture.LoadImage(payload.thumbnailBytes, true))
                {
                    UnityEngine.Object.Destroy(texture);
                    entry.thumbnailFailed = true;
                    UpdateThumbnailVisuals(entry);
                    continue;
                }

                texture.wrapMode = TextureWrapMode.Clamp;
                texture.filterMode = FilterMode.Bilinear;
                texture.name = entry.fileName;
                entry.thumbnail = texture;
                UpdateThumbnailVisuals(entry);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch
            {
                entry.thumbnailFailed = true;
                UpdateThumbnailVisuals(entry);
            }
            finally
            {
                entry.thumbnailLoading = false;
                UpdateThumbnailVisuals(entry);
            }

            await UniTask.DelayFrame(1, cancellationToken: cancellationToken);
        }
    }

    private void UpdateThumbnailVisuals(ThumbnailEntry entry)
    {
        if (_imageByPath.TryGetValue(entry.fullPath, out var image))
            image.image = entry.thumbnail;

        if (_statusByPath.TryGetValue(entry.fullPath, out var status))
        {
            if (entry.thumbnail != null)
            {
                status.style.display = DisplayStyle.None;
            }
            else
            {
                status.text = entry.thumbnailFailed ? "无法预览" : (entry.thumbnailLoading ? "加载中..." : "等待加载");
                status.style.display = DisplayStyle.Flex;
            }
        }

        if (_timeLabelByPath.TryGetValue(entry.fullPath, out var timeLabel))
            timeLabel.text = FormatThumbnailTime(entry);

        if (string.Equals(_selectedThumbnailPath, entry.fullPath, StringComparison.OrdinalIgnoreCase))
            UpdateSelectionTips(entry);
    }

    private void RestoreSelectionState()
    {
        if (string.IsNullOrWhiteSpace(_selectedDirectoryPath) || !Directory.Exists(_selectedDirectoryPath))
            return;

        try
        {
            var root = Path.GetPathRoot(_selectedDirectoryPath);
            if (!string.IsNullOrWhiteSpace(root) && !string.Equals(_currentDriveRoot, root, StringComparison.OrdinalIgnoreCase))
            {
                _drivePopup.SetValueWithoutNotify(root);
                SetDrive(root, false);
            }

            ExpandToDirectory(_selectedDirectoryPath);
            _directoryTree.schedule.Execute(() => _directoryTree.SetSelectionById(StableId(_selectedDirectoryPath)));
        }
        catch
        {
        }
    }

    private void ExpandToDirectory(string directoryPath)
    {
        if (_directoryTree == null || string.IsNullOrWhiteSpace(directoryPath) || string.IsNullOrWhiteSpace(_currentDriveRoot))
            return;

        string full;
        string root;
        try
        {
            full = Path.GetFullPath(directoryPath);
            root = Path.GetFullPath(_currentDriveRoot);
        }
        catch
        {
            return;
        }

        if (!root.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal))
            root += Path.DirectorySeparatorChar;
        if (!full.StartsWith(root, StringComparison.OrdinalIgnoreCase))
            return;

        var relative = full.Substring(root.Length).Trim(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var current = root;
        if (current.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal))
            current = current.TrimEnd(Path.DirectorySeparatorChar);
        current += Path.DirectorySeparatorChar;

        var rootId = StableId(current);
        _directoryTree.ExpandItem(rootId);
        EnsureDirectoryChildrenLoaded(rootId, current);

        if (string.IsNullOrEmpty(relative))
            return;

        var parts = relative.Split(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i < parts.Length; i++)
        {
            current = Path.Combine(current, parts[i]);
            var id = StableId(current);
            _directoryTree.ExpandItem(id);
            EnsureDirectoryChildrenLoaded(id, current);
        }
    }

    private void ScrollToSelectedThumbnailSoon()
    {
        if (_thumbnailScroll == null || string.IsNullOrWhiteSpace(_selectedThumbnailPath))
            return;

        _thumbnailScroll.schedule.Execute(() =>
        {
            _thumbnailScroll.schedule.Execute(ScrollToSelectedThumbnailNow);
        });
    }

    private void ScrollToSelectedThumbnailNow()
    {
        if (_thumbnailScroll == null ||
            string.IsNullOrWhiteSpace(_selectedThumbnailPath) ||
            !_cardByPath.TryGetValue(_selectedThumbnailPath, out var card) ||
            card == null)
            return;

        var viewportHeight = _thumbnailScroll.contentViewport.resolvedStyle.height;
        if (viewportHeight <= 1f)
            return;

        var containerTop = _thumbnailScroll.contentContainer.worldBound.yMin;
        var cardTop = card.worldBound.yMin - containerTop;
        var cardBottom = card.worldBound.yMax - containerTop;
        var current = _thumbnailScroll.scrollOffset.y;
        const float padding = 24f;
        var target = current;

        if (cardTop < current + padding)
            target = Mathf.Max(0f, cardTop - padding);
        else if (cardBottom > current + viewportHeight - padding)
            target = Mathf.Max(0f, cardBottom - viewportHeight + padding);

        _thumbnailScroll.scrollOffset = new Vector2(_thumbnailScroll.scrollOffset.x, target);
    }

    private void ShowGridStatus(string text)
    {
        if (_thumbnailGrid == null)
            return;

        _thumbnailGrid.Clear();
        _visibleEntries.Clear();
        _cardByPath.Clear();
        _imageByPath.Clear();
        _statusByPath.Clear();
        _timeLabelByPath.Clear();

        var label = new Label(text);
        label.style.color = new Color(0.82f, 0.86f, 0.92f, 1f);
        label.style.unityTextAlign = TextAnchor.MiddleCenter;
        label.style.width = Length.Percent(100);
        label.style.marginTop = 36;
        label.style.marginBottom = 36;
        _thumbnailGrid.Add(label);
    }

    private void ClearThumbnailEntries(bool destroyTextures)
    {
        if (destroyTextures)
        {
            foreach (var entry in _thumbnailEntries)
            {
                if (entry.thumbnail != null)
                    UnityEngine.Object.Destroy(entry.thumbnail);
                entry.thumbnail = null;
            }
        }

        _thumbnailEntries.Clear();
        _visibleEntries.Clear();
        _entryByPath.Clear();
    }

    private static List<ThumbnailEntry> ScanDirectoryEntries(string directoryPath)
    {
        try
        {
            return Directory.EnumerateFiles(directoryPath)
                .Where(IsImageFile)
                .Select(path =>
                {
                    var info = new FileInfo(path);
                    var name = info.Name;
                    return new ThumbnailEntry
                    {
                        fullPath = path,
                        fileName = name,
                        modifiedTime = info.LastWriteTime,
                        fileSize = info.Exists ? info.Length : 0,
                        type = GuessType(name),
                        favorite = name.Contains("fav", StringComparison.OrdinalIgnoreCase) ||
                                   name.Contains("star", StringComparison.OrdinalIgnoreCase)
                    };
                })
                .OrderBy(entry => entry.fileName, ExplorerComparer)
                .ToList();
        }
        catch
        {
            return new List<ThumbnailEntry>();
        }
    }

    private static ThumbnailPayload LoadThumbnailPayload(string filePath, int maxEdge)
    {
        if (TryBuildThumbnailPayloadWithSystemDrawing(filePath, maxEdge, out var payload))
            return payload;

        return new ThumbnailPayload
        {
            thumbnailBytes = LoadImageBytes(filePath)
        };
    }

    private static bool TryBuildThumbnailPayloadWithSystemDrawing(string filePath, int maxEdge, out ThumbnailPayload payload)
    {
        payload = new ThumbnailPayload();
        var imageType = ResolveSystemDrawingType("System.Drawing.Image");
        var bitmapType = ResolveSystemDrawingType("System.Drawing.Bitmap");
        var sizeType = ResolveSystemDrawingType("System.Drawing.Size");
        var imageFormatType = ResolveSystemDrawingType("System.Drawing.Imaging.ImageFormat");
        if (imageType == null || bitmapType == null || sizeType == null || imageFormatType == null)
            return false;

        var fromStream = imageType.GetMethod("FromStream", new[] { typeof(Stream) });
        if (fromStream == null)
            return false;

        Stream stream = null;
        object image = null;
        object bitmap = null;
        try
        {
            stream = File.OpenRead(filePath);
            image = fromStream.Invoke(null, new object[] { stream });
            if (image == null)
                return false;

            payload.captureTime = ReadExifDate(image, imageType, 0x9003) ?? ReadExifDate(image, imageType, 0x0132);
            payload.locationText = ReadGpsLocation(image, imageType);
            payload.cameraText = ReadCameraModel(image, imageType);
            payload.apertureText = ReadAperture(image, imageType);

            var width = Convert.ToInt32(imageType.GetProperty("Width")?.GetValue(image));
            var height = Convert.ToInt32(imageType.GetProperty("Height")?.GetValue(image));
            if (width <= 0 || height <= 0)
                return false;

            var scale = Mathf.Min(1f, maxEdge / (float)Mathf.Max(width, height));
            var thumbWidth = Mathf.Max(1, Mathf.RoundToInt(width * scale));
            var thumbHeight = Mathf.Max(1, Mathf.RoundToInt(height * scale));
            var size = Activator.CreateInstance(sizeType, thumbWidth, thumbHeight);
            bitmap = Activator.CreateInstance(bitmapType, image, size);
            if (bitmap == null)
                return false;

            var pngFormat = imageFormatType.GetProperty("Png", BindingFlags.Public | BindingFlags.Static)?.GetValue(null);
            var saveMethod = bitmapType.GetMethod("Save", new[] { typeof(Stream), imageFormatType });
            if (pngFormat == null || saveMethod == null)
                return false;

            using var ms = new MemoryStream();
            saveMethod.Invoke(bitmap, new[] { ms, pngFormat });
            payload.thumbnailBytes = ms.ToArray();
            return payload.thumbnailBytes != null && payload.thumbnailBytes.Length > 0;
        }
        catch
        {
            return false;
        }
        finally
        {
            TryDispose(bitmap);
            TryDispose(image);
            stream?.Dispose();
        }
    }

    private static Type ResolveSystemDrawingType(string fullName)
    {
        return Type.GetType(fullName + ", System.Drawing") ??
               Type.GetType(fullName + ", System.Drawing.Common");
    }

    private static void TryDispose(object obj)
    {
        if (obj == null)
            return;

        try
        {
            obj.GetType().GetMethod("Dispose", Type.EmptyTypes)?.Invoke(obj, null);
        }
        catch
        {
        }
    }

    private static object TryGetPropertyItem(object image, Type imageType, int id)
    {
        try
        {
            return imageType.GetMethod("GetPropertyItem", new[] { typeof(int) })?.Invoke(image, new object[] { id });
        }
        catch
        {
            return null;
        }
    }

    private static byte[] GetPropertyItemBytes(object propertyItem)
    {
        return propertyItem?.GetType().GetProperty("Value")?.GetValue(propertyItem) as byte[];
    }

    private static string ReadCameraModel(object image, Type imageType)
    {
        var model = ReadExifAscii(TryGetPropertyItem(image, imageType, 0x0110));
        if (!string.IsNullOrWhiteSpace(model))
            return model;

        var make = ReadExifAscii(TryGetPropertyItem(image, imageType, 0x010F));
        return string.IsNullOrWhiteSpace(make) ? PendingText : make;
    }

    private static string ReadAperture(object image, Type imageType)
    {
        var value = ReadRational(TryGetPropertyItem(image, imageType, 0x829D));
        if (value.HasValue && value.Value > 0.01)
            return $"f/{value.Value:0.0#}";
        return PendingText;
    }

    private static string ReadGpsLocation(object image, Type imageType)
    {
        var latValues = ReadRationalArray(TryGetPropertyItem(image, imageType, 0x0002));
        var lonValues = ReadRationalArray(TryGetPropertyItem(image, imageType, 0x0004));
        if (latValues == null || lonValues == null || latValues.Length < 3 || lonValues.Length < 3)
            return PendingText;

        var lat = latValues[0] + latValues[1] / 60d + latValues[2] / 3600d;
        var lon = lonValues[0] + lonValues[1] / 60d + lonValues[2] / 3600d;
        var latRef = ReadExifAscii(TryGetPropertyItem(image, imageType, 0x0001));
        var lonRef = ReadExifAscii(TryGetPropertyItem(image, imageType, 0x0003));
        if (string.Equals(latRef, "S", StringComparison.OrdinalIgnoreCase))
            lat = -lat;
        if (string.Equals(lonRef, "W", StringComparison.OrdinalIgnoreCase))
            lon = -lon;

        return $"GPS {lat:0.0000}, {lon:0.0000}";
    }

    private static DateTime? ReadExifDate(object image, Type imageType, int propertyId)
    {
        var text = ReadExifAscii(TryGetPropertyItem(image, imageType, propertyId));
        if (string.IsNullOrWhiteSpace(text))
            return null;

        if (DateTime.TryParseExact(
                text.Trim(),
                new[] { "yyyy:MM:dd HH:mm:ss", "yyyy-MM-dd HH:mm:ss" },
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out var value))
            return value;

        return null;
    }

    private static string ReadExifAscii(object propertyItem)
    {
        var bytes = GetPropertyItemBytes(propertyItem);
        if (bytes == null || bytes.Length == 0)
            return null;

        var text = System.Text.Encoding.ASCII.GetString(bytes);
        return text.Trim('\0', ' ', '\t', '\r', '\n');
    }

    private static double? ReadRational(object propertyItem)
    {
        var bytes = GetPropertyItemBytes(propertyItem);
        if (bytes == null || bytes.Length < 8)
            return null;

        var numerator = BitConverter.ToUInt32(bytes, 0);
        var denominator = BitConverter.ToUInt32(bytes, 4);
        if (denominator == 0)
            return null;
        return numerator / (double)denominator;
    }

    private static double[] ReadRationalArray(object propertyItem)
    {
        var bytes = GetPropertyItemBytes(propertyItem);
        if (bytes == null || bytes.Length < 8 || bytes.Length % 8 != 0)
            return null;

        var result = new double[bytes.Length / 8];
        for (var i = 0; i < result.Length; i++)
        {
            var numerator = BitConverter.ToUInt32(bytes, i * 8);
            var denominator = BitConverter.ToUInt32(bytes, i * 8 + 4);
            result[i] = denominator == 0 ? 0d : numerator / (double)denominator;
        }

        return result;
    }

    private static byte[] LoadImageBytes(string filePath)
    {
        try
        {
            return File.ReadAllBytes(filePath);
        }
        catch
        {
            return null;
        }
    }

    private static Label CreateBadge(string text, Color background)
    {
        var badge = new Label(text);
        badge.style.paddingLeft = 8;
        badge.style.paddingRight = 8;
        badge.style.paddingTop = 3;
        badge.style.paddingBottom = 3;
        badge.style.backgroundColor = new StyleColor(background);
        badge.style.color = background.grayscale > 0.7f ? Color.black : Color.white;
        badge.style.borderTopLeftRadius = 10;
        badge.style.borderTopRightRadius = 10;
        badge.style.borderBottomLeftRadius = 10;
        badge.style.borderBottomRightRadius = 10;
        badge.style.unityFontStyleAndWeight = FontStyle.Bold;
        badge.style.fontSize = 10;
        return badge;
    }

    private static LibraryImageType GuessType(string fileName)
    {
        var lower = (fileName ?? string.Empty).ToLowerInvariant();
        if (lower.Contains("raw") || lower.EndsWith(".cr2") || lower.EndsWith(".cr3") || lower.EndsWith(".nef") || lower.EndsWith(".arw") || lower.EndsWith(".dng"))
            return LibraryImageType.Original;
        if (lower.Contains("edit") || lower.Contains("retouch") || lower.Contains("result") || lower.Contains("output") || lower.Contains("fix"))
            return LibraryImageType.Edited;
        return LibraryImageType.Unknown;
    }

    private static bool IsImageFile(string filePath)
    {
        var ext = Path.GetExtension(filePath);
        return !string.IsNullOrWhiteSpace(ext) && ImageExtensions.Contains(ext);
    }

    private int GetDepthForPath(string path)
    {
        if (string.IsNullOrWhiteSpace(_currentDriveRoot) || string.IsNullOrWhiteSpace(path))
            return 0;

        try
        {
            var relative = Path.GetRelativePath(_currentDriveRoot, path);
            return relative.Split(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries).Length;
        }
        catch
        {
            return 0;
        }
    }

    private static string DirectoryNameFromPath(string directoryPath)
    {
        var name = Path.GetFileName(directoryPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        return string.IsNullOrWhiteSpace(name) ? directoryPath : name;
    }

    private static bool HasSubDirectoriesSafe(string path)
    {
        return EnumerateDirectoriesSafe(path, 1).Any();
    }

    private static IEnumerable<string> EnumerateDirectoriesSafe(string path, int maxCount)
    {
        if (string.IsNullOrWhiteSpace(path))
            return Array.Empty<string>();

        try
        {
            return Directory.EnumerateDirectories(path)
                .OrderBy(DirectoryNameFromPath, ExplorerComparer)
                .ThenBy(path => path, StringComparer.OrdinalIgnoreCase)
                .Take(Mathf.Max(1, maxCount))
                .ToArray();
        }
        catch
        {
            return Array.Empty<string>();
        }
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes <= 0)
            return "0 B";
        string[] units = { "B", "KB", "MB", "GB", "TB" };
        var value = bytes;
        var unitIndex = 0;
        double display = value;
        while (display >= 1024 && unitIndex < units.Length - 1)
        {
            display /= 1024d;
            unitIndex++;
        }
        return $"{display:0.##} {units[unitIndex]}";
    }

    private static string NormalizeDisplay(string text)
    {
        return string.IsNullOrWhiteSpace(text) ? EmptyText : text;
    }

    private static string FormatThumbnailTime(ThumbnailEntry entry)
    {
        return entry.DisplayTime.ToString("yyyy-MM-dd HH:mm");
    }

    private static int StableId(string text)
    {
        if (string.IsNullOrEmpty(text))
            return 1;

        unchecked
        {
            const uint offset = 2166136261;
            const uint prime = 16777619;
            var hash = offset;
            for (var i = 0; i < text.Length; i++)
            {
                var c = text[i];
                hash ^= (byte)(c & 0xFF);
                hash *= prime;
                hash ^= (byte)((c >> 8) & 0xFF);
                hash *= prime;
            }

            var id = (int)(hash & 0x7FFFFFFF);
            return id == 0 ? 1 : id;
        }
    }

    private sealed class ExplorerStringComparer : IComparer<string>
    {
        private readonly CompareInfo _compareInfo = CompareInfo.GetCompareInfo("zh-CN");

        [DllImport("shlwapi.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
        private static extern int StrCmpLogicalW(string left, string right);

        public int Compare(string x, string y)
        {
            if (ReferenceEquals(x, y))
                return 0;
            if (x == null)
                return -1;
            if (y == null)
                return 1;

            if (IsWindows())
            {
                try
                {
                    var logicalResult = StrCmpLogicalW(x, y);
                    if (logicalResult != 0)
                        return logicalResult;
                }
                catch
                {
                }
            }

            var fallback = _compareInfo.Compare(x, y, CompareOptions.IgnoreCase | CompareOptions.StringSort);
            if (fallback != 0)
                return fallback;
            return string.CompareOrdinal(x, y);
        }

        private static bool IsWindows()
        {
            var platform = Environment.OSVersion.Platform;
            return platform == PlatformID.Win32NT ||
                   platform == PlatformID.Win32S ||
                   platform == PlatformID.Win32Windows ||
                   platform == PlatformID.WinCE;
        }
    }
}
