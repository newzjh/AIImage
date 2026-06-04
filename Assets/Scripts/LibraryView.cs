using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

public sealed class LibraryView : BasePageView
{
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
        public long fileSize;
        public Texture2D thumbnail;
        public LibraryImageType type;
        public string locationText;
        public string faceText;
        public string clipText;
        public bool favorite;
    }

    private static readonly HashSet<string> ImageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".psd", ".tiff", ".tif", ".exr", ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng"
    };

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
    private VisualElement _selectionTips;

    private readonly List<ThumbnailEntry> _thumbnailEntries = new List<ThumbnailEntry>();
    private readonly HashSet<int> _loadedDirectoryIds = new HashSet<int>();
    private string _currentDriveRoot;
    private string _selectedDirectoryPath;
    private string _selectedThumbnailPath;
    private long _lastClickTicks;
    private string _lastClickPath;

    protected override AppPageId? ResolveSwipeTarget(SwipeDirection direction)
    {
        return direction == SwipeDirection.Right ? AppPageId.MainView2 : null;
    }

    protected override void BuildPage(VisualElement contentRoot)
    {
        contentRoot.style.flexDirection = FlexDirection.Column;
        contentRoot.style.flexGrow = 1;
        contentRoot.style.minHeight = 0;

        var topBar = BuildTopBar();
        contentRoot.Add(topBar);

        var body = new VisualElement();
        body.style.flexGrow = 1;
        body.style.minHeight = 0;
        body.style.flexDirection = FlexDirection.Row;
        body.style.paddingLeft = 12;
        body.style.paddingRight = 12;
        body.style.paddingTop = 8;
        body.style.paddingBottom = 0;
        contentRoot.Add(body);

        var leftPane = BuildLeftPane();
        body.Add(leftPane);

        var rightPane = BuildRightPane();
        body.Add(rightPane);

        BuildStandardOverlays();
    }

    protected override void OnShown()
    {
        PopulateDrives();
        if (!string.IsNullOrWhiteSpace(_selectedDirectoryPath))
            RefreshThumbnailGrid(_selectedDirectoryPath);
    }

    protected override void OnLayoutChanged(bool isPortrait, Rect layoutRect)
    {
        if (ContentRoot == null)
            return;
        if (ContentRoot.childCount < 2)
            return;
        var body = ContentRoot[1];
        body.style.flexDirection = isPortrait ? FlexDirection.Column : FlexDirection.Row;
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

        var title = new Label("图库");
        title.style.fontSize = 18;
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        title.style.marginRight = 14;
        bar.Add(title);

        bar.Add(CreateFilterToggle("时间", true, out _sortTimeToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("人脸", false, out _sortFaceToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("地点", false, out _sortLocationToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("原图", true, out _showOriginalToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("修图", true, out _showEditedToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("未知", true, out _showUnknownToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("收藏", false, out _favoritesOnlyToggle, ApplyFilters));
        return bar;
    }

    private VisualElement BuildLeftPane()
    {
        var pane = new VisualElement();
        pane.style.width = 310;
        pane.style.minWidth = 260;
        pane.style.maxWidth = 360;
        pane.style.flexShrink = 0;
        pane.style.marginRight = 12;
        pane.style.backgroundColor = new StyleColor(new Color(0.11f, 0.12f, 0.15f, 0.95f));
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
        _drivePopup.RegisterValueChangedCallback(evt => SetDrive(evt.newValue));
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
        var pane = new VisualElement();
        pane.style.flexGrow = 1;
        pane.style.minHeight = 0;
        pane.style.backgroundColor = new StyleColor(new Color(0.09f, 0.10f, 0.13f, 0.95f));
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

        _selectionTips = new VisualElement();
        _selectionTips.style.flexShrink = 0;
        _selectionTips.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.06f));
        _selectionTips.style.borderTopLeftRadius = 18;
        _selectionTips.style.borderTopRightRadius = 18;
        _selectionTips.style.borderBottomLeftRadius = 18;
        _selectionTips.style.borderBottomRightRadius = 18;
        _selectionTips.style.paddingLeft = 14;
        _selectionTips.style.paddingRight = 14;
        _selectionTips.style.paddingTop = 10;
        _selectionTips.style.paddingBottom = 10;
        _selectionTips.style.marginBottom = 10;
        pane.Add(_selectionTips);

        _selectionTipsTitle = new Label("缩略图信息");
        _selectionTipsTitle.style.color = Color.white;
        _selectionTipsTitle.style.unityFontStyleAndWeight = FontStyle.Bold;
        _selectionTips.Add(_selectionTipsTitle);

        _selectionTipsDetail = new Label("单击缩略图查看拍摄时间、地点、大小和 CLIP 分类信息；双击直接进入编辑页。");
        _selectionTipsDetail.style.color = new Color(0.82f, 0.86f, 0.92f, 1f);
        _selectionTipsDetail.style.whiteSpace = WhiteSpace.Normal;
        _selectionTipsDetail.style.marginTop = 4;
        _selectionTips.Add(_selectionTipsDetail);

        _thumbnailScroll = new ScrollView(ScrollViewMode.Vertical);
        _thumbnailScroll.style.flexGrow = 1;
        _thumbnailScroll.style.minHeight = 0;
        pane.Add(_thumbnailScroll);

        _thumbnailGrid = new VisualElement();
        _thumbnailGrid.style.flexDirection = FlexDirection.Row;
        _thumbnailGrid.style.flexWrap = Wrap.Wrap;
        _thumbnailGrid.style.alignContent = Align.FlexStart;
        _thumbnailScroll.Add(_thumbnailGrid);
        return pane;
    }

    private VisualElement CreateFilterToggle(string text, bool defaultValue, out Toggle toggle, Action onChanged)
    {
        toggle = new Toggle(text);
        toggle.value = defaultValue;
        toggle.style.height = 34;
        toggle.style.marginRight = 8;
        toggle.style.paddingLeft = 10;
        toggle.style.paddingRight = 10;
        toggle.style.paddingTop = 6;
        toggle.style.paddingBottom = 6;
        toggle.style.borderTopLeftRadius = 16;
        toggle.style.borderTopRightRadius = 16;
        toggle.style.borderBottomLeftRadius = 16;
        toggle.style.borderBottomRightRadius = 16;
        toggle.style.borderLeftWidth = 1;
        toggle.style.borderRightWidth = 1;
        toggle.style.borderTopWidth = 1;
        toggle.style.borderBottomWidth = 1;
        toggle.style.unityTextAlign = TextAnchor.MiddleCenter;
        toggle.style.color = Color.white;
        toggle.style.backgroundColor = new StyleColor(defaultValue ? new Color(0.24f, 0.49f, 0.97f, 0.9f) : new Color(1f, 1f, 1f, 0.08f));
        toggle.style.borderLeftColor = new StyleColor(defaultValue ? new Color(0.46f, 0.68f, 1f, 1f) : new Color(1f, 1f, 1f, 0.16f));
        toggle.style.borderRightColor = toggle.style.borderLeftColor;
        toggle.style.borderTopColor = toggle.style.borderLeftColor;
        toggle.style.borderBottomColor = toggle.style.borderLeftColor;

        var checkmark = toggle.Q(className: "unity-checkmark");
        if (checkmark != null)
        {
            checkmark.style.display = DisplayStyle.None;
        }

        var label = toggle.Q<Label>();
        if (label != null)
        {
            label.style.flexGrow = 1;
            label.style.unityTextAlign = TextAnchor.MiddleCenter;
            label.style.marginLeft = 0;
        }

        toggle.RegisterValueChangedCallback(_ => onChanged?.Invoke());
        return toggle;
    }

    private void PopulateDrives()
    {
        List<string> drives;
        try
        {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            drives = Environment.GetLogicalDrives()
                .Select(d => Path.GetPathRoot(d))
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
        var lastPath = Host?.GetLastImagePath();
        if (!string.IsNullOrWhiteSpace(lastPath))
        {
            var root = Path.GetPathRoot(lastPath);
            var match = drives.FirstOrDefault(d => string.Equals(d, root, StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrWhiteSpace(match))
                preferred = match;
        }

        _drivePopup.SetValueWithoutNotify(preferred);
        SetDrive(preferred);
    }

    private void SetDrive(string driveRoot)
    {
        if (string.IsNullOrWhiteSpace(driveRoot))
            return;
        _currentDriveRoot = driveRoot;
        _loadedDirectoryIds.Clear();

        var rootItem = BuildDirectoryItem(driveRoot, driveRoot.TrimEnd(Path.DirectorySeparatorChar), 0);
        _directoryTree.SetRootItems(new[] { rootItem });
        _directoryTree.Rebuild();
        _directoryTree.schedule.Execute(() =>
        {
            _directoryTree.ExpandItem(rootItem.id);
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

        RemovePlaceholderChildren(parentId);
        foreach (var child in children)
            _directoryTree.AddItem(child, parentId, -1, false);
        _directoryTree.RefreshItems();
    }

    private void RemovePlaceholderChildren(int parentId)
    {
        var item = _directoryTree.GetItemDataForId<DirectoryEntryData>(parentId);
        if (item == null)
            return;
        try
        {
            var method = _directoryTree.GetType().GetMethod("RemoveItem", new[] { typeof(int), typeof(bool) });
            var placeholderId = StableId(item.path + "|placeholder");
            method?.Invoke(_directoryTree, new object[] { placeholderId, true });
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
        RefreshThumbnailGrid(entry.path);
    }

    private void RefreshThumbnailGrid(string directoryPath)
    {
        _thumbnailEntries.Clear();
        _thumbnailGrid.Clear();

        if (string.IsNullOrWhiteSpace(directoryPath) || !Directory.Exists(directoryPath))
        {
            _selectionTipsDetail.text = "该目录不存在。";
            return;
        }

        try
        {
            var files = Directory.EnumerateFiles(directoryPath)
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
                        locationText = "地点待接入",
                        faceText = "人脸待接入",
                        clipText = "CLIP 分类待接入",
                        favorite = name.Contains("fav", StringComparison.OrdinalIgnoreCase) || name.Contains("star", StringComparison.OrdinalIgnoreCase)
                    };
                })
                .OrderByDescending(entry => entry.modifiedTime)
                .ToList();
            _thumbnailEntries.AddRange(files);
        }
        catch
        {
        }

        ApplyFilters();
    }

    private void ApplyFilters()
    {
        _thumbnailGrid.Clear();

        IEnumerable<ThumbnailEntry> filtered = _thumbnailEntries;
        filtered = filtered.Where(entry =>
            ((_showOriginalToggle?.value ?? true) || entry.type != LibraryImageType.Original) &&
            ((_showEditedToggle?.value ?? true) || entry.type != LibraryImageType.Edited) &&
            ((_showUnknownToggle?.value ?? true) || entry.type != LibraryImageType.Unknown));

        if (_favoritesOnlyToggle?.value == true)
            filtered = filtered.Where(entry => entry.favorite);

        if (_sortFaceToggle?.value == true)
            filtered = filtered.OrderByDescending(entry => entry.faceText);
        else if (_sortLocationToggle?.value == true)
            filtered = filtered.OrderBy(entry => entry.locationText);
        else
            filtered = filtered.OrderByDescending(entry => entry.modifiedTime);

        foreach (var entry in filtered)
            _thumbnailGrid.Add(BuildThumbnailCard(entry));
    }

    private VisualElement BuildThumbnailCard(ThumbnailEntry entry)
    {
        var card = new VisualElement();
        card.style.width = IsPortraitLayout ? 164 : 196;
        card.style.height = IsPortraitLayout ? 220 : 258;
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

        var image = new Image();
        image.style.height = IsPortraitLayout ? 156 : 192;
        image.style.width = Length.Percent(100);
        image.scaleMode = ScaleMode.ScaleAndCrop;
        image.style.borderTopLeftRadius = 14;
        image.style.borderTopRightRadius = 14;
        image.style.borderBottomLeftRadius = 14;
        image.style.borderBottomRightRadius = 14;
        card.Add(image);

        entry.thumbnail ??= Host?.LoadTexture(entry.fullPath, false);
        image.image = entry.thumbnail;

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

        var sub = new Label(entry.modifiedTime.ToString("yyyy-MM-dd HH:mm"));
        sub.style.color = new Color(0.80f, 0.84f, 0.90f, 1f);
        sub.style.fontSize = 10;
        card.Add(sub);

        card.RegisterCallback<PointerUpEvent>(evt =>
        {
            if (evt.button != 0)
                return;
            OnThumbnailClicked(entry);
        });

        if (string.Equals(_selectedThumbnailPath, entry.fullPath, StringComparison.OrdinalIgnoreCase))
        {
            card.style.borderLeftWidth = 2;
            card.style.borderRightWidth = 2;
            card.style.borderTopWidth = 2;
            card.style.borderBottomWidth = 2;
            card.style.borderLeftColor = new StyleColor(new Color(0.22f, 0.60f, 1f, 1f));
            card.style.borderRightColor = new StyleColor(new Color(0.22f, 0.60f, 1f, 1f));
            card.style.borderTopColor = new StyleColor(new Color(0.22f, 0.60f, 1f, 1f));
            card.style.borderBottomColor = new StyleColor(new Color(0.22f, 0.60f, 1f, 1f));
        }
        return card;
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

    private void UpdateSelectionTips(ThumbnailEntry entry)
    {
        _selectionTipsTitle.text = entry.fileName;
        _selectionTipsDetail.text =
            $"时间: {entry.modifiedTime:yyyy-MM-dd HH:mm:ss}\n" +
            $"大小: {FormatBytes(entry.fileSize)}\n" +
            $"地点: {entry.locationText}\n" +
            $"人脸: {entry.faceText}\n" +
            $"CLIP: {entry.clipText}";
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
            return Directory.EnumerateDirectories(path).Take(Mathf.Max(1, maxCount)).ToArray();
        }
        catch
        {
            return Array.Empty<string>();
        }
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
}
