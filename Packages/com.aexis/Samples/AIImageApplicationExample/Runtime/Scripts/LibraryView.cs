using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Threading;
using Aexis.Samples.Async;
using UnityEngine;
using UnityEngine.UIElements;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

public sealed class LibraryView : BasePageView
{
    private const float ClipEditedMatchThreshold = 0.90f;
    private const float ClipNearDuplicateOriginalThreshold = 0.985f;
    private const float ClipNearDuplicateOriginalMinSizeRatio = 0.45f;
    private const float ClipNearDuplicateRepresentativeEpsilon = 0.0005f;
    private const string PendingText = "\u5F85\u63D0\u53D6";
    private const string PendingClipText = "\u5F85\u63A5\u5165";
    private const string EmptyText = "\u65E0";
    private const int ThumbnailMaxEdge = 640;
    private const int HiddenOriginalImportLimit = 512;
    private static readonly string[] HiddenOriginalDirectoryKeywords =
    {
        "原图",
        "原片",
        "底片",
        "raw",
        "original"
    };
    private static readonly IComparer<string> ExplorerComparer = ImageNavigationUtility.ExplorerNameComparer;

    private enum LibraryImageType
    {
        RawOriginal,
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

    private sealed class StorageRootOption
    {
        public string rootPath;
        public string displayName;
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
        public bool clipClassificationLoading;
        public bool clipClassificationReady;
        public LibraryImageType type;
        public string locationText = PendingText;
        public string faceText = PendingClipText;
        public string clipBaseText = PendingClipText;
        public string clipText = PendingClipText;
        public string cameraText = PendingText;
        public string apertureText = PendingText;
        public string mappedOriginalPath;
        public string mappedOriginalName;
        public string mappedOriginalLocationText;
        public string mappedOriginalCameraText;
        public string mappedOriginalApertureText;
        public DateTime? mappedOriginalCaptureTime;
        public float metadataOriginalScore;
        public float mappedOriginalSimilarity;
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

    private sealed class DirectoryScanResult
    {
        public List<ThumbnailEntry> entries = new List<ThumbnailEntry>();
        public StorageAccessSnapshot accessSnapshot;
    }

    private sealed class OriginalMetadataSnapshot
    {
        public string directoryPath;
        public string fileName;
        public LibraryImageType type;
        public float score;
        public DateTime? captureTime;
        public string locationText;
        public string cameraText;
        public string apertureText;
    }

    private static readonly HashSet<string> ImageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".psd", ".tiff", ".tif", ".exr",
        ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".srw", ".pef"
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
    private VisualElement _selectionTipsMeta;
    private Button _mappedOriginalLinkButton;

    private readonly List<ThumbnailEntry> _thumbnailEntries = new List<ThumbnailEntry>();
    private readonly List<ThumbnailEntry> _visibleEntries = new List<ThumbnailEntry>();
    private readonly HashSet<int> _loadedDirectoryIds = new HashSet<int>();
    private readonly Dictionary<string, ThumbnailEntry> _entryByPath = new Dictionary<string, ThumbnailEntry>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Image> _imageByPath = new Dictionary<string, Image>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Label> _statusByPath = new Dictionary<string, Label>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Label> _timeLabelByPath = new Dictionary<string, Label>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, VisualElement> _cardByPath = new Dictionary<string, VisualElement>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Label> _typeBadgeByPath = new Dictionary<string, Label>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, OriginalMetadataSnapshot> _originalMetadataByPath = new Dictionary<string, OriginalMetadataSnapshot>(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _hiddenOriginalDirectoryPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _hiddenOriginalImportedDirectories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    private readonly List<StorageRootOption> _storageRoots = new List<StorageRootOption>();
    private readonly SemaphoreSlim _clipClassificationSemaphore = new SemaphoreSlim(1, 1);
    private readonly Dictionary<string, StorageAccessSnapshot> _storageAccessByPath = new Dictionary<string, StorageAccessSnapshot>(StringComparer.OrdinalIgnoreCase);
    private bool _didInitialPathSync;
    private string _currentDriveRoot;
    private string _selectedDirectoryPath;
    private string _selectedThumbnailPath;
    private string _materializedDirectoryPath;
    private long _lastClickTicks;
    private string _lastClickPath;
    private CancellationTokenSource _thumbnailLoadCts;
    private CancellationTokenSource _directoryScanCts;
    private CancellationTokenSource _clipClassificationCts;
    private int _thumbnailLoadGeneration;
    private int _directoryScanGeneration;
    private bool _storagePermissionRequestInFlight;

    private sealed class StorageAccessSnapshot
    {
        public bool filesAccessible;
        public bool directoriesAccessible;
        public bool sawUnauthorized;
        public bool sawIoError;
    }

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
        RestoreSelectionState().Forget();
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
        CancelClipClassification();
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
        CancelClipClassification();
        ClearThumbnailEntries(true);
        _clipClassificationSemaphore.Dispose();
        base.OnDestroy();
    }

    protected override bool HandleDirectionalImageNavigation(int direction)
    {
        if (_visibleEntries.Count == 0)
            return false;

        var currentPath = ResolveSelectedOrCurrentVisiblePath();
        if (string.IsNullOrWhiteSpace(currentPath))
            return false;

        var currentIndex = _visibleEntries.FindIndex(entry => string.Equals(entry.fullPath, currentPath, StringComparison.OrdinalIgnoreCase));
        if (currentIndex < 0)
            return false;

        var targetIndex = currentIndex + Math.Sign(direction);
        if (targetIndex < 0 || targetIndex >= _visibleEntries.Count)
            return false;

        return SelectVisibleThumbnail(_visibleEntries[targetIndex], true);
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

        var title = new Label("\u56FE\u5E93");
        title.style.fontSize = 18;
        title.style.color = Color.white;
        title.style.unityFontStyleAndWeight = FontStyle.Bold;
        title.style.marginRight = 14;
        bar.Add(title);

        bar.Add(CreateFilterToggle("\u540D\u79F0", true, out _sortTimeToggle, OnSortToggleChanged));
        bar.Add(CreateFilterToggle("\u4EBA\u8138", false, out _sortFaceToggle, OnSortToggleChanged));
        bar.Add(CreateFilterToggle("\u5730\u70B9", false, out _sortLocationToggle, OnSortToggleChanged));
        bar.Add(CreateFilterToggle("\u539F\u56FE", true, out _showOriginalToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("\u4FEE\u56FE", true, out _showEditedToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("\u672A\u77E5", true, out _showUnknownToggle, ApplyFilters));
        bar.Add(CreateFilterToggle("\u6536\u85CF", false, out _favoritesOnlyToggle, ApplyFilters));
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

        var driveLabel = new Label(GetStorageRootLabel());
        driveLabel.style.color = Color.white;
        driveLabel.style.minWidth = 42;
        driveRow.Add(driveLabel);

        _drivePopup = new PopupField<string>(new List<string> { string.Empty }, 0);
        _drivePopup.style.flexGrow = 1;
        _drivePopup.RegisterValueChangedCallback(evt => OnStorageRootChanged(evt.newValue));
        driveRow.Add(_drivePopup);

        _directorySummary = new Label("\u8BF7\u9009\u62E9\u76EE\u5F55");
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
#if UNITY_6000_0_OR_NEWER
        _directoryTree.itemExpandedChanged += OnDirectoryExpandedChanged;
#endif
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

        _selectionTipsTitle = new Label("\u7F29\u7565\u56FE\u4FE1\u606F");
        _selectionTipsTitle.style.color = Color.white;
        _selectionTipsTitle.style.unityFontStyleAndWeight = FontStyle.Bold;
        selectionTips.Add(_selectionTipsTitle);

        _selectionTipsDetail = new Label("\u5355\u51FB\u7F29\u7565\u56FE\u67E5\u770B\u4FE1\u606F\uFF0C\u53CC\u51FB\u76F4\u63A5\u8FDB\u5165\u4E3B\u7F16\u8F91\u9875\u3002");
        _selectionTipsDetail.style.color = new Color(0.82f, 0.86f, 0.92f, 1f);
        _selectionTipsDetail.style.whiteSpace = WhiteSpace.Normal;
        _selectionTipsDetail.style.marginTop = 4;
        selectionTips.Add(_selectionTipsDetail);

        _selectionTipsMeta = new VisualElement();
        _selectionTipsMeta.style.marginTop = 8;
        _selectionTipsMeta.style.flexDirection = FlexDirection.Row;
        _selectionTipsMeta.style.alignItems = Align.Center;
        _selectionTipsMeta.style.flexWrap = Wrap.Wrap;
        selectionTips.Add(_selectionTipsMeta);

        var mappedOriginalLabel = new Label("\u6620\u5C04\u539F\u56FE:");
        mappedOriginalLabel.style.color = new Color(0.82f, 0.86f, 0.92f, 1f);
        mappedOriginalLabel.style.marginRight = 8;
        _selectionTipsMeta.Add(mappedOriginalLabel);

        _mappedOriginalLinkButton = new Button(OnMappedOriginalLinkClicked);
        _mappedOriginalLinkButton.text = EmptyText;
        _mappedOriginalLinkButton.style.backgroundColor = Color.clear;
        _mappedOriginalLinkButton.style.borderBottomWidth = 0;
        _mappedOriginalLinkButton.style.borderLeftWidth = 0;
        _mappedOriginalLinkButton.style.borderRightWidth = 0;
        _mappedOriginalLinkButton.style.borderTopWidth = 0;
        _mappedOriginalLinkButton.style.paddingLeft = 0;
        _mappedOriginalLinkButton.style.paddingRight = 0;
        _mappedOriginalLinkButton.style.paddingTop = 0;
        _mappedOriginalLinkButton.style.paddingBottom = 0;
        _mappedOriginalLinkButton.style.marginLeft = 0;
        _mappedOriginalLinkButton.style.marginRight = 0;
        _mappedOriginalLinkButton.style.unityTextAlign = TextAnchor.MiddleLeft;
        _mappedOriginalLinkButton.style.color = new Color(0.42f, 0.77f, 1f, 1f);
        _mappedOriginalLinkButton.style.display = DisplayStyle.None;
        _selectionTipsMeta.Add(_mappedOriginalLinkButton);

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
        var roots = BuildStorageRoots();
        _storageRoots.Clear();
        _storageRoots.AddRange(roots);

        if (_storageRoots.Count == 0)
        {
            var fallback = NormalizeRootPath(Path.GetPathRoot(Application.persistentDataPath));
            if (!string.IsNullOrWhiteSpace(fallback))
            {
                _storageRoots.Add(new StorageRootOption
                {
                    rootPath = fallback,
                    displayName = fallback
                });
            }
        }

        var choices = _storageRoots.Select(root => root.displayName).ToList();
        if (choices.Count == 0)
            choices.Add(string.Empty);

        _drivePopup.choices = choices;

        StorageRootOption preferred = _storageRoots.FirstOrDefault();
        var preferredRootPath = ResolvePreferredStorageRootPath();
        if (!string.IsNullOrWhiteSpace(preferredRootPath))
        {
            var match = _storageRoots.FirstOrDefault(root => string.Equals(root.rootPath, preferredRootPath, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                preferred = match;
        }

        if (preferred == null)
            return;

        _drivePopup.SetValueWithoutNotify(preferred.displayName);
        SwitchStorageRootAsync(preferred.rootPath, string.IsNullOrWhiteSpace(_selectedDirectoryPath)).Forget();
    }

    private void OnStorageRootChanged(string displayName)
    {
        if (string.IsNullOrWhiteSpace(displayName))
            return;

        var option = _storageRoots.FirstOrDefault(root => string.Equals(root.displayName, displayName, StringComparison.Ordinal));
        if (option == null || string.IsNullOrWhiteSpace(option.rootPath))
            return;

        SwitchStorageRootAsync(option.rootPath, true).Forget();
    }

    private static string GetStorageRootLabel()
    {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        return "\u76D8\u7B26";
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        return "\u5B58\u50A8";
#elif UNITY_STANDALONE_LINUX || UNITY_EDITOR_LINUX
        return "\u5B58\u50A8";
#elif UNITY_ANDROID
        return "\u5B58\u50A8";
#elif UNITY_IOS
        return "\u4F4D\u7F6E";
#else
        return "\u5B58\u50A8";
#endif
    }

    private string ResolvePreferredStorageRootPath()
    {
        if (!string.IsNullOrWhiteSpace(_selectedDirectoryPath))
        {
            var selectedRoot = GetBestStorageRootForPath(_selectedDirectoryPath);
            if (!string.IsNullOrWhiteSpace(selectedRoot))
                return selectedRoot;
        }

        var lastPath = Host?.GetLastImagePath();
        if (!string.IsNullOrWhiteSpace(lastPath))
            return GetBestStorageRootForPath(lastPath);

        return _storageRoots.Count > 0 ? _storageRoots[0].rootPath : null;
    }

    private string GetBestStorageRootForPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || _storageRoots.Count == 0)
            return null;

        try
        {
            var normalizedPath = Path.GetFullPath(path);
            StorageRootOption best = null;
            for (var i = 0; i < _storageRoots.Count; i++)
            {
                var candidate = _storageRoots[i];
                if (candidate == null || string.IsNullOrWhiteSpace(candidate.rootPath))
                    continue;
                if (!IsSameDirectoryOrChildOf(normalizedPath, candidate.rootPath))
                    continue;
                if (best == null || candidate.rootPath.Length > best.rootPath.Length)
                    best = candidate;
            }

            return best?.rootPath;
        }
        catch
        {
            return null;
        }
    }

    private List<StorageRootOption> BuildStorageRoots()
    {
        var roots = new List<StorageRootOption>();
        try
        {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            var drives = Environment.GetLogicalDrives()
                .Select(Path.GetPathRoot)
                .Where(d => !string.IsNullOrWhiteSpace(d))
                .Select(d => d.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar)
                .Where(Directory.Exists)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(s => s, StringComparer.OrdinalIgnoreCase)
                .ToList();
            for (var i = 0; i < drives.Count; i++)
            {
                roots.Add(new StorageRootOption
                {
                    rootPath = drives[i],
                    displayName = drives[i]
                });
            }
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
            AddStorageRoot(roots, "/", "\u672C\u5730");
            TryAddMountedRoots(roots, "/Volumes", "\u5916\u63A5");
#elif UNITY_STANDALONE_LINUX || UNITY_EDITOR_LINUX
            AddStorageRoot(roots, "/", "\u672C\u5730");
            TryAddMountedRoots(roots, "/mnt", "\u6302\u8F7D");
            TryAddMountedRoots(roots, "/media", "\u5916\u63A5");
#elif UNITY_ANDROID
            // Android blocks traversal into Android/data from a public storage root.
            // Keep the app-owned external-files location as its own root so images
            // copied there for normal use or runner validation remain browseable.
            AddStorageRoot(roots, Application.persistentDataPath, "\u5E94\u7528\u6587\u4EF6");
            AddStorageRoot(roots, "/storage/emulated/0/Pictures", "\u56FE\u7247");
            AddStorageRoot(roots, "/storage/emulated/0", "\u672C\u5730");
            TryAddMountedRoots(roots, "/storage", "\u5916\u63A5");
#elif UNITY_IOS
            var documents = Application.persistentDataPath;
            var root = documents;
            try
            {
                var parent = Directory.GetParent(documents);
                if (parent != null && !string.IsNullOrWhiteSpace(parent.FullName))
                    root = parent.FullName;
            }
            catch
            {
            }

            AddStorageRoot(roots, root, "\u672C\u5730");
#else
            AddStorageRoot(roots, Path.GetPathRoot(Application.persistentDataPath), "\u672C\u5730");
#endif
        }
        catch
        {
        }

        return roots
            .Where(root => root != null && !string.IsNullOrWhiteSpace(root.rootPath))
            .GroupBy(root => root.rootPath, StringComparer.OrdinalIgnoreCase)
            .Select(group => group.First())
            .OrderBy(root => root.displayName, ExplorerComparer)
            .ToList();
    }

    private static void AddStorageRoot(List<StorageRootOption> roots, string path, string displayName)
    {
        var normalized = NormalizeRootPath(path);
        if (string.IsNullOrWhiteSpace(normalized) || !Directory.Exists(normalized))
            return;

        roots.Add(new StorageRootOption
        {
            rootPath = normalized,
            displayName = displayName
        });
    }

    private static void TryAddMountedRoots(List<StorageRootOption> roots, string parentDirectory, string labelPrefix)
    {
        if (roots == null || string.IsNullOrWhiteSpace(parentDirectory) || !Directory.Exists(parentDirectory))
            return;

        try
        {
            foreach (var child in Directory.EnumerateDirectories(parentDirectory))
            {
                var normalized = NormalizeRootPath(child);
                if (string.IsNullOrWhiteSpace(normalized))
                    continue;

                var name = DirectoryNameFromPath(normalized);
                roots.Add(new StorageRootOption
                {
                    rootPath = normalized,
                    displayName = string.IsNullOrWhiteSpace(name) ? labelPrefix : (labelPrefix + " · " + name)
                });
            }
        }
        catch
        {
        }
    }

    private static string NormalizeRootPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return null;

        try
        {
            var full = Path.GetFullPath(path);
            if (string.IsNullOrWhiteSpace(full))
                return null;

            return full.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        }
        catch
        {
            return null;
        }
    }

    private async UniTaskVoid SwitchStorageRootAsync(string driveRoot, bool autoSelectRoot)
    {
        if (string.IsNullOrWhiteSpace(driveRoot))
            return;

        var accessOk = await EnsureStorageAccessAsync(driveRoot, true);
        if (!accessOk)
            return;

        SetDrive(driveRoot, autoSelectRoot);
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

#if UNITY_6000_0_OR_NEWER
    private void OnDirectoryExpandedChanged(TreeViewExpansionChangedArgs args)
    {
        if (!_directoryTree.IsExpanded(args.id))
            return;

        var data = _directoryTree.GetItemDataForId<DirectoryEntryData>(args.id);
        if (data == null || string.IsNullOrWhiteSpace(data.path))
            return;

        EnsureDirectoryChildrenLoaded(args.id, data.path);
    }
#else
    private void OnDirectoryExpandedChanged(int id, bool isExpanded)
    {
        if (!isExpanded)
            return;

        var data = _directoryTree.GetItemDataForId<DirectoryEntryData>(id);
        if (data == null || string.IsNullOrWhiteSpace(data.path))
            return;

        EnsureDirectoryChildrenLoaded(id, data.path);
    }
#endif

    private void EnsureDirectoryChildrenLoaded(int parentId, string directoryPath)
    {
        if (_loadedDirectoryIds.Contains(parentId))
            return;

        _loadedDirectoryIds.Add(parentId);
        RegisterHiddenOriginalDirectories(directoryPath);
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

    private async UniTask<bool> EnsureStorageAccessAsync(string path, bool showBusy)
    {
        if (string.IsNullOrWhiteSpace(path))
            return false;

#if UNITY_ANDROID
        if (!await EnsureAndroidReadPermissionAsync())
        {
            ShowToast("无法访问存储，请授予图片或文件读取权限", 2800);
            return false;
        }
#endif

        if (showBusy)
            ShowBusy("正在检查存储访问...");

        try
        {
            var snapshot = await UniTask.RunOnThreadPool(() => ProbeStorageAccess(path));
            if (snapshot != null)
                _storageAccessByPath[path] = snapshot;

            if (snapshot == null)
                return false;

            if (!snapshot.directoriesAccessible && !snapshot.filesAccessible)
            {
                ShowToast("当前存储位置无法访问", 2600);
                return false;
            }

            if (snapshot.directoriesAccessible && !snapshot.filesAccessible)
            {
                ShowToast("目录可见，但当前系统未返回文件列表", 2800);
            }

            return true;
        }
        catch
        {
            ShowToast("检查存储访问失败", 2400);
            return false;
        }
        finally
        {
            if (showBusy)
                HideBusy();
        }
    }

#if UNITY_ANDROID
    private async UniTask<bool> EnsureAndroidReadPermissionAsync()
    {
        if (Permission.HasUserAuthorizedPermission(Permission.ExternalStorageRead))
            return true;

        var readMediaImages = "android.permission.READ_MEDIA_IMAGES";
        if (Permission.HasUserAuthorizedPermission(readMediaImages))
            return true;

        if (_storagePermissionRequestInFlight)
        {
            await UniTask.WaitWhile(() => _storagePermissionRequestInFlight);
            return Permission.HasUserAuthorizedPermission(Permission.ExternalStorageRead) ||
                   Permission.HasUserAuthorizedPermission(readMediaImages);
        }

        _storagePermissionRequestInFlight = true;
        try
        {
            var callbacks = new PermissionCallbacks();
            var granted = false;
            var finished = false;

            callbacks.PermissionGranted += _ =>
            {
                granted = true;
                finished = true;
            };
            callbacks.PermissionDenied += _ => finished = true;
            callbacks.PermissionDeniedAndDontAskAgain += _ => finished = true;

            try
            {
                Permission.RequestUserPermission(readMediaImages, callbacks);
            }
            catch
            {
                Permission.RequestUserPermission(Permission.ExternalStorageRead, callbacks);
            }

            await UniTask.WaitUntil(() => finished);
            if (granted)
                return true;

            if (!Permission.HasUserAuthorizedPermission(readMediaImages))
            {
                finished = false;
                granted = false;
                callbacks = new PermissionCallbacks();
                callbacks.PermissionGranted += _ =>
                {
                    granted = true;
                    finished = true;
                };
                callbacks.PermissionDenied += _ => finished = true;
                callbacks.PermissionDeniedAndDontAskAgain += _ => finished = true;
                Permission.RequestUserPermission(Permission.ExternalStorageRead, callbacks);
                await UniTask.WaitUntil(() => finished);
            }

            return granted || Permission.HasUserAuthorizedPermission(Permission.ExternalStorageRead) ||
                   Permission.HasUserAuthorizedPermission(readMediaImages);
        }
        finally
        {
            _storagePermissionRequestInFlight = false;
        }
    }
#endif

    private StorageAccessSnapshot GetStorageAccessSnapshot(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return null;

        if (_storageAccessByPath.TryGetValue(path, out var snapshot))
            return snapshot;

        return null;
    }

    private void MaybeShowStorageAccessToast(string directoryPath)
    {
        var snapshot = GetStorageAccessSnapshot(directoryPath);
        if (snapshot == null)
            return;

        if (snapshot.directoriesAccessible && !snapshot.filesAccessible)
        {
            ShowToast("该目录可展开，但当前系统未返回文件列表", 2800);
            return;
        }

        if (snapshot.sawUnauthorized)
        {
            ShowToast("当前目录没有读取权限", 2600);
            return;
        }

        if (snapshot.sawIoError)
            ShowToast("读取目录内容失败", 2400);
    }

    private static StorageAccessSnapshot ProbeStorageAccess(string path)
    {
        var snapshot = new StorageAccessSnapshot();
        if (string.IsNullOrWhiteSpace(path))
            return snapshot;

        try
        {
            if (Directory.Exists(path))
            {
                snapshot.directoriesAccessible = true;
                try
                {
                    Directory.EnumerateDirectories(path).Take(1).ToArray();
                    snapshot.directoriesAccessible = true;
                }
                catch (UnauthorizedAccessException)
                {
                    snapshot.sawUnauthorized = true;
                    snapshot.directoriesAccessible = false;
                }
                catch (IOException)
                {
                    snapshot.sawIoError = true;
                    snapshot.directoriesAccessible = false;
                }

                try
                {
                    Directory.EnumerateFiles(path).Take(1).ToArray();
                    snapshot.filesAccessible = true;
                }
                catch (UnauthorizedAccessException)
                {
                    snapshot.sawUnauthorized = true;
                    snapshot.filesAccessible = false;
                }
                catch (IOException)
                {
                    snapshot.sawIoError = true;
                    snapshot.filesAccessible = false;
                }
            }
            else if (File.Exists(path))
            {
                snapshot.filesAccessible = true;
                snapshot.directoriesAccessible = true;
            }
        }
        catch (UnauthorizedAccessException)
        {
            snapshot.sawUnauthorized = true;
        }
        catch (IOException)
        {
            snapshot.sawIoError = true;
        }

        return snapshot;
    }

    private void OnDirectorySelectionChanged(IEnumerable<object> selectedItems)
    {
        var first = selectedItems.FirstOrDefault();
        if (first is not DirectoryEntryData entry || entry.isPlaceholder || string.IsNullOrWhiteSpace(entry.path))
            return;

        SelectDirectoryAsync(entry.path).Forget();
    }

    private async UniTaskVoid SelectDirectoryAsync(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            return;

        var accessOk = await EnsureStorageAccessAsync(directoryPath, false);
        if (!accessOk)
            return;

        _selectedDirectoryPath = directoryPath;
        _directorySummary.text = directoryPath;
        RefreshThumbnailGrid(directoryPath, !string.Equals(_materializedDirectoryPath, directoryPath, StringComparison.OrdinalIgnoreCase));
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
            ShowGridStatus(string.IsNullOrWhiteSpace(directoryPath) || !Directory.Exists(directoryPath) ? "\u8BE5\u76EE\u5F55\u4E0D\u5B58\u5728\u3002" : "\u6B63\u5728\u626B\u63CF\u76EE\u5F55...");
            if (string.IsNullOrWhiteSpace(directoryPath) || !Directory.Exists(directoryPath))
            {
                _selectionTipsDetail.text = "\u8BE5\u76EE\u5F55\u4E0D\u5B58\u5728\u3002";
                return;
            }

            _directoryScanGeneration++;
            var generation = _directoryScanGeneration;
            var scanCts = new CancellationTokenSource();
            _directoryScanCts = scanCts;
            var cancellationToken = scanCts.Token;

            try
            {
                RegisterHiddenOriginalDirectories(directoryPath);

                var scanResult = await UniTask.RunOnThreadPool(
                    () => ScanDirectoryEntries(directoryPath),
                    cancellationToken: cancellationToken);
                var files = scanResult?.entries ?? new List<ThumbnailEntry>();
                if (scanResult?.accessSnapshot != null)
                    _storageAccessByPath[directoryPath] = scanResult.accessSnapshot;

                if (cancellationToken.IsCancellationRequested ||
                    generation != _directoryScanGeneration ||
                    !string.Equals(_selectedDirectoryPath, directoryPath, StringComparison.OrdinalIgnoreCase))
                    return;

                _thumbnailEntries.AddRange(files);
                foreach (var entry in files)
                    _entryByPath[entry.fullPath] = entry;

                await ImportHiddenOriginalDirectoriesAsync(directoryPath, cancellationToken);

                if (files.Count == 0)
                    MaybeShowStorageAccessToast(directoryPath);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch
            {
                ShowGridStatus("\u76EE\u5F55\u626B\u63CF\u5931\u8D25\u3002");
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
        _typeBadgeByPath.Clear();
        _visibleEntries.Clear();
        _thumbnailGrid.Clear();

        IEnumerable<ThumbnailEntry> filtered = _thumbnailEntries;
        filtered = filtered.Where(entry =>
            ((_showOriginalToggle?.value ?? true) || (entry.type != LibraryImageType.Original && entry.type != LibraryImageType.RawOriginal)) &&
            ((_showEditedToggle?.value ?? true) || entry.type != LibraryImageType.Edited) &&
            ((_showUnknownToggle?.value ?? true) || entry.type != LibraryImageType.Unknown));

        if (_favoritesOnlyToggle?.value == true)
            filtered = filtered.Where(entry => entry.favorite);

        if (_sortFaceToggle?.value == true)
            filtered = filtered.OrderByDescending(entry => entry.faceText, ExplorerComparer)
                .ThenBy(entry => entry.fileName, ExplorerComparer);
        else if (_sortLocationToggle?.value == true)
            filtered = filtered.OrderBy(entry => ResolveDisplayLocation(entry), ExplorerComparer)
                .ThenBy(entry => entry.fileName, ExplorerComparer);
        else
            filtered = filtered.OrderBy(entry => entry.fileName, ExplorerComparer)
                .ThenByDescending(entry => entry.DisplayTime);

        _visibleEntries.AddRange(filtered);
        if (_visibleEntries.Count == 0)
        {
            ShowGridStatus(_directoryScanCts != null ? "\u6B63\u5728\u626B\u63CF\u76EE\u5F55..." : "\u5F53\u524D\u7B5B\u9009\u4E0B\u6CA1\u6709\u56FE\u7247\u3002");
            RestoreSelectedThumbnailTips();
            return;
        }

        foreach (var entry in _visibleEntries)
            _thumbnailGrid.Add(BuildThumbnailCard(entry));

        StartThumbnailRefresh();
        StartPendingClipClassification();
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

        var typeBadge = CreateBadge(GetTypeBadgeText(entry.type), GetTypeBadgeBackground(entry.type));
        badgeRow.Add(typeBadge);
        _typeBadgeByPath[entry.fullPath] = typeBadge;

        if (entry.favorite)
        {
            var favorite = CreateBadge("\u2605", new Color(0.96f, 0.28f, 0.31f, 0.96f));
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

    private void StartPendingClipClassification()
    {
        if (Host?.ClipRunner == null || _thumbnailLoadCts == null)
            return;

        var cancellationToken = _thumbnailLoadCts.Token;
        foreach (var entry in _visibleEntries)
        {
            var needsEmbeddingUpgrade = NeedsClipEmbeddingUpgrade(entry);
            if (entry.thumbnail == null || entry.clipClassificationLoading || (entry.clipClassificationReady && !needsEmbeddingUpgrade))
                continue;

            StartClipClassificationForEntry(entry, cancellationToken).Forget();
        }
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
        SelectVisibleThumbnail(entry, true);

        if (isDoubleClick)
            Host?.OpenLibraryImageInMain(entry.fullPath);
    }

    public void SyncSelectionFromImagePath(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
            return;

        try
        {
            var directory = Path.GetDirectoryName(imagePath);
            if (!string.IsNullOrWhiteSpace(directory) && Directory.Exists(directory))
                _selectedDirectoryPath = directory;
        }
        catch
        {
        }

        _selectedThumbnailPath = imagePath;

        if (_visibleEntries.Count == 0)
            return;

        if (_entryByPath.TryGetValue(imagePath, out var entry))
            SelectVisibleThumbnail(entry, false);
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
        var locationText = ResolveDisplayLocation(entry);
        var cameraText = ResolveDisplayCamera(entry);
        var apertureText = ResolveDisplayAperture(entry);
        var captureTime = ResolveDisplayCaptureTime(entry);
        var mappedOriginalText = ResolveMappedOriginalSummary(entry);
        _selectionTipsDetail.text =
            $"\u62CD\u6444\u65F6\u95F4: {captureTime:yyyy-MM-dd HH:mm:ss}\n" +
            $"\u6587\u4EF6\u5927\u5C0F: {FormatBytes(entry.fileSize)}\n" +
            $"\u5730\u70B9: {NormalizeDisplay(locationText)}\n" +
            $"\u76F8\u673A: {NormalizeDisplay(cameraText)}\n" +
            $"\u5149\u5708: {NormalizeDisplay(apertureText)}\n" +
            $"\u4EBA\u8138: {NormalizeDisplay(entry.faceText)}\n" +
            $"CLIP: {NormalizeDisplay(entry.clipText)}";

        if (_mappedOriginalLinkButton != null)
        {
            var canOpenMappedOriginal = entry.type == LibraryImageType.Edited &&
                                        !string.IsNullOrWhiteSpace(entry.mappedOriginalPath) &&
                                        File.Exists(entry.mappedOriginalPath);
            _mappedOriginalLinkButton.text = NormalizeDisplay(mappedOriginalText);
            _mappedOriginalLinkButton.style.display = DisplayStyle.Flex;
            _mappedOriginalLinkButton.SetEnabled(canOpenMappedOriginal);
            _mappedOriginalLinkButton.style.color = canOpenMappedOriginal
                ? new Color(0.42f, 0.77f, 1f, 1f)
                : new Color(0.82f, 0.86f, 0.92f, 1f);
        }
    }

    private void OnMappedOriginalLinkClicked()
    {
        if (string.IsNullOrWhiteSpace(_selectedThumbnailPath) ||
            !_entryByPath.TryGetValue(_selectedThumbnailPath, out var entry) ||
            entry == null ||
            string.IsNullOrWhiteSpace(entry.mappedOriginalPath))
        {
            ShowToast("\u6CA1\u6709\u53EF\u5B9A\u4F4D\u7684\u539F\u56FE", 2200);
            return;
        }

        RevealFileInShell(entry.mappedOriginalPath);
    }

    private void StartThumbnailRefresh()
    {
        CancelThumbnailRefresh();
        CancelClipClassification();
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

    private void CancelClipClassification()
    {
        if (_clipClassificationCts == null)
            return;

        try { _clipClassificationCts.Cancel(); } catch { }
        try { _clipClassificationCts.Dispose(); } catch { }
        _clipClassificationCts = null;
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

                ApplyPayloadMetadata(entry, payload);
                PersistCachedMetadata(entry);

                ApplyTypeFromMetadata(entry);

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
                StartClipClassificationForEntry(entry, cancellationToken).Forget();
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

    private async UniTaskVoid StartClipClassificationForEntry(ThumbnailEntry entry, CancellationToken thumbnailCancellationToken)
    {
        if (entry == null || entry.thumbnail == null || Host?.ClipRunner == null)
            return;

        var needsEmbeddingUpgrade = false;
        if (ClipClassificationCache.TryGetForFile(Host.ClipRunner, entry.fullPath, out var cached))
        {
            ApplyClipClassification(entry, cached);
            needsEmbeddingUpgrade = NeedsClipEmbeddingUpgrade(entry);
            if (!needsEmbeddingUpgrade)
                return;
        }

        if (entry.clipClassificationLoading || (entry.clipClassificationReady && !needsEmbeddingUpgrade))
            return;

        entry.clipClassificationLoading = true;

        if (_clipClassificationCts == null)
            _clipClassificationCts = new CancellationTokenSource();
        using (var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(thumbnailCancellationToken, _clipClassificationCts.Token))
        {
            var ct = linkedCts.Token;
            var acquired = false;

            try
            {
                await _clipClassificationSemaphore.WaitAsync(ct);
                acquired = true;
                try
                {
                    if (!needsEmbeddingUpgrade &&
                        ClipClassificationCache.TryGetForFile(Host.ClipRunner, entry.fullPath, out cached))
                    {
                        ApplyClipClassification(entry, cached);
                        return;
                    }

                    var result = await ClipClassificationCache.GetOrClassifyForFileAsync(
                        Host.ClipRunner,
                        entry.thumbnail,
                        entry.fullPath,
                        ct,
                        needsEmbeddingUpgrade);
                    ApplyClipClassification(entry, result);
                }
                finally
                {
                    if (acquired)
                        _clipClassificationSemaphore.Release();
                }
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning("[LibraryView] CLIP classification failed for " + (entry.fullPath ?? string.Empty) + " | " + e.Message);
            }
            finally
            {
                entry.clipClassificationLoading = false;
            }
        }
    }

    private void ApplyClipClassification(ThumbnailEntry entry, ClipClassificationResult result)
    {
        ApplyClipClassificationCore(entry, result, true);
    }

    private void ApplyClipClassificationCore(ThumbnailEntry entry, ClipClassificationResult result, bool refreshUi)
    {
        if (entry == null || !string.IsNullOrWhiteSpace(result.error))
            return;

        var best = string.IsNullOrWhiteSpace(result.bestLabel) ? EmptyText : result.bestLabel;
        var top = FormatClipTopScores(result.scores, 2);
        entry.clipBaseText = string.IsNullOrWhiteSpace(top) ? best : (best + "  " + top);
        entry.clipText = entry.clipBaseText;
        entry.faceText = best;
        entry.clipClassificationReady = HasClipEmbedding(result);
        ApplyTypeFromClipMapping(entry);

        if (!refreshUi)
            return;

        if (_sortFaceToggle?.value == true)
        {
            ApplyFilters();
            return;
        }

        if (string.Equals(_selectedThumbnailPath, entry.fullPath, StringComparison.OrdinalIgnoreCase))
            UpdateSelectionTips(entry);
    }

    private bool NeedsClipEmbeddingUpgrade(ThumbnailEntry entry)
    {
        return entry != null &&
               Host?.ClipRunner != null &&
               !string.IsNullOrWhiteSpace(entry.fullPath) &&
               ClipClassificationCache.NeedsEmbeddingUpgradeForFile(Host.ClipRunner, entry.fullPath);
    }

    private static bool HasClipEmbedding(ClipClassificationResult result)
    {
        return result.imageEmbedding != null && result.imageEmbedding.Length > 0;
    }

    private bool TryApplyCachedMetadata(ThumbnailEntry entry)
    {
        if (entry == null || string.IsNullOrWhiteSpace(entry.fullPath))
            return false;

        if (!ClipClassificationCache.TryGetMetadataForFile(entry.fullPath, out var metadata) || metadata == null)
            return false;

        if (metadata.captureTime.HasValue)
            entry.captureTime = metadata.captureTime.Value;
        if (!string.IsNullOrWhiteSpace(metadata.locationText))
            entry.locationText = metadata.locationText;
        if (!string.IsNullOrWhiteSpace(metadata.cameraText))
            entry.cameraText = metadata.cameraText;
        if (!string.IsNullOrWhiteSpace(metadata.apertureText))
            entry.apertureText = metadata.apertureText;

        return metadata.captureTime.HasValue ||
               !string.IsNullOrWhiteSpace(metadata.locationText) ||
               !string.IsNullOrWhiteSpace(metadata.cameraText) ||
               !string.IsNullOrWhiteSpace(metadata.apertureText);
    }

    private void PersistCachedMetadata(ThumbnailEntry entry)
    {
        if (entry == null || string.IsNullOrWhiteSpace(entry.fullPath))
            return;

        ClipClassificationCache.StoreFileMetadata(
            entry.fullPath,
            entry.captureTime,
            GetCacheableMetadataText(entry.locationText),
            GetCacheableMetadataText(entry.cameraText),
            GetCacheableMetadataText(entry.apertureText));
    }

    private static string GetCacheableMetadataText(string text)
    {
        return HasUsableMetadata(text) ? text : null;
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
                status.text = entry.thumbnailFailed ? "\u65E0\u6CD5\u9884\u89C8" : (entry.thumbnailLoading ? "\u52A0\u8F7D\u4E2D..." : "\u7B49\u5F85\u52A0\u8F7D");
                status.style.display = DisplayStyle.Flex;
            }
        }

        if (_timeLabelByPath.TryGetValue(entry.fullPath, out var timeLabel))
            timeLabel.text = FormatThumbnailTime(entry);

        if (_typeBadgeByPath.TryGetValue(entry.fullPath, out var typeBadge))
            UpdateBadgeVisual(typeBadge, entry.type);

        if (string.Equals(_selectedThumbnailPath, entry.fullPath, StringComparison.OrdinalIgnoreCase))
            UpdateSelectionTips(entry);
    }

    private async UniTaskVoid RestoreSelectionState()
    {
        if (string.IsNullOrWhiteSpace(_selectedDirectoryPath) || !Directory.Exists(_selectedDirectoryPath))
            return;

        try
        {
            var root = GetBestStorageRootForPath(_selectedDirectoryPath);
            if (!string.IsNullOrWhiteSpace(root) && !string.Equals(_currentDriveRoot, root, StringComparison.OrdinalIgnoreCase))
            {
                var option = _storageRoots.FirstOrDefault(item => string.Equals(item.rootPath, root, StringComparison.OrdinalIgnoreCase));
                if (option != null)
                    _drivePopup.SetValueWithoutNotify(option.displayName);
                if (!await EnsureStorageAccessAsync(root, false))
                    return;
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

    private string ResolveSelectedOrCurrentVisiblePath()
    {
        if (!string.IsNullOrWhiteSpace(_selectedThumbnailPath))
            return _selectedThumbnailPath;

        var currentPath = Host?.MainPage?.CurrentSourcePathForSync;
        if (!string.IsNullOrWhiteSpace(currentPath))
            return currentPath;

        return _visibleEntries.Count > 0 ? _visibleEntries[0].fullPath : null;
    }

    private bool SelectVisibleThumbnail(ThumbnailEntry entry, bool ensureVisible)
    {
        if (entry == null || string.IsNullOrWhiteSpace(entry.fullPath))
            return false;

        var previousPath = _selectedThumbnailPath;
        _selectedThumbnailPath = entry.fullPath;

        if (!string.IsNullOrWhiteSpace(previousPath) &&
            _cardByPath.TryGetValue(previousPath, out var previousCard) &&
            previousCard != null)
        {
            ApplySelectedCardBorder(previousCard, false);
        }

        if (_cardByPath.TryGetValue(entry.fullPath, out var nextCard) && nextCard != null)
            ApplySelectedCardBorder(nextCard, true);

        UpdateSelectionTips(entry);

        if (ensureVisible)
            ScrollToSelectedThumbnailSoon();
        else
            ScrollToSelectedThumbnailNow();

        return true;
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
        _typeBadgeByPath.Clear();

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
        _originalMetadataByPath.Clear();
        _hiddenOriginalDirectoryPaths.Clear();
        _hiddenOriginalImportedDirectories.Clear();
    }

    private static DirectoryScanResult ScanDirectoryEntries(string directoryPath)
    {
        var result = new DirectoryScanResult
        {
            accessSnapshot = new StorageAccessSnapshot()
        };

        try
        {
            result.entries = Directory.EnumerateFiles(directoryPath)
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
            result.accessSnapshot.filesAccessible = true;
            result.accessSnapshot.directoriesAccessible = true;
            return result;
        }
        catch (UnauthorizedAccessException)
        {
            result.accessSnapshot.sawUnauthorized = true;
            return result;
        }
        catch (IOException)
        {
            result.accessSnapshot.sawIoError = true;
            return result;
        }
        catch
        {
            result.accessSnapshot.sawIoError = true;
            return result;
        }
    }

    private static ThumbnailPayload LoadThumbnailPayload(string filePath, int maxEdge)
    {
        if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
            return new ThumbnailPayload();

        if (RawPhotoParser.IsRawExtension(filePath))
        {
            if (RawPhotoParser.TryParse(filePath, out var rawPhoto))
                return CreateThumbnailPayload(rawPhoto?.previewBytes, rawPhoto);
            return new ThumbnailPayload();
        }

        if (RawPhotoParser.TryLoadDisplayBytes(filePath, out var imageBytes, out var photoData))
            return CreateThumbnailPayload(imageBytes, photoData);

        if (RawPhotoParser.TryReadMetadata(filePath, out var metadataOnly))
            return CreateThumbnailPayload(null, metadataOnly);

        return new ThumbnailPayload { thumbnailBytes = LoadImageBytes(filePath) };
    }

    private static ThumbnailPayload LoadMetadataPayload(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
            return new ThumbnailPayload();

        if (RawPhotoParser.TryReadMetadata(filePath, out var metadataOnly))
            return CreateThumbnailPayload(null, metadataOnly);

        return new ThumbnailPayload();
    }

    private static void ApplyPayloadMetadata(ThumbnailEntry entry, ThumbnailPayload payload)
    {
        if (entry == null || payload == null)
            return;

        if (!string.IsNullOrWhiteSpace(payload.locationText))
            entry.locationText = payload.locationText;
        if (!string.IsNullOrWhiteSpace(payload.cameraText))
            entry.cameraText = payload.cameraText;
        if (!string.IsNullOrWhiteSpace(payload.apertureText))
            entry.apertureText = payload.apertureText;
        if (payload.captureTime.HasValue)
            entry.captureTime = payload.captureTime.Value;
    }

    private static ThumbnailPayload CreateThumbnailPayload(byte[] thumbnailBytes, RawPhotoParser.RawPhotoData photoData)
    {
        return new ThumbnailPayload
        {
            thumbnailBytes = thumbnailBytes,
            captureTime = photoData?.captureTime,
            locationText = photoData?.locationText,
            cameraText = photoData?.cameraText,
            apertureText = photoData?.apertureText
        };
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

    private static void UpdateBadgeVisual(Label badge, LibraryImageType type)
    {
        if (badge == null)
            return;

        var background = GetTypeBadgeBackground(type);
        badge.text = GetTypeBadgeText(type);
        badge.style.backgroundColor = new StyleColor(background);
        badge.style.color = background.grayscale > 0.7f ? Color.black : Color.white;
    }

    private static string GetTypeBadgeText(LibraryImageType type)
    {
        return type switch
        {
            LibraryImageType.RawOriginal => "RAW",
            LibraryImageType.Original => "\u539F\u56FE",
            LibraryImageType.Edited => "\u4FEE\u8FC7\u56FE",
            _ => "?"
        };
    }

    private static Color GetTypeBadgeBackground(LibraryImageType type)
    {
        return type switch
        {
            LibraryImageType.RawOriginal => new Color(1f, 0.86f, 0.38f, 0.96f),
            LibraryImageType.Original => new Color(1f, 1f, 1f, 0.92f),
            LibraryImageType.Edited => new Color(0.18f, 0.72f, 1f, 0.92f),
            _ => new Color(0.58f, 0.58f, 0.64f, 0.92f)
        };
    }

    private static LibraryImageType GuessType(string fileName)
    {
        if (RawPhotoParser.IsRawExtension(fileName))
            return LibraryImageType.RawOriginal;
        return LibraryImageType.Unknown;
    }

    private static bool IsImageFile(string filePath)
    {
        var ext = Path.GetExtension(filePath);
        return !string.IsNullOrWhiteSpace(ext) && (ImageExtensions.Contains(ext) || RawPhotoParser.IsRawExtension(filePath));
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

    private static bool IsHiddenOriginalDirectoryName(string directoryName)
    {
        if (string.IsNullOrWhiteSpace(directoryName))
            return false;

        for (var i = 0; i < HiddenOriginalDirectoryKeywords.Length; i++)
        {
            if (directoryName.IndexOf(HiddenOriginalDirectoryKeywords[i], StringComparison.OrdinalIgnoreCase) >= 0)
                return true;
        }

        return false;
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
                .Where(child => !IsHiddenOriginalDirectoryName(DirectoryNameFromPath(child)))
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

    private void RegisterHiddenOriginalDirectories(string parentDirectory)
    {
        if (string.IsNullOrWhiteSpace(parentDirectory) || !Directory.Exists(parentDirectory))
            return;

        try
        {
            foreach (var child in Directory.EnumerateDirectories(parentDirectory))
            {
                var name = DirectoryNameFromPath(child);
                if (IsHiddenOriginalDirectoryName(name))
                    _hiddenOriginalDirectoryPaths.Add(child);
            }
        }
        catch
        {
        }
    }

    private async UniTask ImportHiddenOriginalDirectoriesAsync(string parentDirectory, CancellationToken cancellationToken)
    {
        if (Host?.ClipRunner == null || string.IsNullOrWhiteSpace(parentDirectory))
            return;

        var targets = _hiddenOriginalDirectoryPaths
            .Where(path => !string.IsNullOrWhiteSpace(path) &&
                           IsSameDirectoryOrChildOf(path, parentDirectory) &&
                           !_hiddenOriginalImportedDirectories.Contains(path))
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToList();

        for (var i = 0; i < targets.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var hiddenDirectory = targets[i];
            await ImportHiddenOriginalDirectoryAsync(hiddenDirectory, cancellationToken);
            _hiddenOriginalImportedDirectories.Add(hiddenDirectory);
        }
    }

    private async UniTask ImportHiddenOriginalDirectoryAsync(string directoryPath, CancellationToken cancellationToken)
    {
        List<ThumbnailEntry> originals;
        try
        {
            var scanResult = await UniTask.RunOnThreadPool(
                () => ScanDirectoryEntries(directoryPath),
                cancellationToken: cancellationToken);
            originals = scanResult?.entries ?? new List<ThumbnailEntry>();
            if (scanResult?.accessSnapshot != null)
                _storageAccessByPath[directoryPath] = scanResult.accessSnapshot;
        }
        catch
        {
            return;
        }

        if (originals == null || originals.Count == 0)
            return;

        var imported = 0;
        for (var i = 0; i < originals.Count && imported < HiddenOriginalImportLimit; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var entry = originals[i];
            if (entry == null || string.IsNullOrWhiteSpace(entry.fullPath) || _entryByPath.ContainsKey(entry.fullPath))
                continue;

            var hasCachedMetadata = TryApplyCachedMetadata(entry);
            var hasCachedClip = ClipClassificationCache.TryGetForFile(Host.ClipRunner, entry.fullPath, out var cachedResult);
            var hasCachedEmbedding = hasCachedClip && HasClipEmbedding(cachedResult);

            if (!hasCachedMetadata && hasCachedEmbedding)
            {
                ThumbnailPayload metadataPayload = null;
                try
                {
                    metadataPayload = await UniTask.RunOnThreadPool(
                        () => LoadMetadataPayload(entry.fullPath),
                        cancellationToken: cancellationToken);
                }
                catch
                {
                }

                ApplyPayloadMetadata(entry, metadataPayload);
                PersistCachedMetadata(entry);
            }

            ApplyTypeFromMetadata(entry);
            if (IsHiddenOriginalSourcePath(entry.fullPath))
            {
                entry.type = LibraryImageType.Original;
                entry.metadataOriginalScore = Mathf.Max(entry.metadataOriginalScore, 1f);
            }

            if (hasCachedEmbedding)
            {
                ApplyClipClassificationCore(entry, cachedResult, false);
                RememberOriginalMetadata(entry);
                _entryByPath[entry.fullPath] = entry;
                imported++;
                continue;
            }

            ThumbnailPayload payload = null;
            try
            {
                payload = await UniTask.RunOnThreadPool(
                    () => LoadThumbnailPayload(entry.fullPath, ThumbnailMaxEdge),
                    cancellationToken: cancellationToken);
            }
            catch
            {
                continue;
            }

            ApplyPayloadMetadata(entry, payload);
            PersistCachedMetadata(entry);
            ApplyTypeFromMetadata(entry);
            if (IsHiddenOriginalSourcePath(entry.fullPath))
            {
                entry.type = LibraryImageType.Original;
                entry.metadataOriginalScore = Mathf.Max(entry.metadataOriginalScore, 1f);
            }

            if (payload?.thumbnailBytes == null || payload.thumbnailBytes.Length == 0)
                continue;

            var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            if (!texture.LoadImage(payload.thumbnailBytes, true))
            {
                UnityEngine.Object.Destroy(texture);
                continue;
            }

            texture.wrapMode = TextureWrapMode.Clamp;
            texture.filterMode = FilterMode.Bilinear;
            texture.name = entry.fileName;
            entry.thumbnail = texture;

            try
            {
                var result = await ClipClassificationCache.GetOrClassifyForFileAsync(
                    Host.ClipRunner,
                    texture,
                    entry.fullPath,
                    cancellationToken,
                    true);

                if (!string.IsNullOrWhiteSpace(result.error))
                    continue;

                ApplyClipClassificationCore(entry, result, false);
                RememberOriginalMetadata(entry);
                _entryByPath[entry.fullPath] = entry;
                try { UnityEngine.Object.Destroy(texture); } catch { }
                entry.thumbnail = null;
                imported++;
            }
            catch
            {
                try { UnityEngine.Object.Destroy(texture); } catch { }
                entry.thumbnail = null;
            }
        }
    }

    private static string NormalizeDisplay(string text)
    {
        return string.IsNullOrWhiteSpace(text) ? EmptyText : text;
    }

    private static string FormatThumbnailTime(ThumbnailEntry entry)
    {
        return entry.DisplayTime.ToString("yyyy-MM-dd HH:mm");
    }

    private void ApplyTypeFromMetadata(ThumbnailEntry entry)
    {
        if (entry == null)
            return;

        if (IsHiddenOriginalSourcePath(entry.fullPath))
        {
            entry.type = LibraryImageType.Original;
            entry.metadataOriginalScore = 1f;
            ClearMappedOriginal(entry);
            RememberOriginalMetadata(entry);
            RefreshEditedMappingsForKnownOriginal(entry);
            return;
        }

        if (IsRawOriginalFile(entry.fullPath))
        {
            entry.type = LibraryImageType.RawOriginal;
            entry.metadataOriginalScore = 1f;
            ClearMappedOriginal(entry);
            RememberOriginalMetadata(entry);
            RefreshEditedMappingsForKnownOriginal(entry);
            return;
        }

        entry.metadataOriginalScore = ScoreOriginalMetadata(entry);
        if (entry.metadataOriginalScore >= 0.62f)
        {
            entry.type = LibraryImageType.Original;
            ClearMappedOriginal(entry);
        }
        else if (entry.type != LibraryImageType.Edited)
        {
            entry.type = LibraryImageType.Unknown;
            ClearMappedOriginal(entry);
        }

        RememberOriginalMetadata(entry);
        RefreshEditedMappingsForKnownOriginal(entry);
    }

    private void ApplyTypeFromClipMapping(ThumbnailEntry entry)
    {
        if (entry == null || Host?.ClipRunner == null)
            return;

        if (entry.type == LibraryImageType.RawOriginal)
            return;

        if (entry.metadataOriginalScore >= 0.62f)
        {
            entry.type = LibraryImageType.Original;
            ClearMappedOriginal(entry);
            RememberOriginalMetadata(entry);
            RefreshEditedMappingsForKnownOriginal(entry);
            return;
        }

        if (HasStrongOriginalMetadata(entry))
        {
            entry.type = LibraryImageType.Original;
            entry.metadataOriginalScore = Mathf.Max(entry.metadataOriginalScore, 0.62f);
            ClearMappedOriginal(entry);
            RememberOriginalMetadata(entry);
            RefreshEditedMappingsForKnownOriginal(entry);
            return;
        }

        if (!ClipClassificationCache.TryGetImageRecordForFile(Host.ClipRunner, entry.fullPath, out var sourceRecord))
            return;

        var allRecords = ClipClassificationCache.GetAllImageRecords(Host.ClipRunner);
        if (allRecords == null || allRecords.Count == 0)
            return;

        var originalCandidates = new List<ClipClassificationCache.CachedClipImageRecord>();
        var selectedDirectory = _selectedDirectoryPath;
        for (var i = 0; i < allRecords.Count; i++)
        {
            var candidate = allRecords[i];
            if (candidate == null || string.IsNullOrWhiteSpace(candidate.filePath))
                continue;
            if (string.Equals(candidate.filePath, entry.fullPath, StringComparison.OrdinalIgnoreCase))
                continue;

            if (_entryByPath.TryGetValue(candidate.filePath, out var candidateEntry))
            {
                if (!IsMappingCandidateAllowed(candidate.filePath, selectedDirectory))
                    continue;
                if (candidateEntry.type == LibraryImageType.RawOriginal || candidateEntry.metadataOriginalScore >= 0.62f)
                    originalCandidates.Add(candidate);
            }
            else if (_originalMetadataByPath.TryGetValue(candidate.filePath, out var snapshot))
            {
                if (!IsMappingCandidateAllowed(candidate.filePath, selectedDirectory))
                    continue;
                if (snapshot.type == LibraryImageType.RawOriginal || snapshot.score >= 0.62f)
                    originalCandidates.Add(candidate);
            }
            else if (IsRawOriginalFile(candidate.filePath))
            {
                if (!IsMappingCandidateAllowed(candidate.filePath, selectedDirectory))
                    continue;
                originalCandidates.Add(candidate);
            }
        }

        var best = ClipImageSimilarity.FindBestMatch(sourceRecord, originalCandidates);
        if (best != null && best.target != null && ShouldTreatAsOriginalFromSimilarity(entry, sourceRecord, best, allRecords))
        {
            entry.type = LibraryImageType.Original;
            entry.metadataOriginalScore = Mathf.Max(entry.metadataOriginalScore, 0.62f);
            ClearMappedOriginal(entry);
            RememberOriginalMetadata(entry);
            RefreshEditedMappingsForKnownOriginal(entry);
            return;
        }

        if (best == null || best.target == null || best.cosineSimilarity < ClipEditedMatchThreshold)
        {
            if (entry.type != LibraryImageType.RawOriginal && entry.metadataOriginalScore < 0.62f)
            {
                entry.type = LibraryImageType.Unknown;
                ClearMappedOriginal(entry);
            }
            return;
        }

        entry.type = LibraryImageType.Edited;
        entry.mappedOriginalSimilarity = best.cosineSimilarity;
        entry.mappedOriginalPath = best.target.filePath;

        if (_entryByPath.TryGetValue(best.target.filePath, out var originalEntry))
        {
            entry.mappedOriginalName = originalEntry.fileName;
            entry.mappedOriginalLocationText = ResolveDisplayLocation(originalEntry);
            entry.mappedOriginalCameraText = ResolveDisplayCamera(originalEntry);
            entry.mappedOriginalApertureText = ResolveDisplayAperture(originalEntry);
            entry.mappedOriginalCaptureTime = originalEntry.captureTime ?? originalEntry.modifiedTime;
            entry.clipText = BuildMappedClipText(entry.clipBaseText, best.cosineSimilarity, originalEntry.fileName);
        }
        else
        {
            ApplyMappedOriginalSnapshot(entry, best.target.filePath);
            entry.clipText = BuildMappedClipText(entry.clipBaseText, best.cosineSimilarity, entry.mappedOriginalName);
        }
    }

    private bool ShouldTreatAsOriginalFromSimilarity(
        ThumbnailEntry entry,
        ClipClassificationCache.CachedClipImageRecord sourceRecord,
        ClipImageSimilarity.SimilarImageMatch best,
        IReadOnlyList<ClipClassificationCache.CachedClipImageRecord> allRecords)
    {
        if (entry == null || sourceRecord == null || best?.target == null)
            return false;

        if (best.cosineSimilarity < ClipNearDuplicateOriginalThreshold)
            return false;

        if (entry.metadataOriginalScore >= 0.62f || HasStrongOriginalMetadata(entry) || IsHiddenOriginalSourcePath(entry.fullPath))
            return true;

        if (HasSharedReliableMetadata(entry, best.target.filePath))
            return true;

        return HasReasonableNearDuplicateSize(entry, best.target.filePath) &&
               IsBestVisibleNearDuplicate(entry, sourceRecord, best.target, best.cosineSimilarity, allRecords);
    }

    private static string BuildMappedClipText(string currentClipText, float similarity, string originalName)
    {
        var suffix = "\u6620\u5C04\u539F\u56FE";
        if (!string.IsNullOrWhiteSpace(originalName))
            suffix += ": " + originalName;
        suffix += " @" + (similarity * 100f).ToString("0.0", CultureInfo.InvariantCulture) + "%";

        if (string.IsNullOrWhiteSpace(currentClipText) || string.Equals(currentClipText, PendingClipText, StringComparison.Ordinal))
            return suffix;

        return currentClipText + " | " + suffix;
    }

    private static float ScoreOriginalMetadata(ThumbnailEntry entry)
    {
        if (entry == null)
            return 0f;

        var score = 0f;
        if (entry.captureTime.HasValue)
            score += 0.28f;
        if (HasUsableMetadata(entry.cameraText))
            score += 0.28f;
        if (HasUsableMetadata(entry.apertureText))
            score += 0.18f;
        if (HasUsableMetadata(entry.locationText))
            score += 0.26f;

        return Mathf.Clamp01(score);
    }

    private static bool HasStrongOriginalMetadata(ThumbnailEntry entry)
    {
        if (entry == null)
            return false;

        var strongSignals = 0;
        if (entry.captureTime.HasValue)
            strongSignals++;
        if (HasUsableMetadata(entry.cameraText))
            strongSignals++;
        if (HasUsableMetadata(entry.apertureText))
            strongSignals++;
        if (HasUsableMetadata(entry.locationText))
            strongSignals++;

        return strongSignals >= 2;
    }

    private bool HasSharedReliableMetadata(ThumbnailEntry entry, string originalPath)
    {
        if (entry == null || !TryGetOriginalSnapshot(originalPath, out var snapshot) || snapshot == null)
            return false;

        var sharedSignals = 0;
        if (entry.captureTime.HasValue &&
            snapshot.captureTime.HasValue &&
            Math.Abs((entry.captureTime.Value - snapshot.captureTime.Value).TotalSeconds) <= 2d)
        {
            sharedSignals++;
        }

        if (HasUsableMetadata(entry.cameraText) &&
            HasUsableMetadata(snapshot.cameraText) &&
            string.Equals(entry.cameraText.Trim(), snapshot.cameraText.Trim(), StringComparison.OrdinalIgnoreCase))
        {
            sharedSignals++;
        }

        if (HasUsableMetadata(entry.apertureText) &&
            HasUsableMetadata(snapshot.apertureText) &&
            string.Equals(entry.apertureText.Trim(), snapshot.apertureText.Trim(), StringComparison.OrdinalIgnoreCase))
        {
            sharedSignals++;
        }

        if (HasUsableMetadata(entry.locationText) &&
            HasUsableMetadata(snapshot.locationText) &&
            string.Equals(entry.locationText.Trim(), snapshot.locationText.Trim(), StringComparison.OrdinalIgnoreCase))
        {
            sharedSignals++;
        }

        return sharedSignals > 0;
    }

    private bool HasReasonableNearDuplicateSize(ThumbnailEntry entry, string originalPath)
    {
        if (entry == null || entry.fileSize <= 0 || !TryGetFileSize(originalPath, out var targetLength) || targetLength <= 0)
            return true;

        var minSize = Math.Min(entry.fileSize, targetLength);
        var maxSize = Math.Max(entry.fileSize, targetLength);
        if (maxSize <= 0)
            return true;

        return minSize / (float)maxSize >= ClipNearDuplicateOriginalMinSizeRatio;
    }

    private bool IsBestVisibleNearDuplicate(
        ThumbnailEntry entry,
        ClipClassificationCache.CachedClipImageRecord sourceRecord,
        ClipClassificationCache.CachedClipImageRecord targetRecord,
        float currentSimilarity,
        IReadOnlyList<ClipClassificationCache.CachedClipImageRecord> allRecords)
    {
        if (entry == null || sourceRecord == null || targetRecord?.imageEmbedding == null || allRecords == null)
            return false;

        for (var i = 0; i < _thumbnailEntries.Count; i++)
        {
            var candidate = _thumbnailEntries[i];
            if (candidate == null ||
                string.Equals(candidate.fullPath, entry.fullPath, StringComparison.OrdinalIgnoreCase) ||
                IsHiddenOriginalSourcePath(candidate.fullPath))
            {
                continue;
            }

            var candidateRecord = FindImageRecordByPath(allRecords, candidate.fullPath);
            if (candidateRecord?.imageEmbedding == null)
                continue;

            var similarity = ClipImageSimilarity.CosineSimilarity(candidateRecord.imageEmbedding, targetRecord.imageEmbedding);
            if (similarity > currentSimilarity + ClipNearDuplicateRepresentativeEpsilon)
                return false;
        }

        return true;
    }

    private bool TryGetOriginalSnapshot(string originalPath, out OriginalMetadataSnapshot snapshot)
    {
        snapshot = null;
        if (string.IsNullOrWhiteSpace(originalPath))
            return false;

        if (_originalMetadataByPath.TryGetValue(originalPath, out snapshot) && snapshot != null)
            return true;

        if (_entryByPath.TryGetValue(originalPath, out var entry) && entry != null)
        {
            snapshot = new OriginalMetadataSnapshot
            {
                directoryPath = Path.GetDirectoryName(entry.fullPath),
                fileName = entry.fileName,
                type = entry.type,
                score = entry.metadataOriginalScore,
                captureTime = entry.captureTime ?? entry.modifiedTime,
                locationText = entry.locationText,
                cameraText = entry.cameraText,
                apertureText = entry.apertureText
            };
            return true;
        }

        return false;
    }

    private bool TryGetFileSize(string filePath, out long fileSize)
    {
        fileSize = 0;
        if (string.IsNullOrWhiteSpace(filePath))
            return false;

        if (_entryByPath.TryGetValue(filePath, out var entry) && entry != null && entry.fileSize > 0)
        {
            fileSize = entry.fileSize;
            return true;
        }

        try
        {
            var info = new FileInfo(filePath);
            if (!info.Exists || info.Length <= 0)
                return false;

            fileSize = info.Length;
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static ClipClassificationCache.CachedClipImageRecord FindImageRecordByPath(
        IReadOnlyList<ClipClassificationCache.CachedClipImageRecord> records,
        string filePath)
    {
        if (records == null || string.IsNullOrWhiteSpace(filePath))
            return null;

        for (var i = 0; i < records.Count; i++)
        {
            var record = records[i];
            if (record != null &&
                string.Equals(record.filePath, filePath, StringComparison.OrdinalIgnoreCase))
            {
                return record;
            }
        }

        return null;
    }

    private void RememberOriginalMetadata(ThumbnailEntry entry)
    {
        if (entry == null || string.IsNullOrWhiteSpace(entry.fullPath))
            return;

        if (entry.type == LibraryImageType.RawOriginal || entry.type == LibraryImageType.Original)
        {
            _originalMetadataByPath[entry.fullPath] = new OriginalMetadataSnapshot
            {
                directoryPath = Path.GetDirectoryName(entry.fullPath),
                fileName = entry.fileName,
                type = entry.type,
                score = entry.metadataOriginalScore,
                captureTime = entry.captureTime ?? entry.modifiedTime,
                locationText = entry.locationText,
                cameraText = entry.cameraText,
                apertureText = entry.apertureText
            };
        }
        else
        {
            _originalMetadataByPath.Remove(entry.fullPath);
        }
    }

    private void RefreshEditedMappingsForKnownOriginal(ThumbnailEntry originalEntry)
    {
        if (originalEntry == null || string.IsNullOrWhiteSpace(originalEntry.fullPath))
            return;
        if (originalEntry.type != LibraryImageType.RawOriginal && originalEntry.type != LibraryImageType.Original)
            return;

        for (var i = 0; i < _thumbnailEntries.Count; i++)
        {
            var candidate = _thumbnailEntries[i];
            if (candidate == null ||
                ReferenceEquals(candidate, originalEntry) ||
                string.Equals(candidate.fullPath, originalEntry.fullPath, StringComparison.OrdinalIgnoreCase) ||
                candidate.thumbnail == null ||
                !candidate.clipClassificationReady ||
                candidate.type == LibraryImageType.RawOriginal ||
                candidate.type == LibraryImageType.Original)
            {
                continue;
            }

            ApplyTypeFromClipMapping(candidate);
            UpdateThumbnailVisuals(candidate);
        }
    }

    private void ApplyMappedOriginalSnapshot(ThumbnailEntry entry, string originalPath)
    {
        entry.mappedOriginalPath = originalPath;
        entry.mappedOriginalName = Path.GetFileName(originalPath);

        if (string.IsNullOrWhiteSpace(originalPath))
            return;

        if (_originalMetadataByPath.TryGetValue(originalPath, out var snapshot) && snapshot != null)
        {
            if (!string.IsNullOrWhiteSpace(snapshot.fileName))
                entry.mappedOriginalName = snapshot.fileName;
            entry.mappedOriginalLocationText = snapshot.locationText;
            entry.mappedOriginalCameraText = snapshot.cameraText;
            entry.mappedOriginalApertureText = snapshot.apertureText;
            entry.mappedOriginalCaptureTime = snapshot.captureTime;
        }
    }

    private static bool HasUsableMetadata(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return false;

        return !string.Equals(text, PendingText, StringComparison.Ordinal) &&
               !string.Equals(text, EmptyText, StringComparison.Ordinal);
    }

    private bool IsMappingCandidateAllowed(string candidatePath, string selectedDirectory)
    {
        if (string.IsNullOrWhiteSpace(candidatePath))
            return false;
        if (string.IsNullOrWhiteSpace(selectedDirectory))
            return true;

        var candidateDirectory = Path.GetDirectoryName(candidatePath);
        if (string.IsNullOrWhiteSpace(candidateDirectory))
            return false;

        if (string.Equals(candidateDirectory, selectedDirectory, StringComparison.OrdinalIgnoreCase))
            return true;

        return _hiddenOriginalDirectoryPaths.Contains(candidateDirectory) &&
               IsSameDirectoryOrChildOf(candidateDirectory, selectedDirectory);
    }

    private void RevealFileInShell(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        if (!File.Exists(path))
        {
            ShowToast("\u539F\u56FE\u6587\u4EF6\u4E0D\u5B58\u5728", 2200);
            return;
        }

        try
        {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            var normalizedPath = Path.GetFullPath(path).Replace('/', '\\');
            Process.Start(new ProcessStartInfo("explorer.exe", "/select,\"" + normalizedPath + "\"") { UseShellExecute = true });
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
            Process.Start(new ProcessStartInfo("open", "-R \"" + path + "\"") { UseShellExecute = false });
#elif UNITY_EDITOR
            UnityEditor.EditorUtility.RevealInFinder(path);
#elif UNITY_STANDALONE_LINUX
            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(directory))
                Process.Start(new ProcessStartInfo("xdg-open", "\"" + directory + "\"") { UseShellExecute = false });
#elif UNITY_ANDROID || UNITY_IOS
            var url = "file://" + path.Replace('\\', '/');
            Application.OpenURL(url);
#else
            var url = "file://" + path.Replace('\\', '/');
            Application.OpenURL(url);
#endif
        }
        catch
        {
            ShowToast("\u65E0\u6CD5\u6253\u5F00\u539F\u56FE\u4F4D\u7F6E", 2200);
        }
    }

    private static bool IsRawOriginalFile(string filePath)
    {
        return RawPhotoParser.IsRawExtension(filePath);
    }

    private bool IsHiddenOriginalSourcePath(string filePath)
    {
        var directory = string.IsNullOrWhiteSpace(filePath) ? null : Path.GetDirectoryName(filePath);
        return !string.IsNullOrWhiteSpace(directory) && _hiddenOriginalDirectoryPaths.Contains(directory);
    }

    private static bool IsSameDirectoryOrChildOf(string path, string rootDirectory)
    {
        if (string.IsNullOrWhiteSpace(path) || string.IsNullOrWhiteSpace(rootDirectory))
            return false;

        try
        {
            var normalizedPath = Path.GetFullPath(path)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var normalizedRoot = Path.GetFullPath(rootDirectory)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

            if (string.Equals(normalizedPath, normalizedRoot, StringComparison.OrdinalIgnoreCase))
                return true;

            var prefix = normalizedRoot + Path.DirectorySeparatorChar;
            return normalizedPath.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private static void ClearMappedOriginal(ThumbnailEntry entry)
    {
        if (entry == null)
            return;

        entry.clipText = string.IsNullOrWhiteSpace(entry.clipBaseText) ? entry.clipText : entry.clipBaseText;
        entry.mappedOriginalPath = null;
        entry.mappedOriginalName = null;
        entry.mappedOriginalLocationText = null;
        entry.mappedOriginalCameraText = null;
        entry.mappedOriginalApertureText = null;
        entry.mappedOriginalCaptureTime = null;
        entry.mappedOriginalSimilarity = 0f;
    }

    private static string ResolveDisplayLocation(ThumbnailEntry entry)
    {
        if (entry == null)
            return null;

        if (entry.type == LibraryImageType.Edited && HasUsableMetadata(entry.mappedOriginalLocationText))
            return entry.mappedOriginalLocationText;
        return entry.locationText;
    }

    private static string ResolveDisplayCamera(ThumbnailEntry entry)
    {
        if (entry == null)
            return null;

        if (entry.type == LibraryImageType.Edited && HasUsableMetadata(entry.mappedOriginalCameraText))
            return entry.mappedOriginalCameraText;
        return entry.cameraText;
    }

    private static string ResolveDisplayAperture(ThumbnailEntry entry)
    {
        if (entry == null)
            return null;

        if (entry.type == LibraryImageType.Edited && HasUsableMetadata(entry.mappedOriginalApertureText))
            return entry.mappedOriginalApertureText;
        return entry.apertureText;
    }

    private static DateTime ResolveDisplayCaptureTime(ThumbnailEntry entry)
    {
        if (entry != null && entry.type == LibraryImageType.Edited && entry.mappedOriginalCaptureTime.HasValue)
            return entry.mappedOriginalCaptureTime.Value;
        return entry != null ? entry.DisplayTime : DateTime.MinValue;
    }

    private static string ResolveMappedOriginalSummary(ThumbnailEntry entry)
    {
        if (entry == null || entry.type != LibraryImageType.Edited || string.IsNullOrWhiteSpace(entry.mappedOriginalName))
            return null;

        var percent = (entry.mappedOriginalSimilarity * 100f).ToString("0.0", CultureInfo.InvariantCulture) + "%";
        return entry.mappedOriginalName + " (" + percent + ")";
    }

    private static string FormatClipTopScores(ClipLabelScore[] scores, int count)
    {
        if (scores == null || scores.Length == 0 || count <= 0)
            return string.Empty;

        var take = Mathf.Min(count, scores.Length);
        var parts = new List<string>(take);
        for (var i = 0; i < take; i++)
        {
            var score = scores[i];
            if (string.IsNullOrWhiteSpace(score.label))
                continue;

            parts.Add(score.label + ":" + (score.probability * 100f).ToString("0", CultureInfo.InvariantCulture) + "%");
        }

        return string.Join("  ", parts);
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
