using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

[DisallowMultipleComponent]
[RequireComponent(typeof(UIDocument))]
public class MainView : MonoBehaviour
{
    private const string PrefKeyLastImagePath = "MainView.LastImagePath";

    [SerializeField] private int leftPaneWidth = 320;
    [SerializeField] private int leftTopPaneHeight = 320;
    [SerializeField] private int leftCenterPaneHeight = 260;
    [SerializeField] private int maxDirectoryDepth = 6;
    [SerializeField] private int maxChildrenPerDirectory = 250;
    [SerializeField] private int textureCacheLimit = 12;

    private UIDocument _uiDocument;

    private TwoPaneSplitView _mainSplitView;
    private TwoPaneSplitView _leftSplitTop;
    private TwoPaneSplitView _leftSplitBottom;

    private PopupField<string> _drivePopup;
    private TreeView _directoryTree;
    private ListView _imageList;
    private ListView _historyList;
    private SplitCompareView _imageViewer;
    private Image2ImageAI _image2ImageAI;
    private bool _aiRunning;
    private VisualElement _busyOverlay;
    private VisualElement _busyBarTrack;
    private VisualElement _busyBar;
    private Label _busyText;
    private IVisualElementScheduledItem _busyAnim;
    private float _busyPhase;
    private VisualElement _choiceOverlay;
    private UniTaskCompletionSource<int> _choiceTcs;

    private readonly List<ImageFileEntry> _imageFiles = new List<ImageFileEntry>();
    private readonly List<HistoryEntry> _historyEntries = new List<HistoryEntry>();
    private readonly Dictionary<string, Texture2D> _textureCache = new Dictionary<string, Texture2D>(StringComparer.OrdinalIgnoreCase);
    private readonly Queue<string> _textureCacheOrder = new Queue<string>();

    private string _currentDriveRoot;
    private readonly HashSet<int> _loadedDirectoryIds = new HashSet<int>();
    private bool _suppressTreeEvents;

    private static readonly HashSet<string> ImageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".psd", ".tiff", ".tif", ".exr"
    };

    private void Awake()
    {
        _uiDocument = GetComponent<UIDocument>();
        _image2ImageAI = GetComponent<Image2ImageAI>();
        if (_image2ImageAI == null)
            _image2ImageAI = gameObject.AddComponent<Image2ImageAI>();

        _image2ImageAI.SelectResultIndex -= OnSelectAIResultIndex;
        _image2ImageAI.SelectResultIndex += OnSelectAIResultIndex;
    }

    private void OnEnable()
    {
        if (!Application.isPlaying)
            return;

        BuildUI();
        PopulateDrives();
        RestoreLastSelection();
    }

    private void OnDisable()
    {
        if (!Application.isPlaying)
            return;

        _imageViewer?.Clear();
        ClearHistory();
        HideBusy();
        HideChoice();

        if (_image2ImageAI != null)
            _image2ImageAI.SelectResultIndex -= OnSelectAIResultIndex;

        foreach (var kv in _textureCache)
        {
            if (kv.Value != null) Destroy(kv.Value);
        }

        _textureCache.Clear();
        _textureCacheOrder.Clear();

        var root = _uiDocument.rootVisualElement;
        if (root!=null && _mainSplitView != null)
            root.Remove(_mainSplitView);
        //root.Clear();
    }

    private void BuildUI()
    {
        var root = _uiDocument.rootVisualElement;
        root.Clear();
        root.style.width = Length.Percent(100);
        root.style.height = Length.Percent(100);
        root.style.flexGrow = 1;
        root.style.flexDirection = FlexDirection.Column;
        root.style.minHeight = 0;
        root.style.position = Position.Relative;

        _mainSplitView = new TwoPaneSplitView(0, leftPaneWidth, TwoPaneSplitViewOrientation.Horizontal);
        _mainSplitView.style.backgroundColor = Color.gray;
        _mainSplitView.style.flexGrow = 1;
        _mainSplitView.style.minHeight = 0;
        _mainSplitView.style.height = Length.Percent(100);
        root.Add(_mainSplitView);

        var leftPane = new VisualElement();
        leftPane.style.flexGrow = 1;
        leftPane.style.flexBasis = 0;
        leftPane.style.minWidth = 220;
        leftPane.style.maxWidth = 360;
        leftPane.style.minHeight = 0;
        leftPane.style.height = Length.Percent(100);
        _mainSplitView.Add(leftPane);

        var rightPane = new VisualElement();
        rightPane.style.flexGrow = 1;
        rightPane.style.flexBasis = 0;
        rightPane.style.minWidth = 320;
        rightPane.style.minHeight = 0;
        rightPane.style.height = Length.Percent(100);
        _mainSplitView.Add(rightPane);

        _leftSplitTop = new TwoPaneSplitView(0, leftTopPaneHeight, TwoPaneSplitViewOrientation.Vertical);
        _leftSplitTop.style.flexGrow = 1;
        _leftSplitTop.style.minHeight = 0;
        _leftSplitTop.style.height = Length.Percent(100);
        leftPane.Add(_leftSplitTop);

        var leftTop = new VisualElement();
        leftTop.style.flexGrow = 1;
        leftTop.style.flexBasis = 0;
        leftTop.style.minHeight = 0;
        leftTop.style.maxHeight = 540;
        leftTop.style.height = Length.Percent(100);
        _leftSplitTop.Add(leftTop);

        var leftBottomContainer = new VisualElement();
        leftBottomContainer.style.flexGrow = 1;
        leftBottomContainer.style.flexBasis = 0;
        leftBottomContainer.style.minHeight = 0;
        leftBottomContainer.style.height = Length.Percent(100);
        _leftSplitTop.Add(leftBottomContainer);

        _leftSplitBottom = new TwoPaneSplitView(0, leftCenterPaneHeight, TwoPaneSplitViewOrientation.Vertical);
        _leftSplitBottom.style.flexGrow = 1;
        _leftSplitBottom.style.minHeight = 0;
        _leftSplitBottom.style.height = Length.Percent(100);
        leftBottomContainer.Add(_leftSplitBottom);

        var leftCenter = new VisualElement();
        leftCenter.style.flexGrow = 1;
        leftCenter.style.flexBasis = 0;
        leftCenter.style.minHeight = 0;
        leftCenter.style.height = Length.Percent(100);
        _leftSplitBottom.Add(leftCenter);

        var leftBottom = new VisualElement();
        leftBottom.style.flexGrow = 1;
        leftBottom.style.flexBasis = 0;
        leftBottom.style.minHeight = 0;
        leftBottom.style.maxHeight = 270;
        leftBottom.style.height = Length.Percent(100);
        _leftSplitBottom.Add(leftBottom);

        BuildDirectoryBrowser(leftTop);
        BuildImageList(leftCenter);
        BuildHistoryList(leftBottom);
        BuildImageViewer(rightPane);
        BuildBusyOverlay(root);
    }

    private void BuildBusyOverlay(VisualElement root)
    {
        _busyOverlay = new VisualElement();
        _busyOverlay.style.position = Position.Absolute;
        _busyOverlay.style.left = 0;
        _busyOverlay.style.top = 0;
        _busyOverlay.style.right = 0;
        _busyOverlay.style.bottom = 0;
        _busyOverlay.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.35f));
        _busyOverlay.style.alignItems = Align.Center;
        _busyOverlay.style.justifyContent = Justify.Center;
        _busyOverlay.style.display = DisplayStyle.None;

        var panel = new VisualElement();
        panel.style.width = 360;
        panel.style.paddingLeft = 14;
        panel.style.paddingRight = 14;
        panel.style.paddingTop = 12;
        panel.style.paddingBottom = 12;
        panel.style.backgroundColor = new StyleColor(new Color(0.12f, 0.12f, 0.12f, 0.95f));
        panel.style.borderTopLeftRadius = 8;
        panel.style.borderTopRightRadius = 8;
        panel.style.borderBottomLeftRadius = 8;
        panel.style.borderBottomRightRadius = 8;
        panel.style.borderLeftWidth = 1;
        panel.style.borderRightWidth = 1;
        panel.style.borderTopWidth = 1;
        panel.style.borderBottomWidth = 1;
        panel.style.borderLeftColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        panel.style.borderRightColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        panel.style.borderTopColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        panel.style.borderBottomColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        panel.style.flexDirection = FlexDirection.Column;
        panel.style.alignItems = Align.Stretch;
        _busyOverlay.Add(panel);

        _busyText = new Label("处理中…");
        _busyText.style.unityTextAlign = TextAnchor.MiddleLeft;
        _busyText.style.whiteSpace = WhiteSpace.NoWrap;
        _busyText.style.overflow = Overflow.Hidden;
        _busyText.style.textOverflow = TextOverflow.Ellipsis;
        _busyText.style.marginBottom = 10;
        panel.Add(_busyText);

        _busyBarTrack = new VisualElement();
        _busyBarTrack.style.height = 10;
        _busyBarTrack.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.10f));
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
        _busyBar.style.width = 120;
        _busyBar.style.backgroundColor = new StyleColor(new Color(0.35f, 0.78f, 1f, 0.85f));
        _busyBarTrack.Add(_busyBar);

        root.Add(_busyOverlay);
    }

    private void ShowBusy(string text)
    {
        if (_busyOverlay == null) return;
        _busyText.text = string.IsNullOrWhiteSpace(text) ? "处理中…" : text;
        _busyOverlay.style.display = DisplayStyle.Flex;
        _busyOverlay.BringToFront();
        _busyPhase = 0f;

        if (_busyAnim == null)
        {
            _busyAnim = _busyOverlay.schedule.Execute(() =>
            {
                if (_busyOverlay.resolvedStyle.display == DisplayStyle.None)
                    return;

                var w = _busyBarTrack.resolvedStyle.width;
                if (w <= 1f) return;

                _busyPhase += 0.10f;
                var t = (Mathf.Sin(_busyPhase) + 1f) * 0.5f;
                var barW = Mathf.Clamp(w * (0.25f + 0.20f * (Mathf.Sin(_busyPhase * 1.7f) * 0.5f + 0.5f)), 50f, w);
                var x = (w - barW) * t;
                var a = 0.55f + 0.35f * (Mathf.Sin(_busyPhase * 2.3f) * 0.5f + 0.5f);

                _busyBar.style.width = barW;
                _busyBar.style.left = x;
                _busyBar.style.opacity = a;
            }).Every(16);
        }
        else
        {
            _busyAnim.Resume();
        }
    }

    private void HideBusy()
    {
        if (_busyOverlay == null) return;
        _busyOverlay.style.display = DisplayStyle.None;
        _busyAnim?.Pause();
    }

    private void BuildDirectoryBrowser(VisualElement parent)
    {
        parent.style.flexDirection = FlexDirection.Column;
        parent.style.flexGrow = 1;
        parent.style.minHeight = 0;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.paddingLeft = 8;
        header.style.paddingRight = 8;
        header.style.paddingTop = 6;
        header.style.paddingBottom = 6;
        header.style.flexShrink = 0;
        parent.Add(header);

        var driveLabel = new Label("Drive");
        driveLabel.style.minWidth = 44;
        header.Add(driveLabel);

        _drivePopup = new PopupField<string>(new List<string> { "" }, 0);
        _drivePopup.style.flexGrow = 1;
        _drivePopup.SetEnabled(false);
        _drivePopup.RegisterValueChangedCallback(evt => SetDrive(evt.newValue));
        header.Add(_drivePopup);

        _directoryTree = new TreeView();
        _directoryTree.viewDataKey = "MainView.DirectoryTree";
        _directoryTree.style.flexGrow = 1;
        _directoryTree.style.flexBasis = 0;
        _directoryTree.style.minHeight = 0;
        _directoryTree.virtualizationMethod = CollectionVirtualizationMethod.DynamicHeight;
        _directoryTree.fixedItemHeight = 25;
        _directoryTree.showBorder = true;
        _directoryTree.showAlternatingRowBackgrounds = AlternatingRowBackground.ContentOnly;
        _directoryTree.selectionType = SelectionType.Single;
        _directoryTree.makeItem = () => new Label();
        _directoryTree.bindItem = (element, index) =>
        {
            var label = (Label)element;
            var data = _directoryTree.GetItemDataForIndex<DirectoryEntry>(index);
            label.text = data.displayName;
        };
        _directoryTree.itemExpandedChanged += OnDirectoryItemExpandedChanged;
        _directoryTree.selectionChanged += OnDirectorySelectionChanged;
        parent.Add(_directoryTree);
    }

    private void BuildImageList(VisualElement parent)
    {
        parent.style.flexDirection = FlexDirection.Column;
        parent.style.flexGrow = 1;
        parent.style.minHeight = 0;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.paddingLeft = 8;
        header.style.paddingRight = 8;
        header.style.paddingTop = 6;
        header.style.paddingBottom = 6;
        header.style.flexShrink = 0;
        parent.Add(header);

        var title = new Label("Images");
        title.style.flexGrow = 1;
        header.Add(title);

        _imageList = new ListView();
        _imageList.style.flexGrow = 1;
        _imageList.style.flexBasis = 0;
        _imageList.style.minHeight = 0;
        _imageList.virtualizationMethod = CollectionVirtualizationMethod.FixedHeight;
        _imageList.fixedItemHeight = 24;
        _imageList.showBorder = true;
        _imageList.showAlternatingRowBackgrounds = AlternatingRowBackground.ContentOnly;
        _imageList.selectionType = SelectionType.Single;
        _imageList.itemsSource = _imageFiles;
        _imageList.makeItem = () =>
        {
            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.style.paddingLeft = 8;
            row.style.paddingRight = 8;
            row.style.paddingTop = 4;
            row.style.paddingBottom = 4;

            var label = new Label();
            label.style.flexGrow = 1;
            label.style.unityTextAlign = TextAnchor.MiddleLeft;
            label.style.whiteSpace = WhiteSpace.NoWrap;
            label.style.overflow = Overflow.Hidden;
            label.style.textOverflow = TextOverflow.Ellipsis;
            row.Add(label);

            return row;
        };
        _imageList.bindItem = (element, index) =>
        {
            var row = element;
            var label = row.Q<Label>();
            label.text = _imageFiles[index].fileName;
        };
        _imageList.selectionChanged += OnImageSelectionChanged;
        parent.Add(_imageList);
    }

    private void BuildHistoryList(VisualElement parent)
    {
        parent.style.flexDirection = FlexDirection.Column;
        parent.style.flexGrow = 1;
        parent.style.minHeight = 0;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.paddingLeft = 8;
        header.style.paddingRight = 8;
        header.style.paddingTop = 6;
        header.style.paddingBottom = 6;
        header.style.flexShrink = 0;
        parent.Add(header);

        var title = new Label("History");
        title.style.flexGrow = 1;
        header.Add(title);

        _historyList = new ListView();
        _historyList.style.flexGrow = 1;
        _historyList.style.flexBasis = 0;
        _historyList.style.minHeight = 0;
        _historyList.virtualizationMethod = CollectionVirtualizationMethod.FixedHeight;
        _historyList.fixedItemHeight = 24;
        _historyList.showBorder = true;
        _historyList.showAlternatingRowBackgrounds = AlternatingRowBackground.ContentOnly;
        _historyList.selectionType = SelectionType.Single;
        _historyList.itemsSource = _historyEntries;
        _historyList.makeItem = () =>
        {
            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.style.paddingLeft = 8;
            row.style.paddingRight = 8;
            row.style.paddingTop = 4;
            row.style.paddingBottom = 4;

            var label = new Label();
            label.style.flexGrow = 1;
            label.style.unityTextAlign = TextAnchor.MiddleLeft;
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
        parent.Add(_historyList);
    }

    private void BuildImageViewer(VisualElement parent)
    {
        parent.style.flexDirection = FlexDirection.Column;
        parent.style.flexGrow = 1;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Column;
        header.style.alignItems = Align.Stretch;
        header.style.paddingLeft = 8;
        header.style.paddingRight = 8;
        header.style.paddingTop = 6;
        header.style.paddingBottom = 6;
        header.style.flexShrink = 0;
        parent.Add(header);

        var row0 = new VisualElement();
        row0.style.flexDirection = FlexDirection.Row;
        row0.style.alignItems = Align.Center;
        header.Add(row0);

        row0.Add(new Button(OnFaceSwap) { text = "换脸" });
        row0.Add(new Button(OnSharpen) { text = "清晰化" });
        row0.Add(new Button(OnWhiten) { text = "美白" });
        row0.Add(new Button(OnDeGlare) { text = "去反光" });

        row0.Add(new Button(OnChangeBackground) { text = "换背景" });
        row0.Add(new Button(OnRemovePerson) { text = "去人" });
        row0.Add(new Button(OnColorGrade) { text = "调色" });
        row0.Add(new Button(OnDehaze) { text = "去霾" });

        var fitButton = new Button(() => _imageViewer.FitToView()) { text = "Fit" };
        row0.Add(fitButton);

        var resetButton = new Button(() => _imageViewer.ResetView()) { text = "Reset" };
        row0.Add(resetButton);


        _imageViewer = new SplitCompareView();
        _imageViewer.style.flexGrow = 1;
        parent.Add(_imageViewer);
    }

    private void PopulateDrives()
    {
        var drives = DriveInfo.GetDrives()
            .Where(d => d.IsReady)
            .Select(d => d.RootDirectory.FullName)
            .OrderBy(s => s, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (drives.Count == 0)
        {
            var fallback = Path.GetPathRoot(Application.persistentDataPath);
            _drivePopup.choices = new List<string> { fallback };
            _drivePopup.SetValueWithoutNotify(fallback);
            _drivePopup.SetEnabled(true);
            SetDrive(fallback);
            return;
        }

        _drivePopup.choices = drives;
        var defaultDrive = drives[0];
        var savedImagePath = PlayerPrefs.GetString(PrefKeyLastImagePath, "");
        if (!string.IsNullOrWhiteSpace(savedImagePath))
        {
            try
            {
                var root = Path.GetPathRoot(savedImagePath);
                if (!string.IsNullOrWhiteSpace(root))
                {
                    var match = drives.FirstOrDefault(d => string.Equals(d, root, StringComparison.OrdinalIgnoreCase));
                    if (!string.IsNullOrEmpty(match))
                        defaultDrive = match;
                }
            }
            catch
            {
            }
        }

        _drivePopup.SetValueWithoutNotify(defaultDrive);
        _drivePopup.SetEnabled(true);
        SetDrive(defaultDrive);
    }

    private void SetDrive(string driveRoot)
    {
        if (string.IsNullOrWhiteSpace(driveRoot)) return;

        _currentDriveRoot = driveRoot;
        _loadedDirectoryIds.Clear();

        var items = BuildDirectoryRootItems();
        ApplyDirectoryRootItems(items);
        if (items.Count > 0)
        {
            var rootId = items[0].id;
            _directoryTree.schedule.Execute(() =>
            {
                _directoryTree.ExpandItem(rootId);
                _directoryTree.SetSelectionById(rootId);
            });
        }
    }

    private List<TreeViewItemData<DirectoryEntry>> BuildDirectoryRootItems()
    {
        if (string.IsNullOrWhiteSpace(_currentDriveRoot))
            return new List<TreeViewItemData<DirectoryEntry>>();

        var rootPath = _currentDriveRoot;
        var displayName = rootPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        var root = BuildDirectoryItem(rootPath, displayName, 0);
        return new List<TreeViewItemData<DirectoryEntry>> { root };
    }

    private TreeViewItemData<DirectoryEntry> BuildDirectoryItem(string directoryPath, string displayName, int depth)
    {
        var id = StableId(directoryPath);
        var entry = new DirectoryEntry { path = directoryPath, displayName = displayName, isPlaceholder = false };

        if (depth >= maxDirectoryDepth)
            return new TreeViewItemData<DirectoryEntry>(id, entry, new List<TreeViewItemData<DirectoryEntry>>());

        List<TreeViewItemData<DirectoryEntry>> children;
        if (_loadedDirectoryIds.Contains(id))
        {
            children = EnumerateDirectoriesSafe(directoryPath, maxChildrenPerDirectory)
                .Select(p => BuildDirectoryItem(p, DirectoryNameFromPath(p), depth + 1))
                .ToList();
        }
        else if (depth < maxDirectoryDepth)
        {
            children = new List<TreeViewItemData<DirectoryEntry>> { BuildPlaceholderItem(directoryPath) };
        }
        else
        {
            children = new List<TreeViewItemData<DirectoryEntry>>();
        }

        return new TreeViewItemData<DirectoryEntry>(id, entry, children);
    }

    private TreeViewItemData<DirectoryEntry> BuildPlaceholderItem(string directoryPath)
    {
        var id = StableId(directoryPath + "|__placeholder__");
        var entry = new DirectoryEntry { path = "", displayName = "...", isPlaceholder = true };
        return new TreeViewItemData<DirectoryEntry>(id, entry, new List<TreeViewItemData<DirectoryEntry>>());
    }

    private static string DirectoryNameFromPath(string directoryPath)
    {
        var name = Path.GetFileName(directoryPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        if (string.IsNullOrEmpty(name))
            name = directoryPath;
        return name;
    }

    private static IEnumerable<string> EnumerateDirectoriesSafe(string directoryPath, int maxCount)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            return Array.Empty<string>();

        var limit = Mathf.Max(0, maxCount);

        try
        {
            var directoryType = typeof(Directory);
            var optionsType = directoryType.Assembly.GetType("System.IO.EnumerationOptions");
            if (optionsType != null)
            {
                var ignoreProp = optionsType.GetProperty("IgnoreInaccessible");
                var recurseProp = optionsType.GetProperty("RecurseSubdirectories");
                var options = Activator.CreateInstance(optionsType);
                ignoreProp?.SetValue(options, true);
                recurseProp?.SetValue(options, false);

                var method = directoryType.GetMethod("EnumerateDirectories", new[] { typeof(string), typeof(string), optionsType });
                if (method != null)
                {
                    var enumerable = method.Invoke(null, new[] { directoryPath, "*", options }) as IEnumerable<string>;
                    if (enumerable != null)
                        return enumerable.Take(limit).ToArray();
                }
            }
        }
        catch
        {
        }

        try
        {
            return Directory.EnumerateDirectories(directoryPath).Take(limit).ToArray();
        }
        catch
        {
            return Array.Empty<string>();
        }
    }

    private static bool HasSubDirectoriesSafe(string directoryPath)
    {
        return EnumerateDirectoriesSafe(directoryPath, 1).Any();
    }

    private void OnDirectoryItemExpandedChanged(TreeViewExpansionChangedArgs args)
    {
        if (_suppressTreeEvents) return;
        if (!args.isExpanded) return;

        DirectoryEntry data;
        try
        {
            data = _directoryTree.GetItemDataForId<DirectoryEntry>(args.id);
        }
        catch
        {
            return;
        }

        if (data.isPlaceholder) return;
        if (string.IsNullOrWhiteSpace(data.path)) return;

        EnsureDirectoryChildrenLoaded(args.id, data.path);
    }

    private void ApplyDirectoryRootItems(IList<TreeViewItemData<DirectoryEntry>> items)
    {
        _suppressTreeEvents = true;
        try
        {
            _directoryTree.SetRootItems(items);
            _directoryTree.Rebuild();
        }
        finally
        {
            _suppressTreeEvents = false;
        }
    }

    private void EnsureDirectoryChildrenLoaded(int parentId, string parentPath)
    {
        var parentKeyId = StableId(parentPath);
        if (_loadedDirectoryIds.Contains(parentKeyId))
            return;

        _loadedDirectoryIds.Add(parentKeyId);

        TryRemoveTreeItem(StableId(parentPath + "|__placeholder__"));

        var parentDepth = GetDepthForPath(parentPath);
        var childDepth = parentDepth + 1;

        var childDirs = EnumerateDirectoriesSafe(parentPath, maxChildrenPerDirectory).ToList();
        for (var i = 0; i < childDirs.Count; i++)
        {
            var childPath = childDirs[i];
            var childId = StableId(childPath);
            var childEntry = new DirectoryEntry
            {
                path = childPath,
                displayName = DirectoryNameFromPath(childPath),
                isPlaceholder = false
            };

            List<TreeViewItemData<DirectoryEntry>> childChildren;
            if (childDepth < maxDirectoryDepth)
                childChildren = new List<TreeViewItemData<DirectoryEntry>> { BuildPlaceholderItem(childPath) };
            else
                childChildren = new List<TreeViewItemData<DirectoryEntry>>();

            var item = new TreeViewItemData<DirectoryEntry>(childId, childEntry, childChildren);
            var rebuildTree = i == childDirs.Count - 1;
            _directoryTree.AddItem(item, parentId, -1, rebuildTree);
        }

        if (childDirs.Count == 0)
            _directoryTree.RefreshItems();

        _directoryTree.ExpandItem(parentId);
    }

    private int GetDepthForPath(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(_currentDriveRoot) || string.IsNullOrWhiteSpace(directoryPath))
            return 0;

        try
        {
            var root = Path.GetFullPath(_currentDriveRoot);
            if (!root.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal))
                root += Path.DirectorySeparatorChar;

            var full = Path.GetFullPath(directoryPath);
            if (!full.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal))
                full += Path.DirectorySeparatorChar;

            if (!full.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                return 0;

            var relative = full.Substring(root.Length).TrimEnd(Path.DirectorySeparatorChar);
            if (string.IsNullOrEmpty(relative))
                return 0;

            return relative.Split(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries).Length;
        }
        catch
        {
            return 0;
        }
    }

    private void TryRemoveTreeItem(int itemId)
    {
        try
        {
            var method = _directoryTree.GetType().GetMethod("RemoveItem", new[] { typeof(int), typeof(bool) });
            if (method != null)
            {
                method.Invoke(_directoryTree, new object[] { itemId, true });
                return;
            }
        }
        catch
        {
        }

        try
        {
            var method = _directoryTree.GetType().GetMethod("RemoveItem", new[] { typeof(int) });
            if (method != null)
            {
                method.Invoke(_directoryTree, new object[] { itemId });
            }
        }
        catch
        {
        }
    }

    private static int StableId(string key)
    {
        if (string.IsNullOrEmpty(key)) return 1;

        unchecked
        {
            const uint offsetBasis = 2166136261;
            const uint prime = 16777619;

            var normalized = key.Replace('\\', '/').ToLowerInvariant();
            uint hash = offsetBasis;
            for (var i = 0; i < normalized.Length; i++)
            {
                var c = normalized[i];
                hash ^= (byte)(c & 0xFF);
                hash *= prime;
                hash ^= (byte)((c >> 8) & 0xFF);
                hash *= prime;
            }

            var id = (int)(hash & 0x7FFFFFFF);
            if (id == 0) id = 1;
            if (id == -1) id = 2;
            return id;
        }
    }

    private void OnDirectorySelectionChanged(IEnumerable<object> selectedItems)
    {
        var first = selectedItems.FirstOrDefault();
        if (first is not DirectoryEntry entry) return;
        if (entry.isPlaceholder) return;
        if (string.IsNullOrWhiteSpace(entry.path)) return;

        var id = StableId(entry.path);
        _directoryTree.ExpandItem(id);
        EnsureDirectoryChildrenLoaded(id, entry.path);
        RefreshImageList(entry.path);
    }

    private void RefreshImageList(string directoryPath)
    {
        _imageFiles.Clear();

        if (!Directory.Exists(directoryPath))
        {
            _imageList.RefreshItems();
            _imageViewer.Clear();
            return;
        }

        try
        {
            var files = Directory.EnumerateFiles(directoryPath)
                .Where(IsImageFile)
                .Select(p => new ImageFileEntry { fullPath = p, fileName = Path.GetFileName(p) })
                .OrderBy(f => f.fileName, StringComparer.OrdinalIgnoreCase)
                .ToList();

            _imageFiles.AddRange(files);
        }
        catch
        {
        }

        _imageList.RefreshItems();
        _imageViewer.Clear();
    }

    private static bool IsImageFile(string filePath)
    {
        var ext = Path.GetExtension(filePath);
        return !string.IsNullOrEmpty(ext) && ImageExtensions.Contains(ext);
    }

    private void OnImageSelectionChanged(IEnumerable<object> selectedItems)
    {
        var first = selectedItems.FirstOrDefault();
        if (first is not ImageFileEntry entry) return;

        var tex = LoadTexture(entry.fullPath);
        if (tex != null)
        {
            ResetHistoryWithOriginal(tex, entry.fileName, entry.fullPath);
            CopySelectionToClipboard(entry.fullPath, tex);

            PlayerPrefs.SetString(PrefKeyLastImagePath, entry.fullPath);
            PlayerPrefs.Save();
        }
    }

    private void OnHistorySelectionChanged(IEnumerable<object> selectedItems)
    {
        var first = selectedItems.FirstOrDefault();
        if (first is not HistoryEntry entry) return;
        if (_historyEntries.Count == 0) return;

        var original = _historyEntries[0].texture;
        var current = entry.texture;
        _imageViewer.SetSources(current, original, entry.label);
    }

    private void ResetHistoryWithOriginal(Texture2D originalTexture, string label, string fullPath)
    {
        ClearHistory();

        _historyEntries.Add(new HistoryEntry
        {
            label = "原图: " + (label ?? originalTexture.name),
            texture = originalTexture,
            owned = false,
            sourcePath = fullPath
        });
        _historyList?.RefreshItems();
        _historyList?.SetSelection(0);

        _imageViewer.SetSources(originalTexture, originalTexture, label);
        _imageViewer.FitToView();
    }

    private void ClearHistory()
    {
        for (var i = 0; i < _historyEntries.Count; i++)
        {
            if (_historyEntries[i].owned && _historyEntries[i].texture != null)
                Destroy(_historyEntries[i].texture);
        }

        _historyEntries.Clear();
        _historyList?.RefreshItems();
    }

    private Texture2D GetCurrentHistoryTexture()
    {
        if (_historyList == null) return null;
        var index = _historyList.selectedIndex;
        if (index < 0 || index >= _historyEntries.Count) return null;
        return _historyEntries[index].texture;
    }

    private Texture2D GetOriginalHistoryTexture()
    {
        if (_historyEntries.Count == 0) return null;
        return _historyEntries[0].texture;
    }

    private void AddHistory(Texture2D texture, string label)
    {
        if (texture == null) return;
        if (_historyEntries.Count == 0) return;

        var entry = new HistoryEntry
        {
            label = label ?? texture.name,
            texture = texture,
            owned = true,
            sourcePath = null
        };

        _historyEntries.Insert(1, entry);
        _historyList.RefreshItems();
        _historyList.SetSelection(1);
        _historyList.ScrollToItem(1);

        _imageViewer.SetSources(entry.texture, GetOriginalHistoryTexture(), entry.label);
    }

    private void ApplyOperation(ImageOp op)
    {
        RunAIForOperation(op).Forget();
    }

    private void OnFaceSwap() => ApplyOperation(ImageOp.FaceSwap);
    private void OnSharpen() => ApplyOperation(ImageOp.Sharpen);
    private void OnWhiten() => ApplyOperation(ImageOp.Whiten);
    private void OnDeGlare() => ApplyOperation(ImageOp.DeGlare);
    private void OnChangeBackground() => ApplyOperation(ImageOp.ChangeBackground);
    private void OnRemovePerson() => ApplyOperation(ImageOp.RemovePerson);
    private void OnColorGrade() => ApplyOperation(ImageOp.ColorGrade);
    private void OnDehaze() => ApplyOperation(ImageOp.Dehaze);

    private async UniTaskVoid RunAIForOperation(ImageOp op)
    {
        if (_aiRunning) return;
        if (_image2ImageAI == null) return;

        var src = GetCurrentHistoryTexture();
        if (src == null) src = GetOriginalHistoryTexture();
        var original = GetOriginalHistoryTexture();
        if (src == null || original == null) return;

        _aiRunning = true;
        ShowBusy(OpLabel(op) + "处理中…");
        try
        {
            var prompt = BuildPromptForOp(op);
            var refs = new List<Texture2D> { src };
            if (op == ImageOp.FaceSwap && original != null && !ReferenceEquals(original, src))
                refs.Add(original);

            var result = await _image2ImageAI.ImageToImageAsync(
                refs,
                prompt,
                original.width,
                original.height,
                new System.Threading.CancellationToken());

            if (result != null)
                AddHistory(result, OpLabel(op));
        }
        finally
        {
            _aiRunning = false;
            HideBusy();
        }
    }

    private async UniTask<int> OnSelectAIResultIndex(IReadOnlyList<Texture2D> options)
    {
        HideBusy();
        var idx = await ShowChoiceAsync(options);
        return idx;
    }

    private UniTask<int> ShowChoiceAsync(IReadOnlyList<Texture2D> options)
    {
        if (options == null || options.Count == 0)
            return UniTask.FromResult(0);

        HideChoice();

        _choiceTcs = new UniTaskCompletionSource<int>();
        var root = _uiDocument.rootVisualElement;

        _choiceOverlay = new VisualElement();
        _choiceOverlay.style.position = Position.Absolute;
        _choiceOverlay.style.left = 0;
        _choiceOverlay.style.top = 0;
        _choiceOverlay.style.right = 0;
        _choiceOverlay.style.bottom = 0;
        _choiceOverlay.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.55f));
        _choiceOverlay.style.alignItems = Align.Center;
        _choiceOverlay.style.justifyContent = Justify.Center;

        var panel = new VisualElement();
        panel.style.width = 680;
        panel.style.maxHeight = 520;
        panel.style.paddingLeft = 12;
        panel.style.paddingRight = 12;
        panel.style.paddingTop = 10;
        panel.style.paddingBottom = 12;
        panel.style.backgroundColor = new StyleColor(new Color(0.12f, 0.12f, 0.12f, 0.97f));
        panel.style.borderTopLeftRadius = 10;
        panel.style.borderTopRightRadius = 10;
        panel.style.borderBottomLeftRadius = 10;
        panel.style.borderBottomRightRadius = 10;
        panel.style.borderLeftWidth = 1;
        panel.style.borderRightWidth = 1;
        panel.style.borderTopWidth = 1;
        panel.style.borderBottomWidth = 1;
        panel.style.borderLeftColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        panel.style.borderRightColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        panel.style.borderTopColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        panel.style.borderBottomColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        panel.style.flexDirection = FlexDirection.Column;
        panel.style.minHeight = 0;
        _choiceOverlay.Add(panel);

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.marginBottom = 8;
        panel.Add(header);

        var title = new Label("请选择一张结果图");
        title.style.flexGrow = 1;
        header.Add(title);

        var cancelBtn = new Button(() => ResolveChoice(0)) { text = "取消" };
        header.Add(cancelBtn);

        var scroll = new ScrollView(ScrollViewMode.Vertical);
        scroll.style.flexGrow = 1;
        scroll.style.minHeight = 0;
        panel.Add(scroll);

        var grid = new VisualElement();
        grid.style.flexDirection = FlexDirection.Row;
        grid.style.flexWrap = Wrap.Wrap;
        grid.style.alignContent = Align.FlexStart;
        scroll.Add(grid);

        for (var i = 0; i < options.Count; i++)
        {
            var tex = options[i];
            var idx = i;

            var card = new Button(() => ResolveChoice(idx));
            card.style.width = 160;
            card.style.height = 190;
            card.style.marginLeft = 6;
            card.style.marginRight = 6;
            card.style.marginTop = 6;
            card.style.marginBottom = 6;
            card.style.paddingLeft = 6;
            card.style.paddingRight = 6;
            card.style.paddingTop = 6;
            card.style.paddingBottom = 6;

            var img = new Image();
            img.image = tex;
            img.style.width = Length.Percent(100);
            img.style.height = 140;
            img.scaleMode = ScaleMode.ScaleToFit;
            card.Add(img);

            var lb = new Label("结果 " + idx);
            lb.style.unityTextAlign = TextAnchor.MiddleCenter;
            lb.style.marginTop = 6;
            card.Add(lb);

            grid.Add(card);
        }

        root.Add(_choiceOverlay);
        _choiceOverlay.BringToFront();

        return _choiceTcs.Task;
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
        if (_choiceOverlay == null) return;
        var root = _uiDocument != null ? _uiDocument.rootVisualElement : null;
        if (root != null && root.Contains(_choiceOverlay))
            root.Remove(_choiceOverlay);
        _choiceOverlay = null;
        _choiceTcs = null;
    }

    private static string BuildPromptForOp(ImageOp op)
    {
        return op switch
        {
            ImageOp.FaceSwap => "将输入图片中的前景人物脸部进行自然的换脸处理，保持光照、肤色和细节一致，结果真实且无明显伪影。",
            ImageOp.Sharpen => "对输入图片在保持原有构图基础上，严格保持前景人物脸容发型和五官不变，提高前景人物清晰度",
            ImageOp.Whiten => "对输入图片在保持原有构图基础上，严格保持前景人物脸容发型和五官不变，对前景人物进行自然美白与肤色优化，保持肤质真实，避免假白和过度磨皮。",
            ImageOp.DeGlare => "对输入图片进行去反光/去高光处理，降低镜面反射与眩光，保留细节与真实质感。",
            ImageOp.ChangeBackground => "在保持主体完整的前提下替换背景，边缘自然干净，主体与背景融合自然。",
            ImageOp.RemovePerson => "移除画面中的背景人物，自动补全背景，纹理连贯自然，无明显修补痕迹。",
            ImageOp.ColorGrade => "对输入图片进行调色，提升整体观感与色彩层次，保持自然不过饱和。",
            ImageOp.Dehaze => "对输入图片进行去霾与对比度提升，增强通透感，保留细节避免色偏。",
            _ => op.ToString()
        };
    }

    private static string OpLabel(ImageOp op)
    {
        return op switch
        {
            ImageOp.FaceSwap => "换脸",
            ImageOp.Sharpen => "清晰化",
            ImageOp.Whiten => "美白",
            ImageOp.DeGlare => "去反光",
            ImageOp.ChangeBackground => "换背景",
            ImageOp.RemovePerson => "去人",
            ImageOp.ColorGrade => "调色",
            ImageOp.Dehaze => "去霾",
            _ => op.ToString()
        };
    }

    private static Texture2D GenerateModifiedTexture(Texture2D src, ImageOp op)
    {
        if (src == null) return null;

        Color32[] pixels;
        try
        {
            pixels = src.GetPixels32();
        }
        catch
        {
            return null;
        }

        var w = src.width;
        var h = src.height;

        switch (op)
        {
            case ImageOp.Whiten:
                for (var i = 0; i < pixels.Length; i++)
                {
                    var p = pixels[i];
                    p.r = (byte)Mathf.Clamp(p.r + 18, 0, 255);
                    p.g = (byte)Mathf.Clamp(p.g + 18, 0, 255);
                    p.b = (byte)Mathf.Clamp(p.b + 18, 0, 255);
                    pixels[i] = p;
                }
                break;
            case ImageOp.ColorGrade:
                for (var i = 0; i < pixels.Length; i++)
                {
                    var p = pixels[i];
                    p.r = (byte)Mathf.Clamp(p.r * 1.06f, 0, 255);
                    p.g = (byte)Mathf.Clamp(p.g * 1.00f, 0, 255);
                    p.b = (byte)Mathf.Clamp(p.b * 0.96f, 0, 255);
                    pixels[i] = p;
                }
                break;
            case ImageOp.Dehaze:
                for (var i = 0; i < pixels.Length; i++)
                {
                    var p = pixels[i];
                    var c = (p.r + p.g + p.b) / 3f;
                    var k = 1.12f;
                    p.r = (byte)Mathf.Clamp((p.r - c) * k + c, 0, 255);
                    p.g = (byte)Mathf.Clamp((p.g - c) * k + c, 0, 255);
                    p.b = (byte)Mathf.Clamp((p.b - c) * k + c, 0, 255);
                    pixels[i] = p;
                }
                break;
            case ImageOp.DeGlare:
                for (var i = 0; i < pixels.Length; i++)
                {
                    var p = pixels[i];
                    p.r = (byte)Mathf.Clamp(p.r * 0.97f, 0, 255);
                    p.g = (byte)Mathf.Clamp(p.g * 0.97f, 0, 255);
                    p.b = (byte)Mathf.Clamp(p.b * 0.97f, 0, 255);
                    pixels[i] = p;
                }
                break;
            case ImageOp.Sharpen:
                {
                    var srcCopy = (Color32[])pixels.Clone();
                    var amount = 0.9f;
                    int Idx(int x, int y) => y * w + x;
                    for (var y = 1; y < h - 1; y++)
                    {
                        for (var x = 1; x < w - 1; x++)
                        {
                            var c = srcCopy[Idx(x, y)];
                            var l = srcCopy[Idx(x - 1, y)];
                            var r = srcCopy[Idx(x + 1, y)];
                            var u = srcCopy[Idx(x, y - 1)];
                            var d = srcCopy[Idx(x, y + 1)];
                            var nr = Mathf.Clamp(c.r + amount * (c.r * 4 - l.r - r.r - u.r - d.r), 0, 255);
                            var ng = Mathf.Clamp(c.g + amount * (c.g * 4 - l.g - r.g - u.g - d.g), 0, 255);
                            var nb = Mathf.Clamp(c.b + amount * (c.b * 4 - l.b - r.b - u.b - d.b), 0, 255);
                            pixels[Idx(x, y)] = new Color32((byte)nr, (byte)ng, (byte)nb, c.a);
                        }
                    }
                }
                break;
            case ImageOp.FaceSwap:
            case ImageOp.ChangeBackground:
            case ImageOp.RemovePerson:
            default:
                {
                    for (var y = 0; y < h; y++)
                    {
                        for (var x = 0; x < w; x++)
                        {
                            var i = y * w + x;
                            var p = pixels[i];
                            var t = (x + y) % 16;
                            if (t == 0)
                            {
                                p.r = (byte)Mathf.Clamp(p.r + 24, 0, 255);
                                p.b = (byte)Mathf.Clamp(p.b + 24, 0, 255);
                            }
                            pixels[i] = p;
                        }
                    }
                }
                break;
        }

        var tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
        tex.SetPixels32(pixels);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        tex.name = OpLabel(op);
        return tex;
    }

    private void RestoreLastSelection()
    {
        var lastImagePath = PlayerPrefs.GetString(PrefKeyLastImagePath, "");
        if (string.IsNullOrWhiteSpace(lastImagePath))
            return;

        _directoryTree.schedule.Execute(() =>
        {
            try
            {
                if (!File.Exists(lastImagePath))
                    return;

                var targetDir = Path.GetDirectoryName(lastImagePath);
                if (string.IsNullOrWhiteSpace(targetDir))
                    return;

                var root = Path.GetPathRoot(lastImagePath);
                if (!string.IsNullOrWhiteSpace(root) && !string.Equals(_currentDriveRoot, root, StringComparison.OrdinalIgnoreCase))
                {
                    _drivePopup.SetValueWithoutNotify(root);
                    SetDrive(root);
                }

                ExpandToDirectory(targetDir);
                var dirId = StableId(targetDir);
                _directoryTree.SetSelectionById(dirId);

                RefreshImageList(targetDir);

                var index = _imageFiles.FindIndex(f => string.Equals(f.fullPath, lastImagePath, StringComparison.OrdinalIgnoreCase));
                if (index >= 0)
                {
                    _imageList.SetSelection(index);
                    _imageList.ScrollToItem(index);
                }
            }
            catch
            {
            }
        });
    }

    private void ExpandToDirectory(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            return;
        if (string.IsNullOrWhiteSpace(_currentDriveRoot))
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

        var relative = full.Substring(root.Length)
            .Trim(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

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

    private void CopySelectionToClipboard(string imageFilePath, Texture2D texture)
    {
        if (string.IsNullOrWhiteSpace(imageFilePath)) return;

        GUIUtility.systemCopyBuffer = imageFilePath;

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        if (texture == null) return;

        byte[] pngBytes;
        try
        {
            pngBytes = texture.EncodeToPNG();
        }
        catch
        {
            return;
        }

        if (pngBytes == null || pngBytes.Length == 0) return;

        TrySetImageToWindowsClipboard(pngBytes);
#endif

#if UNITY_ANDROID && !UNITY_EDITOR
        if (texture == null) return;
        try
        {
            var pngBytes = texture.EncodeToPNG();
            if (pngBytes != null && pngBytes.Length > 0)
                TrySetImageToAndroidClipboard(pngBytes, Path.GetFileName(imageFilePath));
        }
        catch
        {
        }
#endif

#if UNITY_IOS && !UNITY_EDITOR
        if (texture == null) return;
        try
        {
            var pngBytes = texture.EncodeToPNG();
            if (pngBytes != null && pngBytes.Length > 0)
                TrySetImageToIOSClipboard(pngBytes);
        }
        catch
        {
        }
#endif

#if UNITY_STANDALONE_OSX && !UNITY_EDITOR
        if (texture == null) return;
        try
        {
            var pngBytes = texture.EncodeToPNG();
            if (pngBytes != null && pngBytes.Length > 0)
                TrySetImageToMacClipboard(pngBytes);
        }
        catch
        {
        }
#endif

#if UNITY_WEBGL && !UNITY_EDITOR
        if (texture == null) return;
        try
        {
            var pngBytes = texture.EncodeToPNG();
            if (pngBytes != null && pngBytes.Length > 0)
                TrySetImageToWebGLClipboard(pngBytes);
        }
        catch
        {
        }
#endif
    }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
    private static void TrySetImageToWindowsClipboard(byte[] pngBytes)
    {
        try
        {
            var clipboardType = Type.GetType("System.Windows.Forms.Clipboard, System.Windows.Forms");
            if (clipboardType == null) return;

            var imageType = Type.GetType("System.Drawing.Image, System.Drawing") ??
                            Type.GetType("System.Drawing.Image, System.Drawing.Common");
            if (imageType == null) return;

            var fromStream = imageType.GetMethod("FromStream", new[] { typeof(Stream) });
            if (fromStream == null) return;

            object imageObj;
            using (var ms = new MemoryStream(pngBytes, false))
            {
                imageObj = fromStream.Invoke(null, new object[] { ms });
            }

            if (imageObj == null) return;

            try
            {
                var setImage = clipboardType.GetMethod("SetImage", new[] { imageType });
                if (setImage != null)
                {
                    setImage.Invoke(null, new[] { imageObj });
                    return;
                }

                var setDataObject = clipboardType.GetMethod("SetDataObject", new[] { typeof(object), typeof(bool) });
                if (setDataObject != null)
                {
                    setDataObject.Invoke(null, new[] { imageObj, true });
                }
            }
            finally
            {
                var dispose = imageObj.GetType().GetMethod("Dispose", Type.EmptyTypes);
                dispose?.Invoke(imageObj, null);
            }
        }
        catch
        {
        }
    }
#endif

#if UNITY_ANDROID && !UNITY_EDITOR
    private static void TrySetImageToAndroidClipboard(byte[] pngBytes, string label)
    {
        var unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
        var activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
        var context = activity.Call<AndroidJavaObject>("getApplicationContext");

        var cacheDir = context.Call<AndroidJavaObject>("getCacheDir");
        var file = new AndroidJavaObject("java.io.File", cacheDir, "aiimage_clipboard.png");
        var fos = new AndroidJavaObject("java.io.FileOutputStream", file);
        fos.Call("write", pngBytes);
        fos.Call("flush");
        fos.Call("close");

        var pkg = context.Call<string>("getPackageName");
        var authority = pkg + ".aiimage.clipboardprovider";
        var fileProvider = new AndroidJavaClass("androidx.core.content.FileProvider");
        var uri = fileProvider.CallStatic<AndroidJavaObject>("getUriForFile", context, authority, file);

        var resolver = context.Call<AndroidJavaObject>("getContentResolver");
        var clipDataClass = new AndroidJavaClass("android.content.ClipData");
        var clip = clipDataClass.CallStatic<AndroidJavaObject>("newUri", resolver, label ?? "image", uri);

        var contextClass = new AndroidJavaClass("android.content.Context");
        var clipboardService = contextClass.GetStatic<string>("CLIPBOARD_SERVICE");
        var clipboard = context.Call<AndroidJavaObject>("getSystemService", clipboardService);
        clipboard.Call("setPrimaryClip", clip);
    }
#endif

#if UNITY_IOS && !UNITY_EDITOR
    [System.Runtime.InteropServices.DllImport("__Internal")]
    private static extern void AIImageClipboardCopyPNG(System.IntPtr data, int length);

    private static void TrySetImageToIOSClipboard(byte[] pngBytes)
    {
        var handle = System.Runtime.InteropServices.GCHandle.Alloc(pngBytes, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            AIImageClipboardCopyPNG(handle.AddrOfPinnedObject(), pngBytes.Length);
        }
        catch
        {
        }
        finally
        {
            handle.Free();
        }
    }
#endif

#if UNITY_STANDALONE_OSX && !UNITY_EDITOR
    private static void TrySetImageToMacClipboard(byte[] pngBytes)
    {
        try
        {
            var tmpDir = Application.temporaryCachePath;
            var tmpPath = Path.Combine(tmpDir, "aiimage_clipboard.png");
            File.WriteAllBytes(tmpPath, pngBytes);

            var script = "set the clipboard to (read (POSIX file " + QuoteAppleScript(tmpPath) + ") as «class PNGf»)";
            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "/usr/bin/osascript",
                Arguments = "-e " + QuoteAppleScript(script),
                UseShellExecute = false,
                CreateNoWindow = true
            };
            System.Diagnostics.Process.Start(psi);
        }
        catch
        {
        }
    }

    private static string QuoteAppleScript(string s)
    {
        if (s == null) return "\"\"";
        return "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }
#endif

#if UNITY_WEBGL && !UNITY_EDITOR
    [System.Runtime.InteropServices.DllImport("__Internal")]
    private static extern void AIImageClipboardCopyPNGFromUnity(System.IntPtr data, int length);

    private static void TrySetImageToWebGLClipboard(byte[] pngBytes)
    {
        var handle = System.Runtime.InteropServices.GCHandle.Alloc(pngBytes, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            AIImageClipboardCopyPNGFromUnity(handle.AddrOfPinnedObject(), pngBytes.Length);
        }
        catch
        {
        }
        finally
        {
            handle.Free();
        }
    }
#endif

    private Texture2D LoadTexture(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath)) return null;

        if (_textureCache.TryGetValue(filePath, out var cached) && cached != null)
            return cached;

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
            Destroy(tex);
            return null;
        }

        tex.name = Path.GetFileName(filePath);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;

        _textureCache[filePath] = tex;
        _textureCacheOrder.Enqueue(filePath);

        while (_textureCacheOrder.Count > Mathf.Max(1, textureCacheLimit))
        {
            var oldKey = _textureCacheOrder.Dequeue();
            if (string.Equals(oldKey, filePath, StringComparison.OrdinalIgnoreCase))
                continue;

            if (_textureCache.TryGetValue(oldKey, out var oldTex) && oldTex != null)
                Destroy(oldTex);

            _textureCache.Remove(oldKey);
        }

        return tex;
    }

    [Serializable]
    private struct DirectoryEntry
    {
        public string path;
        public string displayName;
        public bool isPlaceholder;
    }

    [Serializable]
    private struct ImageFileEntry
    {
        public string fullPath;
        public string fileName;
    }

    [Serializable]
    private struct HistoryEntry
    {
        public string label;
        public Texture2D texture;
        public bool owned;
        public string sourcePath;
    }

    private enum ImageOp
    {
        FaceSwap,
        Sharpen,
        Whiten,
        DeGlare,
        ChangeBackground,
        RemovePerson,
        ColorGrade,
        Dehaze
    }

    private sealed class SplitCompareView : VisualElement
    {
        private readonly Label _info;

        private Texture _texA;
        private Texture _texB;
        public float angleRad;
        private float offset;
        private float thicknessPx = 2f;
        private Color lineColor = new Color(1f, 1f, 1f, 0.9f);

        private float _zoom = 1f;
        private Vector2 _pan;

        private bool _panning;
        private int _panPointerId;
        private Vector3 _panStartPointer;
        private Vector2 _panStartPan;

        private bool _dragSplit;
        private int _splitPointerId;
        private Vector3 _splitDragStartLocal;
        private float _splitDragStartAngle;
        private float _splitDragStartOffset;

        public SplitCompareView()
        {
            style.flexDirection = FlexDirection.Column;
            style.flexGrow = 1;
            style.overflow = Overflow.Hidden;

            _info = new Label();
            _info.style.flexShrink = 0;
            _info.style.paddingLeft = 8;
            _info.style.paddingRight = 8;
            _info.style.paddingTop = 6;
            _info.style.paddingBottom = 6;
            _info.style.whiteSpace = WhiteSpace.NoWrap;
            _info.style.overflow = Overflow.Hidden;
            _info.style.textOverflow = TextOverflow.Ellipsis;
            Add(_info);

            style.flexGrow = 1;
            style.minHeight = 0;
            style.backgroundColor = new StyleColor(new Color(0.12f, 0.12f, 0.12f, 1f));

            pickingMode = PickingMode.Position;
            angleRad = 0f;
            offset = 0f;

            generateVisualContent += OnGenerateVisualContent;
            RegisterCallback<GeometryChangedEvent>(_ => { if (_texA != null || _texB != null) FitToView(); });
            RegisterCallback<WheelEvent>(OnWheel, TrickleDown.TrickleDown);
            RegisterCallback<PointerDownEvent>(OnPointerDown);
            RegisterCallback<PointerMoveEvent>(OnPointerMove);
            RegisterCallback<PointerUpEvent>(OnPointerUp);
            RegisterCallback<PointerCancelEvent>(OnPointerCancel);
        }

        public void SetSources(Texture2D current, Texture2D original, string label)
        {
            _texA = current;
            _texB = original;

            var refTex = _texA != null ? _texA : _texB;
            _info.text = refTex == null ? "" : (label ?? refTex.name);
            MarkDirtyRepaint();
        }

        public void Clear()
        {
            SetSources(null, null, null);
            ResetView();
        }

        public void ResetView()
        {
            _zoom = 1f;
            var refTex = _texA != null ? _texA : _texB;
            if (refTex != null)
            {
                var viewportSize = new Vector2(contentRect.width, Mathf.Max(1f, contentRect.height - _info.resolvedStyle.height));
                var scaledSize = new Vector2(refTex.width * _zoom, refTex.height * _zoom);
                _pan = (viewportSize - scaledSize) * 0.5f;
            }
            else
            {
                _pan = Vector2.zero;
            }
            MarkDirtyRepaint();
        }

        public void FitToView()
        {
            var refTex = _texA != null ? _texA : _texB;
            if (refTex == null) return;

            var viewportSize = new Vector2(contentRect.width, Mathf.Max(1f, contentRect.height - _info.resolvedStyle.height));
            if (viewportSize.x <= 1f || viewportSize.y <= 1f) return;

            var scaleX = viewportSize.x / refTex.width;
            var scaleY = viewportSize.y / refTex.height;
            _zoom = Mathf.Clamp(Mathf.Min(scaleX, scaleY), 0.01f, 20f);

            var scaledSize = new Vector2(refTex.width * _zoom, refTex.height * _zoom);
            _pan = (viewportSize - scaledSize) * 0.5f;
            MarkDirtyRepaint();
        }

        private void OnWheel(WheelEvent evt)
        {
            if (_texA == null && _texB == null) return;

            var viewportTop = Mathf.Max(_info.layout.height, _info.resolvedStyle.height);
            if (evt.localMousePosition.y < viewportTop)
                return;

            var viewportPos = new Vector2(evt.localMousePosition.x, evt.localMousePosition.y - viewportTop);
            var refTex0 = _texA != null ? _texA : _texB;
            var imgRect0 = GetImageRect(refTex0, _zoom, _pan + new Vector2(0f, viewportTop));
            var viewRect0 = new Rect(0, viewportTop, contentRect.width, Mathf.Max(1f, contentRect.height - viewportTop));
            var drawRect0 = IntersectRect(viewRect0, imgRect0);
            if (!drawRect0.Contains(evt.localMousePosition))
                return;
            var oldZoom = _zoom;
            var factor = Mathf.Pow(1.12f, -evt.delta.y / 12f);
            _zoom = Mathf.Clamp(oldZoom * factor, 0.02f, 40f);

            if (Mathf.Approximately(oldZoom, _zoom)) return;

            var imageLocal = (viewportPos - _pan) / oldZoom;
            _pan = viewportPos - imageLocal * _zoom;
            MarkDirtyRepaint();

            evt.StopPropagation();
        }

        private void OnPointerDown(PointerDownEvent evt)
        {
            if (_texA == null && _texB == null) return;

            if (evt.clickCount == 2)
            {
                FitToView();
                evt.StopPropagation();
                return;
            }

            if (evt.button != 0 && evt.button != 2) return;

            var refTex = _texA != null ? _texA : _texB;
            var viewportTop = Mathf.Max(_info.layout.height, _info.resolvedStyle.height);
            if (evt.localPosition.y < viewportTop)
                return;

            var imgRect = GetImageRect(refTex, _zoom, _pan + new Vector2(0f, viewportTop));
            if (imgRect.width <= 1f || imgRect.height <= 1f)
                return;
            var viewRect = new Rect(0, viewportTop, contentRect.width, Mathf.Max(1f, contentRect.height - viewportTop));
            var drawRect = IntersectRect(viewRect, imgRect);
            if (!drawRect.Contains(evt.localPosition))
                return;

            if (evt.button == 0)
            {
                var sd = SignedDistUv(evt.localPosition, imgRect);
                var thresholdUv = 12f / Mathf.Min(imgRect.width, imgRect.height);
                if (Mathf.Abs(sd) <= thresholdUv)
                {
                    _dragSplit = true;
                    _splitPointerId = evt.pointerId;
                    _splitDragStartLocal = evt.localPosition;
                    _splitDragStartAngle = angleRad;
                    _splitDragStartOffset = offset;
                    this.CapturePointer(_splitPointerId);
                    evt.StopPropagation();
                    return;
                }
            }

            _panning = true;
            _panPointerId = evt.pointerId;
            _panStartPointer = evt.localPosition;
            _panStartPan = _pan;
            this.CapturePointer(_panPointerId);
            evt.StopPropagation();
        }

        private void OnPointerMove(PointerMoveEvent evt)
        {
            if (_dragSplit && _splitPointerId == evt.pointerId && this.HasPointerCapture(_splitPointerId))
            {
                var refTex = _texA != null ? _texA : _texB;
                var viewportTop = Mathf.Max(_info.layout.height, _info.resolvedStyle.height);
                var imgRect = GetImageRect(refTex, _zoom, _pan + new Vector2(0f, viewportTop));
                var w = Mathf.Max(1f, imgRect.width);
                var h = Mathf.Max(1f, imgRect.height);
                var deltaLocal = evt.localPosition - _splitDragStartLocal;
                var deltaUv = new Vector2(deltaLocal.x / w, -deltaLocal.y / h);

                if (evt.shiftKey)
                {
                    var deltaAngle = (deltaLocal.x / w) * Mathf.PI * 2f;
                    angleRad = _splitDragStartAngle + deltaAngle;
                }
                else
                {
                    var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
                    offset = _splitDragStartOffset - Vector2.Dot(n, deltaUv);
                }

                MarkDirtyRepaint();
                evt.StopPropagation();
                return;
            }

            if (!_panning || _panPointerId != evt.pointerId || !this.HasPointerCapture(_panPointerId))
                return;

            _pan = _panStartPan + (Vector2)(evt.localPosition - _panStartPointer);
            MarkDirtyRepaint();
            evt.StopPropagation();
        }

        private void OnPointerUp(PointerUpEvent evt)
        {
            if (_dragSplit && _splitPointerId == evt.pointerId)
            {
                _dragSplit = false;
                if (this.HasPointerCapture(_splitPointerId))
                    this.ReleasePointer(_splitPointerId);
                evt.StopPropagation();
                return;
            }

            if (!_panning || _panPointerId != evt.pointerId)
                return;

            _panning = false;
            if (this.HasPointerCapture(_panPointerId))
                this.ReleasePointer(_panPointerId);

            evt.StopPropagation();
        }

        private void OnPointerCancel(PointerCancelEvent evt)
        {
            if (_dragSplit && _splitPointerId == evt.pointerId)
            {
                _dragSplit = false;
                if (this.HasPointerCapture(_splitPointerId))
                    this.ReleasePointer(_splitPointerId);
                evt.StopPropagation();
                return;
            }

            if (_panning && _panPointerId == evt.pointerId)
            {
                _panning = false;
                if (this.HasPointerCapture(_panPointerId))
                    this.ReleasePointer(_panPointerId);
                evt.StopPropagation();
            }
        }

        private static Rect GetImageRect(Texture refTex, float zoom, Vector2 pan)
        {
            if (refTex == null) return default;
            return new Rect(pan.x, pan.y, refTex.width * zoom, refTex.height * zoom);
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

        private float SignedDistUv(Vector2 pLocal, Rect imageRect)
        {
            var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
            var uv = new Vector2(
                (pLocal.x - imageRect.xMin) / imageRect.width,
                1f - ((pLocal.y - imageRect.yMin) / imageRect.height)
            );
            return Vector2.Dot(n, uv - new Vector2(0.5f, 0.5f)) + offset;
        }

        private void OnGenerateVisualContent(MeshGenerationContext mgc)
        {
            var w = contentRect.width;
            var viewportTop = Mathf.Max(_info.layout.height, _info.resolvedStyle.height);
            var h = contentRect.height - viewportTop;
            if (w <= 1f || h <= 1f)
                return;

            var refTex = _texA != null ? _texA : _texB;
            if (refTex == null)
                return;

            var viewRect = new Rect(0, viewportTop, w, h);
            var imageRect = GetImageRect(refTex, _zoom, _pan + new Vector2(0f, viewportTop));
            var drawRect = IntersectRect(viewRect, imageRect);
            if (drawRect.width <= 1f || drawRect.height <= 1f)
                return;

            float SignedDist(Vector2 p) => SignedDistUv(p, imageRect);

            if (_texA != null)
                DrawHalfPlane(mgc, _texA, drawRect, imageRect, SignedDist, keepNegative: true);
            if (_texB != null)
                DrawHalfPlane(mgc, _texB, drawRect, imageRect, SignedDist, keepNegative: false);

            DrawSplitLine(mgc, drawRect, imageRect);
        }

        private void DrawHalfPlane(
            MeshGenerationContext mgc,
            Texture tex,
            Rect drawRect,
            Rect imageRect,
            Func<Vector2, float> signedDistFunc,
            bool keepNegative)
        {
            var rectPoly = new List<Vector2>
            {
                new Vector2(drawRect.xMin, drawRect.yMin),
                new Vector2(drawRect.xMax, drawRect.yMin),
                new Vector2(drawRect.xMax, drawRect.yMax),
                new Vector2(drawRect.xMin, drawRect.yMax)
            };

            var clipped = ClipPolygon(rectPoly, signedDistFunc, keepNegative);
            if (clipped.Count < 3)
                return;

            var vCount = clipped.Count;
            var iCount = (vCount - 2) * 3;
            var mesh = mgc.Allocate(vCount, iCount, tex);

            for (int i = 0; i < vCount; i++)
            {
                var p = clipped[i];
                var uv = new Vector2(
                    (p.x - imageRect.xMin) / imageRect.width,
                    1f - ((p.y - imageRect.yMin) / imageRect.height)
                );
                mesh.SetNextVertex(new Vertex
                {
                    position = p,
                    uv = uv,
                    tint = Color.white
                });
            }

            for (int i = 0; i < vCount - 2; i++)
            {
                mesh.SetNextIndex(0);
                mesh.SetNextIndex((ushort)(i + 1));
                mesh.SetNextIndex((ushort)(i + 2));
            }
        }

        private void DrawSplitLine(MeshGenerationContext mgc, Rect drawRect, Rect imageRect)
        {
            var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
            var t = new Vector2(-n.y, n.x);
            var centerUv = new Vector2(0.5f, 0.5f) - n * offset;
            var centerLocal = new Vector2(
                imageRect.xMin + centerUv.x * imageRect.width,
                imageRect.yMin + (1f - centerUv.y) * imageRect.height
            );

            var len = Mathf.Max(drawRect.width, drawRect.height) * 1.5f;
            var halfThick = thicknessPx * 0.5f;

            var p0 = centerLocal + t * len + n * halfThick;
            var p1 = centerLocal + t * len - n * halfThick;
            var p2 = centerLocal - t * len - n * halfThick;
            var p3 = centerLocal - t * len + n * halfThick;

            var poly = new List<Vector2> { p0, p1, p2, p3 };
            Func<Vector2, float> clipRect = p =>
            {
                if (p.x < drawRect.xMin) return drawRect.xMin - p.x;
                if (p.x > drawRect.xMax) return p.x - drawRect.xMax;
                if (p.y < drawRect.yMin) return drawRect.yMin - p.y;
                if (p.y > drawRect.yMax) return p.y - drawRect.yMax;
                return -1f;
            };

            var clipped = ClipPolygon(poly, clipRect, keepNegative: true);
            if (clipped.Count < 3)
                return;

            var vCount = clipped.Count;
            var iCount = (vCount - 2) * 3;
            var mesh = mgc.Allocate(vCount, iCount);
            for (int i = 0; i < vCount; i++)
            {
                mesh.SetNextVertex(new Vertex
                {
                    position = clipped[i],
                    uv = Vector2.zero,
                    tint = lineColor
                });
            }
            for (int i = 0; i < vCount - 2; i++)
            {
                mesh.SetNextIndex(0);
                mesh.SetNextIndex((ushort)(i + 1));
                mesh.SetNextIndex((ushort)(i + 2));
            }
        }

        private static List<Vector2> ClipPolygon(List<Vector2> poly, Func<Vector2, float> signedDist, bool keepNegative)
        {
            var output = new List<Vector2>(poly.Count + 4);
            if (poly.Count == 0)
                return output;

            Vector2 prev = poly[^1];
            float prevD = signedDist(prev);
            bool prevIn = keepNegative ? prevD <= 0f : prevD >= 0f;

            for (int i = 0; i < poly.Count; i++)
            {
                Vector2 cur = poly[i];
                float curD = signedDist(cur);
                bool curIn = keepNegative ? curD <= 0f : curD >= 0f;

                if (curIn)
                {
                    if (!prevIn)
                    {
                        output.Add(Intersect(prev, cur, prevD, curD));
                    }
                    output.Add(cur);
                }
                else if (prevIn)
                {
                    output.Add(Intersect(prev, cur, prevD, curD));
                }

                prev = cur;
                prevD = curD;
                prevIn = curIn;
            }

            return output;
        }

        private static Vector2 Intersect(Vector2 a, Vector2 b, float da, float db)
        {
            var t = da / (da - db);
            return a + (b - a) * t;
        }
    }
}

