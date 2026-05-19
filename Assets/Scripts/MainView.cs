using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

[DisallowMultipleComponent]
[RequireComponent(typeof(UIDocument))]
public class MainView : MonoBehaviour
{
    private const string PrefKeyLastImagePath = "MainView.LastImagePath";

    [SerializeField] private int leftPaneWidth = 380;
    [SerializeField] private int leftTopPaneHeight = 420;
    [SerializeField] private int maxDirectoryDepth = 6;
    [SerializeField] private int maxChildrenPerDirectory = 250;
    [SerializeField] private int textureCacheLimit = 12;

    private UIDocument _uiDocument;

    private TwoPaneSplitView _mainSplitView;
    private TwoPaneSplitView _leftSplitView;

    private PopupField<string> _drivePopup;
    private TreeView _directoryTree;
    private ListView _imageList;
    private ImageViewer _imageViewer;

    private readonly List<ImageFileEntry> _imageFiles = new List<ImageFileEntry>();
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

        _leftSplitView = new TwoPaneSplitView(0, leftTopPaneHeight, TwoPaneSplitViewOrientation.Vertical);
        _leftSplitView.style.flexGrow = 1;
        _leftSplitView.style.minHeight = 0;
        _leftSplitView.style.height = Length.Percent(100);
        leftPane.Add(_leftSplitView);

        var leftTop = new VisualElement();
        leftTop.style.flexGrow = 1;
        leftTop.style.flexBasis = 0;
        leftTop.style.minHeight = 0;
        leftTop.style.height = Length.Percent(100);
        _leftSplitView.Add(leftTop);

        var leftBottom = new VisualElement();
        leftBottom.style.flexGrow = 1;
        leftBottom.style.flexBasis = 0;
        leftBottom.style.minHeight = 0;
        leftBottom.style.height = Length.Percent(100);
        _leftSplitView.Add(leftBottom);

        BuildDirectoryBrowser(leftTop);
        BuildImageList(leftBottom);
        BuildImageViewer(rightPane);
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

        row0.Add(new Button(() => { }) { text = "换脸" });
        row0.Add(new Button(() => { }) { text = "清晰化" });
        row0.Add(new Button(() => { }) { text = "美白" });
        row0.Add(new Button(() => { }) { text = "去反光" });

        row0.Add(new Button(() => { }) { text = "换背景" });
        row0.Add(new Button(() => { }) { text = "去人" });
        row0.Add(new Button(() => { }) { text = "调色" });
        row0.Add(new Button(() => { }) { text = "去霾" });

        var fitButton = new Button(() => _imageViewer.FitToView()) { text = "Fit" };
        row0.Add(fitButton);

        var resetButton = new Button(() => _imageViewer.ResetView()) { text = "Reset" };
        row0.Add(resetButton);


        _imageViewer = new ImageViewer();
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
        _imageViewer.SetTexture(tex, entry.fileName);
        CopySelectionToClipboard(entry.fullPath, tex);

        PlayerPrefs.SetString(PrefKeyLastImagePath, entry.fullPath);
        PlayerPrefs.Save();
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

    private sealed class ImageViewer : VisualElement
    {
        private readonly VisualElement _viewport;
        private readonly Image _image;
        private readonly Label _info;

        private Texture2D _texture;
        private float _zoom = 1f;
        private Vector3 _pan;

        private bool _dragging;
        private int _dragPointerId;
        private Vector3 _dragStartPointer;
        private Vector3 _dragStartPan;

        public ImageViewer()
        {
            style.flexDirection = FlexDirection.Column;
            style.flexGrow = 1;

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

            _viewport = new VisualElement();
            _viewport.style.flexGrow = 1;
            _viewport.style.overflow = Overflow.Hidden;
            _viewport.style.backgroundColor = new StyleColor(new Color(0.12f, 0.12f, 0.12f, 1f));
            _viewport.style.minHeight = 0;
            Add(_viewport);

            _image = new Image();
            _image.pickingMode = PickingMode.Ignore;
            _image.style.position = Position.Absolute;
            _image.style.left = 0;
            _image.style.top = 0;
            _image.style.transformOrigin = new TransformOrigin(Length.Percent(0), Length.Percent(0));
            _viewport.Add(_image);

            _viewport.RegisterCallback<GeometryChangedEvent>(_ => { if (_texture != null) FitToView(); });
            _viewport.RegisterCallback<WheelEvent>(OnWheel, TrickleDown.TrickleDown);
            _viewport.RegisterCallback<PointerDownEvent>(OnPointerDown, TrickleDown.TrickleDown);
            _viewport.RegisterCallback<PointerMoveEvent>(OnPointerMove, TrickleDown.TrickleDown);
            _viewport.RegisterCallback<PointerUpEvent>(OnPointerUp, TrickleDown.TrickleDown);
        }

        public void SetTexture(Texture2D texture, string fileName)
        {
            _texture = texture;
            _image.image = texture;

            if (texture == null)
            {
                _image.style.width = 0;
                _image.style.height = 0;
                _info.text = "";
                return;
            }

            _image.style.width = texture.width;
            _image.style.height = texture.height;
            _info.text = fileName ?? texture.name;

            FitToView();
        }

        public void Clear()
        {
            SetTexture(null, null);
            ResetView();
        }

        public void ResetView()
        {
            _zoom = 1f;
            if (_texture != null)
            {
                var viewportSize = _viewport.contentRect.size;
                var scaledSize = new Vector2(_texture.width * _zoom, _texture.height * _zoom);
                _pan = (viewportSize - scaledSize) * 0.5f;
            }
            else
            {
                _pan = Vector2.zero;
            }
            ApplyTransform();
        }

        public void FitToView()
        {
            if (_texture == null) return;

            var viewportSize = _viewport.contentRect.size;
            if (viewportSize.x <= 1f || viewportSize.y <= 1f) return;

            var scaleX = viewportSize.x / _texture.width;
            var scaleY = viewportSize.y / _texture.height;
            _zoom = Mathf.Clamp(Mathf.Min(scaleX, scaleY), 0.01f, 20f);

            var scaledSize = new Vector2(_texture.width * _zoom, _texture.height * _zoom);
            _pan = (viewportSize - scaledSize) * 0.5f;
            ApplyTransform();
        }

        private void OnWheel(WheelEvent evt)
        {
            if (_texture == null) return;

            Vector3 viewportPos = evt.localMousePosition;
            var oldZoom = _zoom;
            var factor = Mathf.Pow(1.12f, -evt.delta.y / 12f);
            _zoom = Mathf.Clamp(oldZoom * factor, 0.02f, 40f);

            if (Mathf.Approximately(oldZoom, _zoom)) return;

            var imageLocal = (viewportPos - _pan) / oldZoom;
            _pan = viewportPos - imageLocal * _zoom;
            ApplyTransform();

            evt.StopPropagation();
        }

        private void OnPointerDown(PointerDownEvent evt)
        {
            if (_texture == null) return;

            if (evt.clickCount == 2)
            {
                FitToView();
                evt.StopPropagation();
                return;
            }

            if (evt.button != 0 && evt.button != 2) return;

            _dragging = true;
            _dragPointerId = evt.pointerId;
            _dragStartPointer = evt.localPosition;
            _dragStartPan = _pan;

            _viewport.CapturePointer(_dragPointerId);
            evt.StopPropagation();
        }

        private void OnPointerMove(PointerMoveEvent evt)
        {
            if (!_dragging || _dragPointerId != evt.pointerId) return;

            _pan = _dragStartPan + (evt.localPosition - _dragStartPointer);
            ApplyTransform();
            evt.StopPropagation();
        }

        private void OnPointerUp(PointerUpEvent evt)
        {
            if (!_dragging || _dragPointerId != evt.pointerId) return;

            _dragging = false;
            if (_viewport.HasPointerCapture(_dragPointerId))
                _viewport.ReleasePointer(_dragPointerId);

            evt.StopPropagation();
        }

        private void ApplyTransform()
        {
            _image.style.translate = new Translate(_pan.x, _pan.y, 0f);
            _image.style.scale = new Scale(new Vector3(_zoom, _zoom, 1f));
        }
    }
}

