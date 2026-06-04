using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

/// <summary>
/// LibraryView - 图库页面
/// 左侧：盘符选择、目录浏览、树状浏览
/// 右侧：缩略图展示（按时间、人脸、地点分类）
/// </summary>
public class LibraryView : BasePageView
{
    private TwoPaneSplitView _mainSplitView;
    private PopupField<string> _drivePopup;
    private TreeView _directoryTree;
    private VisualElement _thumbnailContainer;
    private ScrollView _thumbnailScrollView;
    private VisualElement _thumbnailGrid;
    private VisualElement _toastOverlay;
    private VisualElement _thumbnailTooltip;

    private string _currentDriveRoot;
    private readonly HashSet<int> _loadedDirectoryIds = new HashSet<int>();
    private bool _suppressTreeEvents;
    private readonly List<ImageFileEntry> _imageFiles = new List<ImageFileEntry>();

    private const int MaxDirectoryDepth = 6;
    private const int MaxChildrenPerDirectory = 250;
    private const int ThumbnailSize = 240; // 加大一倍

    private static readonly HashSet<string> ImageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".psd", ".tiff", ".tif", ".exr", ".raw", ".cr2", ".nef", ".arw"
    };

    public event Action<string> OnImageDoubleClicked;



    public override void BuildPage()
    {
        _pageContainer = new VisualElement();
        _pageContainer.style.width = Length.Percent(100);
        _pageContainer.style.height = Length.Percent(100);
        _pageContainer.style.position = Position.Relative;
        _pageContainer.style.flexDirection = FlexDirection.Column;

        // 主分割视图
        _mainSplitView = new TwoPaneSplitView(0, IsLandscape() ? 280 : 200, TwoPaneSplitViewOrientation.Horizontal);
        _mainSplitView.style.flexGrow = 1;
        _mainSplitView.style.minHeight = 0;
        _pageContainer.Add(_mainSplitView);

        // 左侧面板
        var leftPane = new VisualElement();
        leftPane.style.flexGrow = 1;
        leftPane.style.flexBasis = 0;
        leftPane.style.minWidth = IsLandscape() ? 200 : 150;
        leftPane.style.maxWidth = IsLandscape() ? 360 : 250;
        leftPane.style.flexDirection = FlexDirection.Column;
        _mainSplitView.Add(leftPane);

        // 右侧面板
        var rightPane = new VisualElement();
        rightPane.style.flexGrow = 1;
        rightPane.style.flexBasis = 0;
        rightPane.style.minWidth = 320;
        rightPane.style.flexDirection = FlexDirection.Column;
        _mainSplitView.Add(rightPane);

        // 构建左侧目录浏览器
        BuildDirectoryBrowser(leftPane);

        // 构建右侧缩略图区域
        BuildThumbnailArea(rightPane);

        // Toast提示
        _toastOverlay = BuildToast();
        _pageContainer.Add(_toastOverlay);

        // 缩略图悬停提示
        BuildThumbnailTooltip();

        // 页面切换指示器
        BuildPageIndicator(PageType.LibraryView);

        // 初始化盘符
        PopulateDrives();
    }

    private void BuildDirectoryBrowser(VisualElement parent)
    {
        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.paddingLeft = 8;
        header.style.paddingRight = 8;
        header.style.paddingTop = 6;
        header.style.paddingBottom = 6;
        header.style.backgroundColor = new StyleColor(new Color(0.15f, 0.15f, 0.15f, 1f));
        parent.Add(header);

        var driveLabel = new Label("盘符");
        driveLabel.style.minWidth = 40;
        driveLabel.style.color = Color.white;
        header.Add(driveLabel);

        _drivePopup = new PopupField<string>(new List<string> { "" }, 0);
        _drivePopup.style.flexGrow = 1;
        _drivePopup.SetEnabled(false);
        _drivePopup.RegisterValueChangedCallback(evt => SetDrive(evt.newValue));
        header.Add(_drivePopup);

        // 目录树
        _directoryTree = new TreeView();
        _directoryTree.style.flexGrow = 1;
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

    private void BuildThumbnailArea(VisualElement parent)
    {
        // 顶部工具栏
        var toolbar = new VisualElement();
        toolbar.style.flexDirection = FlexDirection.Row;
        toolbar.style.alignItems = Align.Center;
        toolbar.style.paddingLeft = 8;
        toolbar.style.paddingRight = 8;
        toolbar.style.paddingTop = 6;
        toolbar.style.paddingBottom = 6;
        toolbar.style.backgroundColor = new StyleColor(new Color(0.15f, 0.15f, 0.15f, 1f));
        parent.Add(toolbar);

        var title = new Label("图库浏览");
        title.style.flexGrow = 1;
        title.style.color = Color.white;
        toolbar.Add(title);

        // 分类按钮
        var sortByTime = new Button(() => SortBy("time")) { text = "时间" };
        toolbar.Add(sortByTime);
        var sortByFace = new Button(() => SortBy("face")) { text = "人脸" };
        toolbar.Add(sortByFace);
        var sortByLocation = new Button(() => SortBy("location")) { text = "地点" };
        toolbar.Add(sortByLocation);

        // 缩略图滚动区域
        _thumbnailScrollView = new ScrollView(ScrollViewMode.Vertical);
        _thumbnailScrollView.style.flexGrow = 1;
        _thumbnailScrollView.style.minHeight = 0;
        parent.Add(_thumbnailScrollView);

        _thumbnailGrid = new VisualElement();
        _thumbnailGrid.style.flexDirection = FlexDirection.Row;
        _thumbnailGrid.style.flexWrap = Wrap.Wrap;
        _thumbnailGrid.style.paddingLeft = 8;
        _thumbnailGrid.style.paddingRight = 8;
        _thumbnailGrid.style.paddingTop = 8;
        _thumbnailGrid.style.paddingBottom = 8;
        _thumbnailScrollView.Add(_thumbnailGrid);
    }

    private void BuildThumbnailTooltip()
    {
        _thumbnailTooltip = new VisualElement();
        _thumbnailTooltip.style.position = Position.Absolute;
        _thumbnailTooltip.style.display = DisplayStyle.None;
        _thumbnailTooltip.style.backgroundColor = new StyleColor(new Color(0.1f, 0.1f, 0.1f, 0.95f));
        _thumbnailTooltip.style.borderTopLeftRadius = 6;
        _thumbnailTooltip.style.borderTopRightRadius = 6;
        _thumbnailTooltip.style.borderBottomLeftRadius = 6;
        _thumbnailTooltip.style.borderBottomRightRadius = 6;
        _thumbnailTooltip.style.paddingLeft = 10;
        _thumbnailTooltip.style.paddingRight = 10;
        _thumbnailTooltip.style.paddingTop = 8;
        _thumbnailTooltip.style.paddingBottom = 8;
        _thumbnailTooltip.style.maxWidth = 300;

        var tooltipText = new Label();
        tooltipText.style.color = Color.white;
        tooltipText.style.whiteSpace = WhiteSpace.Normal;
        tooltipText.name = "tooltip-text";
        _thumbnailTooltip.Add(tooltipText);

        _pageContainer.Add(_thumbnailTooltip);
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
                .Where(d => Directory.Exists(d))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(s => s, StringComparer.OrdinalIgnoreCase)
                .ToList();
#else
            drives = DriveInfo.GetDrives()
                .Where(d => d.IsReady)
                .Select(d => d.RootDirectory.FullName)
                .OrderBy(s => s, StringComparer.OrdinalIgnoreCase)
                .ToList();
#endif
        }
        catch
        {
            drives = new List<string>();
        }

        if (drives.Count == 0)
        {
            var fallback = Path.GetPathRoot(Application.persistentDataPath);
            drives.Add(fallback);
        }

        _drivePopup.choices = drives;
        _drivePopup.SetValueWithoutNotify(drives[0]);
        _drivePopup.SetEnabled(true);
        SetDrive(drives[0]);
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

        if (depth >= MaxDirectoryDepth)
            return new TreeViewItemData<DirectoryEntry>(id, entry, new List<TreeViewItemData<DirectoryEntry>>());

        List<TreeViewItemData<DirectoryEntry>> children;
        if (_loadedDirectoryIds.Contains(id))
        {
            children = EnumerateDirectoriesSafe(directoryPath, MaxChildrenPerDirectory)
                .Select(p => BuildDirectoryItem(p, DirectoryNameFromPath(p), depth + 1))
                .ToList();
        }
        else if (depth < MaxDirectoryDepth)
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

        try
        {
            return Directory.EnumerateDirectories(directoryPath).Take(maxCount).ToArray();
        }
        catch
        {
            return Array.Empty<string>();
        }
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

    private void EnsureDirectoryChildrenLoaded(int parentId, string parentPath)
    {
        var parentKeyId = StableId(parentPath);
        if (_loadedDirectoryIds.Contains(parentKeyId))
            return;

        _loadedDirectoryIds.Add(parentKeyId);

        var childDirs = EnumerateDirectoriesSafe(parentPath, MaxChildrenPerDirectory).ToList();
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

            var childChildren = new List<TreeViewItemData<DirectoryEntry>> { BuildPlaceholderItem(childPath) };
            var item = new TreeViewItemData<DirectoryEntry>(childId, childEntry, childChildren);
            var rebuildTree = i == childDirs.Count - 1;
            _directoryTree.AddItem(item, parentId, -1, rebuildTree);
        }

        if (childDirs.Count == 0)
            _directoryTree.RefreshItems();
    }

    private void OnDirectorySelectionChanged(IEnumerable<object> selectedItems)
    {
        var first = selectedItems.FirstOrDefault();
        if (first is not DirectoryEntry entry) return;
        if (entry.isPlaceholder) return;
        if (string.IsNullOrWhiteSpace(entry.path)) return;

        RefreshThumbnails(entry.path);
    }

    private void RefreshThumbnails(string directoryPath)
    {
        _imageFiles.Clear();
        _thumbnailGrid.Clear();

        if (!Directory.Exists(directoryPath))
            return;

        try
        {
            var files = Directory.EnumerateFiles(directoryPath)
                .Where(IsImageFile)
                .Select(p => new ImageFileEntry
                {
                    fullPath = p,
                    fileName = Path.GetFileName(p),
                    fileType = DetermineImageType(p)
                })
                .OrderBy(f => f.fileName, StringComparer.OrdinalIgnoreCase)
                .ToList();

            _imageFiles.AddRange(files);

            foreach (var file in _imageFiles)
            {
                CreateThumbnail(file);
            }
        }
        catch
        {
        }
    }

    private void CreateThumbnail(ImageFileEntry file)
    {
        var thumbnail = new VisualElement();
        thumbnail.style.width = ThumbnailSize;
        thumbnail.style.height = ThumbnailSize + 30;
        thumbnail.style.marginLeft = 6;
        thumbnail.style.marginRight = 6;
        thumbnail.style.marginTop = 6;
        thumbnail.style.marginBottom = 6;
        thumbnail.style.backgroundColor = new StyleColor(new Color(0.2f, 0.2f, 0.2f, 1f));
        thumbnail.style.borderTopLeftRadius = 8;
        thumbnail.style.borderTopRightRadius = 8;
        thumbnail.style.borderBottomLeftRadius = 8;
        thumbnail.style.borderBottomRightRadius = 8;
        thumbnail.style.flexDirection = FlexDirection.Column;
        thumbnail.style.alignItems = Align.Center;

        // 图片显示
        var imageBox = new Image();
        imageBox.style.width = ThumbnailSize;
        imageBox.style.height = ThumbnailSize;
        imageBox.style.backgroundColor = new StyleColor(new Color(0.15f, 0.15f, 0.15f, 1f));
        imageBox.scaleMode = ScaleMode.ScaleToFit;

        // 异步加载缩略图
        LoadThumbnailAsync(file.fullPath, imageBox);

        thumbnail.Add(imageBox);

        // 文件名
        var nameLabel = new Label(file.fileName);
        nameLabel.style.width = ThumbnailSize;
        nameLabel.style.color = Color.white;
        nameLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
        nameLabel.style.fontSize = 10;
        nameLabel.style.whiteSpace = WhiteSpace.NoWrap;
        nameLabel.style.overflow = Overflow.Hidden;
        nameLabel.style.textOverflow = TextOverflow.Ellipsis;
        thumbnail.Add(nameLabel);

        // 类型标记
        var typeColor = file.fileType switch
        {
            ImageFileType.Original => new Color(0.3f, 0.8f, 0.3f, 1f),
            ImageFileType.Edited => new Color(0.8f, 0.6f, 0.3f, 1f),
            _ => new Color(0.5f, 0.5f, 0.5f, 1f)
        };

        var typeIndicator = new VisualElement();
        typeIndicator.style.position = Position.Absolute;
        typeIndicator.style.top = 4;
        typeIndicator.style.right = 4;
        typeIndicator.style.width = 12;
        typeIndicator.style.height = 12;
        typeIndicator.style.backgroundColor = new StyleColor(typeColor);
        typeIndicator.style.borderTopLeftRadius = 6;
        typeIndicator.style.borderTopRightRadius = 6;
        typeIndicator.style.borderBottomLeftRadius = 6;
        typeIndicator.style.borderBottomRightRadius = 6;
        imageBox.Add(typeIndicator);

        // 单击选中
        bool isSelected = false;
        thumbnail.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (evt.button != 0) return;
            if (evt.clickCount == 1)
            {
                isSelected = !isSelected;
                thumbnail.style.borderLeftWidth = isSelected ? 3 : 0;
                thumbnail.style.borderRightWidth = isSelected ? 3 : 0;
                thumbnail.style.borderTopWidth = isSelected ? 3 : 0;
                thumbnail.style.borderBottomWidth = isSelected ? 3 : 0;
                thumbnail.style.borderLeftColor = new StyleColor(Color.cyan);
                thumbnail.style.borderRightColor = new StyleColor(Color.cyan);
                thumbnail.style.borderTopColor = new StyleColor(Color.cyan);
                thumbnail.style.borderBottomColor = new StyleColor(Color.cyan);
                ShowThumbnailInfo(evt.position, file);
            }
            else if (evt.clickCount == 2)
            {
                // 双击跳转到MainView2
                OnImageDoubleClicked?.Invoke(file.fullPath);
            }
            evt.StopPropagation();
        });

        // 悬停提示
        thumbnail.RegisterCallback<PointerEnterEvent>(evt =>
        {
            ShowThumbnailInfo(evt.position, file);
        });

        thumbnail.RegisterCallback<PointerLeaveEvent>(evt =>
        {
            HideThumbnailInfo();
        });

        _thumbnailGrid.Add(thumbnail);
    }

    private void ShowThumbnailInfo(Vector2 position, ImageFileEntry file)
    {
        if (_thumbnailTooltip == null) return;

        var tooltipText = _thumbnailTooltip.Q<Label>("tooltip-text");
        if (tooltipText != null)
        {
            var info = $"{file.fileName}\n";
            info += $"类型: {GetFileTypeLabel(file.fileType)}\n";
            info += $"路径: {file.fullPath}\n";
            info += "时间: 2024-01-01 12:00\n"; // 占位
            info += "地点: 未知\n"; // 占位，未来从EXIF读取
            info += "CLIP标签: 待分析"; // 占位，未来调用ClipRunner

            tooltipText.text = info;
        }

        _thumbnailTooltip.style.left = position.x + 10;
        _thumbnailTooltip.style.top = position.y + 10;
        _thumbnailTooltip.style.display = DisplayStyle.Flex;
        _thumbnailTooltip.BringToFront();
    }

    private void HideThumbnailInfo()
    {
        if (_thumbnailTooltip != null)
            _thumbnailTooltip.style.display = DisplayStyle.None;
    }

    private string GetFileTypeLabel(ImageFileType type)
    {
        return type switch
        {
            ImageFileType.Original => "原图",
            ImageFileType.Edited => "已编辑",
            _ => "未知"
        };
    }

    private static bool IsImageFile(string filePath)
    {
        var ext = Path.GetExtension(filePath);
        return !string.IsNullOrEmpty(ext) && ImageExtensions.Contains(ext);
    }

    private ImageFileType DetermineImageType(string filePath)
    {
        // 简单判断逻辑，未来可以更复杂
        var fileName = Path.GetFileName(filePath).ToLowerInvariant();
        var ext = Path.GetExtension(filePath).ToLowerInvariant();

        // 原图标记
        if (ext == ".raw" || ext == ".cr2" || ext == ".nef" || ext == ".arw")
            return ImageFileType.Original;

        // 判断是否是编辑过的（文件名包含特定标记）
        if (fileName.Contains("edit") || fileName.Contains("_1") || fileName.Contains("_修改"))
            return ImageFileType.Edited;

        return ImageFileType.Unknown;
    }

    private void SortBy(string sortType)
    {
        ShowToast(_toastOverlay, $"按{sortType}排序");
    }

    private async void LoadThumbnailAsync(string imagePath, Image imageElement)
    {
        if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
            return;

        try
        {
            await System.Threading.Tasks.Task.Run(() =>
            {
                // 在后台线程加载
            });

            // 在主线程加载纹理
            byte[] imageData = File.ReadAllBytes(imagePath);
            var texture = new Texture2D(2, 2);
            if (texture.LoadImage(imageData, false))
            {
                texture.wrapMode = TextureWrapMode.Clamp;
                texture.filterMode = FilterMode.Bilinear;
                imageElement.image = texture;
            }
            else
            {
                UnityEngine.Object.Destroy(texture);
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"Failed to load thumbnail: {imagePath}, Error: {ex.Message}");
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
        public ImageFileType fileType;
    }

    private enum ImageFileType
    {
        Unknown,
        Original,  // 原图（raw格式或标记）
        Edited     // 编辑过的图
    }
}
