using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;
using Aexis.Samples.Async;

/// <summary>
/// 纯代码创建UIToolkit文件对话框（无UXML/USS，仅依赖已有UIDocument）
/// 核心：System.IO遍历文件 + 纯C#构建UI
/// </summary>
[RequireComponent(typeof(UIDocument))]
public class CodeOnlyFileDialog : MonoBehaviour
{
    // 已有UIDocument引用（自动获取）
    private UIDocument _targetUIDocument;
    private VisualElement _root; // UIDocument的根节点
    
    // 动态创建的文件对话框根容器
    private VisualElement _dialogContainer;
    
    // UI元素引用（纯代码创建）
    private ListView _fileListView;
    private Label _currentPathLabel;
    private Button _filterButton;
    private string _currentFilterKey;
    private Button _confirmBtn;
    private Button _cancelBtn;
    private Button _backBtn;
    private TextField _fileNameInput;

    // ========== 新增：盘符相关 ==========
    private Button _driveButton;
    private string _currentDrive;
    private List<string> _availableDrives; // 可用盘符列表

    private VisualElement _dropdownLayer;
    private VisualElement _dropdownMenu;
    private ScrollView _dropdownScroll;
    private Action<string> _dropdownOnSelect;

    // 文件系统相关
    private string _currentPath;
    private List<FileSystemInfo> _currentFileList = new List<FileSystemInfo>();
    private FileSystemInfo _selectedFile;
    private Dictionary<string, string[]> _filterDict = new Dictionary<string, string[]>
    {
        { "All Files", new[] { "*" } },
        { "Text Files", new[] { "txt" } },
        { "JSON Files", new[] { "json" } },
        { "Image Files", new[] { "png", "jpg", "jpeg", "bmp", "tga", "tif", "tiff", "exr", "gif", "raw", "cr2", "cr3", "nef", "arw", "dng", "raf", "rw2", "orf", "srw", "pef" } }
    };

    // 回调（供外部调用）
    public Action<string> OnFileSelected; // 选中文件路径回调
    public Action OnCanceled;            // 取消回调

    private UniTaskCompletionSource<string> _openTcs;
    private Action<string> _prevOnFileSelected;
    private Action _prevOnCanceled;

    public VisualElement DialogContianer
    {
        get
        {
            return _dialogContainer;
        }
    }

    private void Awake()
    {
        // 获取你现有的UIDocument组件
        _targetUIDocument = GetComponent<UIDocument>();
        _root = _targetUIDocument.rootVisualElement;

        // 初始化默认路径（优先用PersistentDataPath，全平台可读写）
        _currentPath = Application.persistentDataPath;

        // 关键修复：确保根节点不裁剪子元素
        _root.style.overflow = Overflow.Visible;

        // 新增：初始化可用盘符（仅Windows）
        InitAvailableDrives();

        // 1. 纯代码创建文件对话框UI结构
        CreateFileDialogUI();

        // 2. 初始隐藏对话框
        _dialogContainer.style.display = DisplayStyle.None;
    }

    public void EnsureInitialized()
    {
        if (_targetUIDocument == null)
        {
            _targetUIDocument = GetComponent<UIDocument>();
            _root = _targetUIDocument != null ? _targetUIDocument.rootVisualElement : null;
        }

        if (_root == null)
            return;

        _root.style.overflow = Overflow.Visible;

        if (_dialogContainer == null || _dialogContainer.parent == null)
        {
            InitAvailableDrives();
            CreateFileDialogUI();
            _dialogContainer.style.display = DisplayStyle.None;
        }
    }

    public UniTask<string> ShowOpenImageAsync()
    {
        EnsureInitialized();
        if (_root == null)
            return UniTask.FromResult(string.Empty);

        if (_openTcs != null)
        {
            _openTcs.TrySetResult(string.Empty);
            _openTcs = null;
        }

        _openTcs = new UniTaskCompletionSource<string>();
        _prevOnFileSelected = OnFileSelected;
        _prevOnCanceled = OnCanceled;

        OnFileSelected = path =>
        {
            OnFileSelected = _prevOnFileSelected;
            OnCanceled = _prevOnCanceled;
            _openTcs?.TrySetResult(path ?? string.Empty);
            _openTcs = null;
        };

        OnCanceled = () =>
        {
            OnFileSelected = _prevOnFileSelected;
            OnCanceled = _prevOnCanceled;
            _openTcs?.TrySetResult(string.Empty);
            _openTcs = null;
        };

        _currentFilterKey = "Image Files";
        if (_filterButton != null)
            _filterButton.text = _currentFilterKey;

        ShowOpenFileDialog();
        return _openTcs.Task;
    }

    #region 新增：盘符检测与初始化
    /// <summary>
    /// 初始化可用盘符（仅Windows平台）
    /// </summary>
    private void InitAvailableDrives()
    {
        _availableDrives = new List<string>();
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        try
        {
            var drives = Environment.GetLogicalDrives();
            foreach (var d in drives)
            {
                var root = Path.GetPathRoot(d);
                if (string.IsNullOrWhiteSpace(root))
                    continue;
                root = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
                if (!Directory.Exists(root))
                    continue;
                _availableDrives.Add(root);
            }
        }
        catch
        {
        }

        if (_availableDrives.Count == 0)
        {
            try
            {
                DriveInfo[] allDrives = DriveInfo.GetDrives();
                foreach (DriveInfo drive in allDrives)
                {
                    try
                    {
                        var root = drive.RootDirectory.FullName;
                        if (string.IsNullOrWhiteSpace(root))
                            continue;
                        root = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
                        if (!Directory.Exists(root))
                            continue;
                        _availableDrives.Add(root);
                    }
                    catch
                    {
                    }
                }
            }
            catch
            {
            }
        }

        if (_availableDrives.Count == 0)
        {
            var fallback = Path.GetPathRoot(Application.persistentDataPath);
            if (!string.IsNullOrWhiteSpace(fallback))
                _availableDrives.Add(fallback);
        }

#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        // MacOS平台：获取根目录和外接存储（无盘符，显示挂载点）
        try
        {
            DriveInfo[] allDrives = DriveInfo.GetDrives();
            foreach (DriveInfo drive in allDrives)
            {
                if (drive.IsReady)
                {
                    // 过滤掉无用的挂载点，只保留根目录和外接存储
                    if (drive.Name == "/" || drive.DriveType == DriveType.Removable)
                    {
                        _availableDrives.Add(drive.Name); // 如 "/"、"/Volumes/MyUSB"
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Failed to list drives on macOS: {e.Message}");
            _availableDrives.Add(Application.persistentDataPath);
        }

#elif UNITY_STANDALONE_LINUX || UNITY_EDITOR_LINUX
        // Linux平台：获取挂载点
        try
        {
            DriveInfo[] allDrives = DriveInfo.GetDrives();
            foreach (DriveInfo drive in allDrives)
            {
                if (drive.IsReady && !drive.Name.Contains("/sys") && !drive.Name.Contains("/proc"))
                {
                    // 过滤系统临时挂载点，保留有用的存储路径
                    _availableDrives.Add(drive.Name); // 如 "/"、"/mnt/usb"
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Failed to list drives on Linux: {e.Message}");
            _availableDrives.Add(Application.persistentDataPath);
        }

#else
            // 移动端/Editor/WebGL：屏蔽盘符下拉框，不添加任何路径
            _availableDrives.Clear();
#endif
    }
    #endregion

    #region 核心：纯代码创建所有UI元素
    private void CreateFileDialogUI()
    {
        // ========== 1. 创建对话框根容器 ==========
        _dialogContainer = new VisualElement();
        _dialogContainer.name = "file-dialog-container";
        // 设置基础样式
        _dialogContainer.style.backgroundColor = Color.white;
        _dialogContainer.style.borderLeftWidth = 2;
        _dialogContainer.style.borderBottomWidth = 2;
        _dialogContainer.style.borderTopWidth = 2;
        _dialogContainer.style.borderRightWidth = 2;
        //_dialogContainer.style.borderWidth = 2;
        _dialogContainer.style.borderLeftColor = new Color(0.2f, 0.2f, 0.2f);
        _dialogContainer.style.borderRightColor = new Color(0.2f, 0.2f, 0.2f);
        _dialogContainer.style.borderTopColor = new Color(0.2f, 0.2f, 0.2f);
        _dialogContainer.style.borderBottomColor = new Color(0.2f, 0.2f, 0.2f);
        //_dialogContainer.style.borderColor = new Color(0.2f, 0.2f, 0.2f);
        _dialogContainer.style.borderBottomLeftRadius = 5;
        _dialogContainer.style.borderBottomRightRadius = 5;
        _dialogContainer.style.borderTopLeftRadius = 5;
        _dialogContainer.style.borderTopRightRadius = 5;
        //_dialogContainer.style.borderRadius = 5;
        _dialogContainer.style.paddingLeft = new StyleLength(new Length(10, LengthUnit.Pixel));
        _dialogContainer.style.paddingRight = new StyleLength(new Length(10, LengthUnit.Pixel));
        _dialogContainer.style.paddingTop = new StyleLength(new Length(10, LengthUnit.Pixel));
        _dialogContainer.style.paddingBottom = new StyleLength(new Length(10, LengthUnit.Pixel));
        //_dialogContainer.style.padding = new StyleLength(new Length(10, LengthUnit.Pixel));
        _dialogContainer.style.flexDirection = FlexDirection.Column;
        _dialogContainer.style.position = Position.Absolute; // 绝对定位，居中显示
        _dialogContainer.style.width = 1280;
        _dialogContainer.style.height = 800;
        _dialogContainer.style.left = _root.layout.width / 2 - 1280/2;
        _dialogContainer.style.top = _root.layout.height / 2 - 800 / 2;
        _dialogContainer.style.unityFont = _root.style.unityFont;
        _dialogContainer.style.overflow = Overflow.Visible;
        //_dialogContainer.style.zIndex = 2000;
        _root.Add(_dialogContainer);

        _dropdownLayer = new VisualElement();
        _dropdownLayer.style.position = Position.Absolute;
        _dropdownLayer.style.left = 0;
        _dropdownLayer.style.top = 0;
        _dropdownLayer.style.right = 0;
        _dropdownLayer.style.bottom = 0;
        _dropdownLayer.style.display = DisplayStyle.None;
        //_dropdownLayer.style.zIndex = 5000;
        _dropdownLayer.pickingMode = PickingMode.Position;
        _dropdownLayer.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (_dropdownMenu != null && _dropdownMenu.worldBound.Contains(evt.position))
                return;
            HideDropdown();
            evt.StopPropagation();
        });
        _root.Add(_dropdownLayer);

        _dropdownMenu = new VisualElement();
        _dropdownMenu.style.position = Position.Absolute;
        _dropdownMenu.style.width = 220;
        _dropdownMenu.style.maxHeight = 320;
        _dropdownMenu.style.backgroundColor = Color.white;
        _dropdownMenu.style.borderLeftWidth = 1;
        _dropdownMenu.style.borderRightWidth = 1;
        _dropdownMenu.style.borderTopWidth = 1;
        _dropdownMenu.style.borderBottomWidth = 1;
        _dropdownMenu.style.borderLeftColor = new Color(0.2f, 0.2f, 0.2f);
        _dropdownMenu.style.borderRightColor = new Color(0.2f, 0.2f, 0.2f);
        _dropdownMenu.style.borderTopColor = new Color(0.2f, 0.2f, 0.2f);
        _dropdownMenu.style.borderBottomColor = new Color(0.2f, 0.2f, 0.2f);
        _dropdownMenu.style.borderBottomLeftRadius = 6;
        _dropdownMenu.style.borderBottomRightRadius = 6;
        _dropdownMenu.style.borderTopLeftRadius = 6;
        _dropdownMenu.style.borderTopRightRadius = 6;
        _dropdownMenu.style.paddingLeft = 6;
        _dropdownMenu.style.paddingRight = 6;
        _dropdownMenu.style.paddingTop = 6;
        _dropdownMenu.style.paddingBottom = 6;
        _dropdownLayer.Add(_dropdownMenu);

        _dropdownScroll = new ScrollView(ScrollViewMode.Vertical);
        _dropdownScroll.style.flexGrow = 1;
        _dropdownScroll.style.minHeight = 0;
        _dropdownMenu.Add(_dropdownScroll);

        // ========== 2. 创建标题栏 ==========
        Label titleLabel = new Label("File Dialog");
        titleLabel.style.fontSize = 16;
        titleLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
        titleLabel.style.marginBottom = 10;
        titleLabel.style.unityFont = _root.style.unityFont;
        _dialogContainer.Add(titleLabel);

        // ========== 3. 创建路径导航栏 ==========
        VisualElement navBar = new VisualElement();
        navBar.style.flexDirection = FlexDirection.Row;
        navBar.style.alignItems = Align.Center;
        navBar.style.marginBottom = 10;
        navBar.style.unityFont = _root.style.unityFont;
        _dialogContainer.Add(navBar);

        if (_availableDrives.Count > 0)
        {
            var driveLabel = new Label("Drive:");
            driveLabel.style.marginRight = 6;
            navBar.Add(driveLabel);

            _currentDrive = _availableDrives[0];
            _driveButton = new Button(() => ShowDropdown(_driveButton, _availableDrives, OnDriveSelected))
            {
                text = _currentDrive
            };
            _driveButton.style.width = 150;
            _driveButton.style.height = 35;
            _driveButton.style.marginRight = 10;
            navBar.Add(_driveButton);
        }

        // 返回按钮
        _backBtn = new Button();
        _backBtn.text = "← back";
        _backBtn.style.width = 80;
        _backBtn.style.left = 200;
        _backBtn.style.marginRight = 10;
        _backBtn.style.backgroundColor = new Color(0.3f, 0.6f, 0.9f);
        _backBtn.style.color = Color.white;
        //_backBtn.style.borderRadius = 3;
        _backBtn.style.borderBottomLeftRadius = 3;
        _backBtn.style.borderBottomRightRadius = 3;
        _backBtn.style.borderTopLeftRadius = 3;
        _backBtn.style.borderTopRightRadius = 3;
        _backBtn.style.unityFont = _root.style.unityFont;
        _backBtn.clicked += OnBackBtnClicked;
        navBar.Add(_backBtn);

        // 当前路径标签
        _currentPathLabel = new Label(_currentPath);
        _currentPathLabel.style.flexGrow = 1;
        _currentPathLabel.style.fontSize = 12;
        _currentPathLabel.style.left = 250;
        _currentPathLabel.style.color = new Color(0.4f, 0.4f, 0.4f);
        _currentPathLabel.style.fontSize = 20;
        _currentPathLabel.style.unityFont = _root.style.unityFont;
        navBar.Add(_currentPathLabel);

        // ========== 4. 创建文件列表 ==========
        _fileListView = new ListView();
        _fileListView.name = "file-list";
        _fileListView.style.flexGrow = 1;
        _fileListView.style.flexBasis = 0;
        _fileListView.style.minHeight = 0;
        _fileListView.style.backgroundColor = new Color(0.95f, 0.95f, 0.95f);
        //_fileListView.style.borderWidth = 1;
        _fileListView.style.borderBottomLeftRadius = 1;
        _fileListView.style.borderBottomRightRadius = 1;
        _fileListView.style.borderTopLeftRadius = 1;
        _fileListView.style.borderTopRightRadius = 1;
        //_fileListView.style.borderColor = new Color(0.8f, 0.8f, 0.8f);
        _fileListView.style.marginBottom = 10;
        _fileListView.style.unityFont = _root.style.unityFont;
        _fileListView.selectionType = SelectionType.Single;
        _fileListView.virtualizationMethod = CollectionVirtualizationMethod.FixedHeight;
        _fileListView.fixedItemHeight = 30;
        // 配置列表项创建和绑定
        _fileListView.makeItem = MakeFileListItem;
        _fileListView.bindItem = BindFileListItem;
        _fileListView.onSelectionChange += OnFileSelectionChanged;
        _dialogContainer.Add(_fileListView);

        // ========== 5. 创建过滤器 + 文件名输入栏 ==========
        VisualElement filterInputBar = new VisualElement();
        filterInputBar.style.flexDirection = FlexDirection.Row;
        filterInputBar.style.alignItems = Align.Center;
        filterInputBar.style.marginBottom = 10;
        filterInputBar.style.overflow = Overflow.Visible;
        filterInputBar.style.position = Position.Relative; // 提供布局上下文
        filterInputBar.style.width = 600;
        filterInputBar.style.height = 50;
        filterInputBar.style.unityFont = _root.style.unityFont;
        _dialogContainer.Add(filterInputBar);

        var fileTypeLabel = new Label("File Type:");
        fileTypeLabel.style.marginRight = 6;
        filterInputBar.Add(fileTypeLabel);

        _currentFilterKey = _filterDict.Keys.FirstOrDefault() ?? "All Files";
        _filterButton = new Button(() =>
        {
            var options = new List<string>(_filterDict.Keys);
            ShowDropdown(_filterButton, options, OnFilterSelected);
        })
        {
            text = _currentFilterKey
        };
        _filterButton.style.width = 160;
        _filterButton.style.height = 35;
        _filterButton.style.marginRight = 10;
        filterInputBar.Add(_filterButton);


        // 文件名输入框
        _fileNameInput = new TextField();
        _fileNameInput.value = "";
        _fileNameInput.style.flexGrow = 1;
        //_fileNameInput.style.borderWidth = 1;
        _fileNameInput.style.borderLeftWidth = 1;
        _fileNameInput.style.borderRightWidth = 1;
        _fileNameInput.style.borderTopWidth = 1;
        _fileNameInput.style.borderBottomWidth = 1;
        _fileNameInput.style.borderBottomColor = new Color(0.8f, 0.8f, 0.8f);
        _fileNameInput.style.borderTopColor = new Color(0.8f, 0.8f, 0.8f);
        _fileNameInput.style.borderLeftColor = new Color(0.8f, 0.8f, 0.8f);
        _fileNameInput.style.borderRightColor = new Color(0.8f, 0.8f, 0.8f);
        //_fileNameInput.style.borderColor = new Color(0.8f, 0.8f, 0.8f);
        //_fileNameInput.style.borderRadius = 3;
        _fileNameInput.style.borderBottomLeftRadius = 3;
        _fileNameInput.style.borderBottomRightRadius = 3;
        _fileNameInput.style.borderTopLeftRadius = 3;
        _fileNameInput.style.borderTopRightRadius = 3;
        _fileNameInput.style.paddingLeft = 5;
        _fileNameInput.style.paddingRight = 5;
        _fileNameInput.style.left = 200;
        _fileNameInput.style.width = 400;
        _fileNameInput.style.unityFont = _root.style.unityFont;
        filterInputBar.Add(_fileNameInput);

        // ========== 6. 创建按钮栏 ==========
        VisualElement buttonBar = new VisualElement();
        buttonBar.style.flexDirection = FlexDirection.Row;
        buttonBar.style.justifyContent = Justify.FlexEnd;
        _dialogContainer.Add(buttonBar);

        // 确认按钮
        _confirmBtn = new Button();
        _confirmBtn.text = "Confirm";
        _confirmBtn.style.width = 80;
        _confirmBtn.style.marginRight = 10;
        _confirmBtn.style.backgroundColor = new Color(0.2f, 0.8f, 0.2f);
        _confirmBtn.style.color = Color.white;
        //_confirmBtn.style.borderRadius = 3;
        _confirmBtn.style.borderBottomLeftRadius = 3;
        _confirmBtn.style.borderBottomRightRadius = 3;
        _confirmBtn.style.borderTopLeftRadius = 3;
        _confirmBtn.style.borderTopRightRadius = 3;
        _confirmBtn.style.unityFont = _root.style.unityFont;
        _confirmBtn.clicked += OnConfirmBtnClicked;
        buttonBar.Add(_confirmBtn);

        // 取消按钮
        _cancelBtn = new Button();
        _cancelBtn.text = "Cancel";
        _cancelBtn.style.width = 80;
        _cancelBtn.style.backgroundColor = new Color(0.8f, 0.2f, 0.2f);
        _cancelBtn.style.color = Color.white;
        //_cancelBtn.style.borderRadius = 3;
        _cancelBtn.style.borderBottomLeftRadius = 3;
        _cancelBtn.style.borderBottomRightRadius = 3;
        _cancelBtn.style.borderTopLeftRadius = 3;
        _cancelBtn.style.borderTopRightRadius = 3;
        _cancelBtn.style.unityFont = _root.style.unityFont;
        _cancelBtn.clicked += OnCancelBtnClicked;
        buttonBar.Add(_cancelBtn);


    }

    //private void Start()
    //{
    //    // ========== 方案2：监听点击事件，动态修改菜单样式 ==========
    //    if (_filterPopup != null)
    //    {
    //        // 监听过滤器下拉框的点击事件
    //        _filterPopup.RegisterCallback<ClickEvent>(evt =>
    //        {
    //            // 延迟获取菜单（点击后菜单已创建）
    //            _root.schedule.Execute(() =>
    //            {
    //                var filterMenu = _filterPopup.Q("unity-popup__menu");
    //                if (filterMenu != null)
    //                {
    //                    filterMenu.style.width = _filterPopup.layout.width;
    //                    filterMenu.style.minWidth = _filterPopup.layout.width;
    //                    filterMenu.style.maxHeight = 200;
    //                    filterMenu.style.backgroundColor = Color.white;
    //                    //filterMenu.style.borderWidth = 1;
    //                    //filterMenu.style.borderColor = new Color(0.5f, 0.5f, 0.5f);
    //                    //filterMenu.style.borderRadius = 3;
    //                    //filterMenu.style.padding = 2;
    //                    filterMenu.style.overflow = Overflow.Visible;

    //                    foreach (var item in filterMenu.Children())
    //                    {
    //                        item.style.height = 25;
    //                        item.style.paddingLeft = 5;
    //                        item.style.paddingRight = 5;
    //                        item.RegisterCallback<MouseEnterEvent>(e => item.style.backgroundColor = new Color(0.8f, 0.9f, 1f));
    //                        item.RegisterCallback<MouseLeaveEvent>(e => item.style.backgroundColor = Color.white);
    //                    }

    //                    filterMenu.style.position = Position.Absolute;
    //                    filterMenu.style.left = _filterPopup.layout.x;
    //                    filterMenu.style.top = _filterPopup.layout.y + _filterPopup.layout.height;
    //                    //filterMenu.style.SetProperty("z-index", 3000);
    //                }
    //            }).StartingIn(10);
    //        });
    //    }

    //    if (_drivePopup != null)
    //    {
    //        // 监听盘符下拉框的点击事件
    //        _drivePopup.RegisterCallback<ClickEvent>(evt =>
    //        {
    //            _root.schedule.Execute(() =>
    //            {
    //                var driveMenu = _drivePopup.Q("unity-popup__menu");
    //                if (driveMenu != null)
    //                {
    //                    driveMenu.style.width = _drivePopup.layout.width;
    //                    driveMenu.style.minWidth = _drivePopup.layout.width;
    //                    driveMenu.style.maxHeight = 200;
    //                    driveMenu.style.backgroundColor = Color.white;
    //                    //driveMenu.style.borderWidth = 1;
    //                    //driveMenu.style.borderColor = new Color(0.5f, 0.5f, 0.5f);
    //                    //driveMenu.style.borderRadius = 3;
    //                    //driveMenu.style.padding = 2;
    //                    driveMenu.style.overflow = Overflow.Visible;

    //                    foreach (var item in driveMenu.Children())
    //                    {
    //                        item.style.height = 25;
    //                        item.style.paddingLeft = 5;
    //                        item.style.paddingRight = 5;
    //                        item.RegisterCallback<MouseEnterEvent>(e => item.style.backgroundColor = new Color(0.8f, 0.9f, 1f));
    //                        item.RegisterCallback<MouseLeaveEvent>(e => item.style.backgroundColor = Color.white);
    //                    }

    //                    driveMenu.style.position = Position.Absolute;
    //                    driveMenu.style.left = _drivePopup.layout.x;
    //                    driveMenu.style.top = _drivePopup.layout.y + _drivePopup.layout.height;
    //                    //driveMenu.style.SetProperty("z-index", 3000);
    //                }
    //            }).StartingIn(10);
    //        });
    //    }
    //}

    // 创建文件列表项（纯代码）
    private VisualElement MakeFileListItem()
    {
        VisualElement item = new VisualElement();
        item.style.flexDirection = FlexDirection.Row;
        item.style.alignItems = Align.Center;
        item.style.height = 30;
        item.style.paddingLeft = 5;

        // 图标占位（无图片也不影响功能）
        VisualElement iconPlaceholder = new VisualElement();
        iconPlaceholder.style.width = 20;
        iconPlaceholder.style.height = 20;
        iconPlaceholder.style.marginRight = 10;
        item.Add(iconPlaceholder);

        // 文件名标签
        Label nameLabel = new Label();
        nameLabel.style.flexGrow = 1;
        item.Add(nameLabel);

        // 存储子元素引用（用于绑定数据）
        item.userData = new Dictionary<string, VisualElement>
        {
            { "icon", iconPlaceholder },
            { "name", nameLabel }
        };

        // 列表项hover效果
        item.RegisterCallback<MouseEnterEvent>(evt =>
        {
            item.style.backgroundColor = new Color(0.9f, 0.9f, 0.9f);
        });
        item.RegisterCallback<MouseLeaveEvent>(evt =>
        {
            item.style.backgroundColor = new Color(0,0,0,0);
        });

        return item;
    }

    // 绑定列表项数据
    private void BindFileListItem(VisualElement item, int index)
    {
        if (index >= _currentFileList.Count) return;

        FileSystemInfo fsInfo = _currentFileList[index];
        Dictionary<string, VisualElement> elements = (Dictionary<string, VisualElement>)item.userData;
        VisualElement icon = elements["icon"];
        Label nameLabel = (Label)elements["name"];

        // 区分目录和文件，设置样式和文本
        if (fsInfo is DirectoryInfo dirInfo)
        {
            // 目录：蓝色背景占位 + 蓝色文字
            icon.style.backgroundColor = new Color(0.2f, 0.5f, 0.8f);
            nameLabel.text = dirInfo.Name == string.Empty ? ".." : dirInfo.Name;
            nameLabel.style.color = new Color(0.2f, 0.5f, 0.8f);
        }
        else if (fsInfo is FileInfo fileInfo)
        {
            // 文件：灰色背景占位 + 黑色文字
            icon.style.backgroundColor = new Color(0.7f, 0.7f, 0.7f);
            nameLabel.text = fileInfo.Name;
            nameLabel.style.color = Color.black;
        }
    }
    #endregion

    #region 文件系统逻辑（纯System.IO，无平台依赖）
    // 加载指定路径的文件/目录
    private void LoadFiles(string path)
    {
        try
        {
            _currentFileList.Clear();

            // 路径合法性检查
            if (!Directory.Exists(path))
            {
                Debug.LogWarning($"Path not found: {path}. Falling back to persistentDataPath.");
                path = Application.persistentDataPath;
            }

            // 更新当前路径
            _currentPath = path;
            _currentPathLabel.text = path;

            // 1. 添加上级目录（..）
            DirectoryInfo parentDir = Directory.GetParent(path);
            if (parentDir != null)
            {
                _currentFileList.Add(parentDir);
            }

            // 2. 添加子目录
            DirectoryInfo[] subDirs = Array.Empty<DirectoryInfo>();
            try { subDirs = new DirectoryInfo(path).GetDirectories(); } catch { }
            foreach (DirectoryInfo subDir in subDirs)
            {
                _currentFileList.Add(subDir);
            }

            // 3. 添加文件（按过滤器筛选）
            if (string.IsNullOrWhiteSpace(_currentFilterKey) || !_filterDict.ContainsKey(_currentFilterKey))
                _currentFilterKey = _filterDict.Keys.FirstOrDefault() ?? "All Files";
            string[] currentFilter = _filterDict[_currentFilterKey];
            FileInfo[] files = Array.Empty<FileInfo>();
            try { files = new DirectoryInfo(path).GetFiles(); } catch { }
            foreach (FileInfo file in files)
            {
                if (currentFilter.Length > 0 && currentFilter[0] == "*")
                {
                    _currentFileList.Add(file);
                    continue;
                }

                var extNoDot = (file.Extension ?? "").TrimStart('.');
                if (Array.Exists(currentFilter, ext => string.Equals(extNoDot, ext, StringComparison.OrdinalIgnoreCase)))
                {
                    _currentFileList.Add(file);
                }
            }

            // 更新列表数据
            _fileListView.itemsSource = _currentFileList;
            _fileListView.Rebuild();

            // 重置选中状态
            _selectedFile = null;
            _fileNameInput.value = string.Empty;
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load files: {e.Message}");
        }
    }
    #endregion

    #region UI交互逻辑
    // 返回上一级目录
    private void OnBackBtnClicked()
    {
        DirectoryInfo parentDir = Directory.GetParent(_currentPath);
        if (parentDir != null)
        {
            LoadFiles(parentDir.FullName);
        }
    }

    // 文件/目录选中
    private void OnFileSelectionChanged(IEnumerable<object> selectedItems)
    {
        foreach (var item in selectedItems)
        {
            _selectedFile = item as FileSystemInfo;
            if (_selectedFile == null) continue;

            // 目录：点击进入
            if (_selectedFile is DirectoryInfo)
            {
                LoadFiles(_selectedFile.FullName);
            }
            // 文件：填充到输入框
            else if (_selectedFile is FileInfo fileInfo)
            {
                _fileNameInput.value = fileInfo.Name;
            }
            break;
        }
    }

    public string finalPath = string.Empty;
    // 确认选择
    private void OnConfirmBtnClicked()
    {
        if (string.IsNullOrEmpty(_fileNameInput.value))
        {
            Debug.LogWarning("Please enter a file name.");
            return;
        }
        
        // 打开模式：返回选中文件路径；保存模式：拼接路径+文件名
        if (_selectedFile is FileInfo)
        {
            finalPath = _selectedFile.FullName;
        }
        else
        {
            finalPath = Path.Combine(_currentPath, _fileNameInput.value);
        }

        // 触发回调
        OnFileSelected?.Invoke(finalPath);
        HideDialog();
    }

    // 取消选择
    private void OnCancelBtnClicked()
    {
        OnCanceled?.Invoke();
        HideDialog();
    }
    #endregion

    #region 公开调用接口（给外部使用）
    /// <summary>
    /// 显示打开文件对话框
    /// </summary>
    public void ShowOpenFileDialog()
    {
        HideDropdown();
        _dialogContainer.style.display = DisplayStyle.Flex;
        _confirmBtn.text = "Open";
        _dialogContainer.MarkDirtyRepaint();
        _root.schedule.Execute(() => { LoadFiles(_currentPath); }).StartingIn(10);
    }

    /// <summary>
    /// 显示保存文件对话框
    /// </summary>
    /// <param name="defaultFileName">默认文件名</param>
    public void ShowSaveFileDialog(string defaultFileName = "new_file")
    {
        HideDropdown();
        _dialogContainer.style.display = DisplayStyle.Flex;
        _confirmBtn.text = "Save";
        _fileNameInput.value = defaultFileName;
        _dialogContainer.MarkDirtyRepaint();
        _root.schedule.Execute(() => { LoadFiles(_currentPath); }).StartingIn(10);
    }

    /// <summary>
    /// 隐藏对话框
    /// </summary>
    public void HideDialog()
    {
        HideDropdown();
        _dialogContainer.style.display = DisplayStyle.None;
        _selectedFile = null;
        _fileNameInput.value = string.Empty;
    }

    /// <summary>
    /// 添加自定义文件过滤器
    /// </summary>
    /// <param name="filterName">过滤器名称（如：脚本文件）</param>
    /// <param name="extensions">扩展名数组（如：cs, lua）</param>
    public void AddCustomFilter(string filterName, string[] extensions)
    {
        if (!_filterDict.ContainsKey(filterName))
        {
            _filterDict.Add(filterName, extensions);
            if (string.IsNullOrWhiteSpace(_currentFilterKey))
                _currentFilterKey = filterName;
        }
    }

    private void ShowDropdown(Button anchor, List<string> options, Action<string> onSelect)
    {
        EnsureInitialized();
        if (_root == null || _dropdownLayer == null || _dropdownMenu == null || _dropdownScroll == null || anchor == null)
            return;

        _dropdownLayer.style.display = DisplayStyle.Flex;
        _dropdownLayer.BringToFront();

        _dropdownOnSelect = onSelect;
        _dropdownScroll.contentContainer.Clear();

        for (int i = 0; i < options.Count; i++)
        {
            var option = options[i] ?? "";
            var btn = new Button(() =>
            {
                _dropdownOnSelect?.Invoke(option);
                HideDropdown();
            })
            {
                text = option
            };
            btn.style.height = 26;
            btn.style.unityTextAlign = TextAnchor.MiddleLeft;
            btn.style.paddingLeft = 6;
            btn.style.paddingRight = 6;
            btn.style.marginBottom = 2;
            btn.style.backgroundColor = Color.white;
            btn.style.color = Color.black;
            _dropdownScroll.Add(btn);
        }

        var rootWorld = _root.worldBound;
        var anchorWorld = anchor.worldBound;

        var left = anchorWorld.xMin - rootWorld.xMin;
        var top = anchorWorld.yMax - rootWorld.yMin;
        var width = Mathf.Max(160f, anchorWorld.width);

        var maxLeft = Mathf.Max(0f, rootWorld.width - width - 6f);
        left = Mathf.Clamp(left, 6f, maxLeft);

        _dropdownMenu.style.left = left;
        _dropdownMenu.style.top = top;
        _dropdownMenu.style.width = width;
        _dropdownMenu.BringToFront();
    }

    private void HideDropdown()
    {
        if (_dropdownLayer == null)
            return;
        _dropdownLayer.style.display = DisplayStyle.None;
        _dropdownOnSelect = null;
        if (_dropdownScroll != null)
            _dropdownScroll.contentContainer.Clear();
    }

    private void OnDriveSelected(string drive)
    {
        if (string.IsNullOrWhiteSpace(drive))
            return;
        if (!Directory.Exists(drive))
        {
            Debug.LogWarning($"Drive not accessible: {drive}");
            return;
        }

        _currentDrive = drive;
        if (_driveButton != null)
            _driveButton.text = _currentDrive;
        _currentPath = drive;
        LoadFiles(_currentPath);
    }

    private void OnFilterSelected(string filterKey)
    {
        if (string.IsNullOrWhiteSpace(filterKey))
            return;
        if (!_filterDict.ContainsKey(filterKey))
            return;

        _currentFilterKey = filterKey;
        if (_filterButton != null)
            _filterButton.text = _currentFilterKey;
        LoadFiles(_currentPath);
    }
    #endregion

    private float _driveRefreshTimer;
    // 适配分辨率变化，保证对话框居中
    private void Update()
    {
        if (_dialogContainer.style.display == DisplayStyle.Flex)
        {
            _dialogContainer.style.left = _root.layout.width / 2 - 1280 / 2;
            _dialogContainer.style.top = _root.layout.height / 2 - 800 / 2;

            // 每2秒刷新一次盘符列表
            _driveRefreshTimer += Time.deltaTime;
            if (_driveRefreshTimer >= 2f && _availableDrives.Count > 0)
            {
                InitAvailableDrives();
                if (_driveButton != null && _availableDrives.Count > 0)
                {
                    if (string.IsNullOrWhiteSpace(_currentDrive) || !_availableDrives.Contains(_currentDrive))
                        _currentDrive = _availableDrives[0];
                    _driveButton.text = _currentDrive;
                }
                _driveRefreshTimer = 0;
            }
        }
    }
}
