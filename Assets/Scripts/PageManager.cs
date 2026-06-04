using System.IO;
using UnityEngine;
using UnityEngine.UIElements;

/// <summary>
/// 页面管理器，协调三个页面（MainView2, LibraryView, DesignView）的切换
/// 挂载在Main2.unity场景的UIDocument对象上
/// </summary>
[DisallowMultipleComponent]
[RequireComponent(typeof(UIDocument))]
public class PageManager : MonoBehaviour
{
    private const string PrefKeyLastImagePath = "MainView.LastImagePath";

    private UIDocument _uiDocument;

    private MainView2 _mainView2;
    private LibraryView _libraryView;
    private DesignView _designView;

    private BasePageView.PageType _currentPage = BasePageView.PageType.MainView2;

    private Texture2D _sharedCurrentImage;
    private string _sharedCurrentImagePath;

    private void Awake()
    {
        _uiDocument = GetComponent<UIDocument>();

        // 初始化三个页面
        InitializePages();
    }

    private void OnEnable()
    {
        if (!Application.isPlaying)
            return;

        // 加载上次编辑的图片
        RestoreLastImage();

        // 默认显示MainView2
        SwitchToPage(BasePageView.PageType.MainView2);
    }

    private void OnDisable()
    {
        if (!Application.isPlaying)
            return;
    }

    private void OnDestroy()
    {
        CleanupPages();
    }

    private void InitializePages()
    {
        // 创建MainView2
        _mainView2 = GetComponent<MainView2>();
        if (_mainView2==null)
            _mainView2 = gameObject.AddComponent<MainView2>();
        _mainView2.OnRequestPageSwitch += OnPageSwitchRequested;
        _mainView2.BuildPage();

        // 创建LibraryView
        _libraryView= gameObject.GetComponent<LibraryView>();
        if (_libraryView==null)
            _libraryView = gameObject.AddComponent<LibraryView>();
        _libraryView.OnRequestPageSwitch += OnPageSwitchRequested;
        _libraryView.OnImageDoubleClicked += OnLibraryImageDoubleClicked;
        _libraryView.BuildPage();

        // 创建DesignView
        _designView = GetComponent<DesignView>();
        if (_designView==null)
            _designView = gameObject.AddComponent<DesignView>();
        _designView.OnRequestPageSwitch += OnPageSwitchRequested;
        _designView.BuildPage();
    }

    private void CleanupPages()
    {
        if (_mainView2 != null)
        {
            _mainView2.OnRequestPageSwitch -= OnPageSwitchRequested;
            GameObject.Destroy(_mainView2);
            _mainView2 = null;
        }

        if (_libraryView != null)
        {
            _libraryView.OnRequestPageSwitch -= OnPageSwitchRequested;
            _libraryView.OnImageDoubleClicked -= OnLibraryImageDoubleClicked;
            GameObject.Destroy(_libraryView);
            _libraryView = null;
        }

        if (_designView != null)
        {
            _designView.OnRequestPageSwitch -= OnPageSwitchRequested;
            GameObject.Destroy(_designView);
            _designView = null;
        }
    }

    private void OnPageSwitchRequested(BasePageView.PageType targetPage)
    {
        // 验证切换是否允许
        if (!IsPageSwitchAllowed(_currentPage, targetPage))
        {
            Debug.LogWarning($"Page switch from {_currentPage} to {targetPage} is not allowed");
            return;
        }

        SwitchToPage(targetPage);
    }

    private bool IsPageSwitchAllowed(BasePageView.PageType from, BasePageView.PageType to)
    {
        // 根据需求定义的切换规则
        switch (from)
        {
            case BasePageView.PageType.MainView2:
                return to == BasePageView.PageType.LibraryView || to == BasePageView.PageType.DesignView;

            case BasePageView.PageType.LibraryView:
                return to == BasePageView.PageType.MainView2; // 只能切换到MainView2

            case BasePageView.PageType.DesignView:
                return to == BasePageView.PageType.MainView2; // 只能切换到MainView2

            default:
                return false;
        }
    }

    private void SwitchToPage(BasePageView.PageType targetPage)
    {
        // 隐藏当前页面
        HideCurrentPage();

        // 更新当前页面
        _currentPage = targetPage;

        // 显示目标页面
        ShowCurrentPage();

        Debug.Log($"Switched to page: {targetPage}");
    }

    private void HideCurrentPage()
    {
        switch (_currentPage)
        {
            case BasePageView.PageType.MainView2:
                _mainView2?.Hide();
                break;

            case BasePageView.PageType.LibraryView:
                _libraryView?.Hide();
                break;

            case BasePageView.PageType.DesignView:
                _designView?.Hide();
                break;
        }
    }

    private void ShowCurrentPage()
    {
        switch (_currentPage)
        {
            case BasePageView.PageType.MainView2:
                _mainView2?.Show();
                // 如果有共享的当前图片，设置到MainView2
                if (_sharedCurrentImage != null)
                {
                    _mainView2.SetCurrentImage(_sharedCurrentImage, _sharedCurrentImagePath);
                }
                break;

            case BasePageView.PageType.LibraryView:
                _libraryView?.Show();
                break;

            case BasePageView.PageType.DesignView:
                _designView?.Show();
                // 如果有共享的当前图片，设置到DesignView
                if (_sharedCurrentImage != null)
                {
                    _designView.SetCurrentImage(_sharedCurrentImage, _sharedCurrentImagePath);
                }
                break;
        }
    }

    private void OnLibraryImageDoubleClicked(string imagePath)
    {
        // LibraryView中双击缩略图，切换到MainView2并加载该图片
        Debug.Log($"Library image double-clicked: {imagePath}");

        // 加载图片
        var texture = LoadTextureFromFile(imagePath);
        if (texture != null)
        {
            // 释放旧图片
            if (_sharedCurrentImage != null)
                Destroy(_sharedCurrentImage);

            _sharedCurrentImage = texture;
            _sharedCurrentImagePath = imagePath;

            // 保存到PlayerPrefs
            PlayerPrefs.SetString(PrefKeyLastImagePath, imagePath);
            PlayerPrefs.Save();

            // 切换到MainView2
            SwitchToPage(BasePageView.PageType.MainView2);
        }
        else
        {
            Debug.LogError($"Failed to load image: {imagePath}");
        }
    }

    private void RestoreLastImage()
    {
        var lastImagePath = PlayerPrefs.GetString(PrefKeyLastImagePath, "");
        if (string.IsNullOrWhiteSpace(lastImagePath) || !File.Exists(lastImagePath))
        {
            Debug.Log("没有找到上次编辑的图片");
            return;
        }

        var texture = LoadTextureFromFile(lastImagePath);
        if (texture != null)
        {
            _sharedCurrentImage = texture;
            _sharedCurrentImagePath = lastImagePath;
            Debug.Log($"已加载上次编辑的图片: {lastImagePath}");
        }
        else
        {
            Debug.LogWarning($"无法加载上次的图片: {lastImagePath}");
        }
    }

    private static Texture2D LoadTextureFromFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return null;

        byte[] data;
        try
        {
            data = System.IO.File.ReadAllBytes(filePath);
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Failed to read image file: {ex.Message}");
            return null;
        }

        var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        if (!tex.LoadImage(data, false))
        {
            Destroy(tex);
            return null;
        }

        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        tex.name = System.IO.Path.GetFileName(filePath);
        return tex;
    }

    // 公共方法：外部可以调用来切换页面
    public void SwitchToMainView2()
    {
        if (_currentPage != BasePageView.PageType.MainView2)
            SwitchToPage(BasePageView.PageType.MainView2);
    }

    public void SwitchToLibraryView()
    {
        if (_currentPage != BasePageView.PageType.LibraryView)
            SwitchToPage(BasePageView.PageType.LibraryView);
    }

    public void SwitchToDesignView()
    {
        if (_currentPage != BasePageView.PageType.DesignView)
            SwitchToPage(BasePageView.PageType.DesignView);
    }

    // 设置当前图片（可以从外部调用，例如从原MainView传递图片）
    public void SetCurrentImage(Texture2D texture, string path)
    {
        _sharedCurrentImage = texture;
        _sharedCurrentImagePath = path;

        // 如果当前在MainView2或DesignView，立即更新
        if (_currentPage == BasePageView.PageType.MainView2 && _mainView2 != null)
        {
            _mainView2.SetCurrentImage(texture, path);
        }
        else if (_currentPage == BasePageView.PageType.DesignView && _designView != null)
        {
            _designView.SetCurrentImage(texture, path);
        }
    }
}
