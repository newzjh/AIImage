using System;
using System.Collections.Generic;
using System.IO;
using Aexis.Samples.Async;
using UnityEngine;
using UnityEngine.UIElements;

[DisallowMultipleComponent]
[RequireComponent(typeof(UIDocument))]
public sealed class AIImagePageHost : MonoBehaviour
{
    private const string PrefKeyLastImagePath = "MainView.LastImagePath";

    [SerializeField] private int textureCacheLimit = 18;

    public UIDocument Document => _uiDocument;
    public Image2ImageAI Image2ImageAI => _image2ImageAI;
    public CodeOnlyFileDialog FileDialog => _fileDialog;
    public AIImageModelDownloadDialog ModelDownloadDialog => _modelDownloadDialog;
    public ComputeShader ImageProcessingCS => _imageProcessingCS;
    public GpuSharpenRunner GpuSharpenRunner => _gpuSharpenRunner;
    public FaceMaskGenerator FaceMaskGenerator => _faceMaskGenerator;
    public RealEsrganNcnnReproRunner RealEsrganReproRunner => _realEsrganReproRunner;
    public GfpganNcnnReproRunner GfpganReproRunner => _gfpganReproRunner;
    public CodeFormerNcnnReproRunner2 CodeFormerReproRunner => _codeFormerReproRunner;
    public MatterNcnnReproRunner MattingReproRunner => _mattingReproRunner;
    public YoloSegNcnnReproRunner YoloSegRunner => _yoloSegRunner;
    public DeepFillV2Runner DeepFillV2Runner => _deepFillV2Runner;
    public SDInpaintingNcnnReproRunner SDInpaintingRunner => _sdInpaintingRunner;
    public ClipNcnnReproRunner ClipRunner => _clipNcnnReproRunner;
    public MainView2 MainPage => _mainView2;
    public LibraryView LibraryPage => _libraryView;
    public DesignView DesignPage => _designView;

    private UIDocument _uiDocument;
    private VisualElement _pageLayer;
    private BasePageView _activePage;
    private bool _transitioning;
    private System.Threading.CancellationTokenSource _startupNavigationCts;

    private Image2ImageAI _image2ImageAI;
    private CodeOnlyFileDialog _fileDialog;
    private AIImageModelDownloadDialog _modelDownloadDialog;
    private ComputeShader _imageProcessingCS;
    private GpuSharpenRunner _gpuSharpenRunner;
    private FaceMaskGenerator _faceMaskGenerator;
    private RealEsrganNcnnReproRunner _realEsrganReproRunner;
    private GfpganNcnnReproRunner _gfpganReproRunner;
    private CodeFormerNcnnReproRunner2 _codeFormerReproRunner;
    private MatterNcnnReproRunner _mattingReproRunner;
    private YoloSegNcnnReproRunner _yoloSegRunner;
    private DeepFillV2Runner _deepFillV2Runner;
    private SDInpaintingNcnnReproRunner _sdInpaintingRunner;
    private ClipNcnnReproRunner _clipNcnnReproRunner;

    private MainView2 _mainView2;
    private LibraryView _libraryView;
    private DesignView _designView;

    private readonly Dictionary<string, Texture2D> _textureCache = new Dictionary<string, Texture2D>(StringComparer.OrdinalIgnoreCase);
    private readonly Queue<string> _textureCacheOrder = new Queue<string>();

    private void Awake()
    {
        Aexis.Samples.AexisSampleStreamingAssets.RegisterManifestPathResolver();
        EnsureHostSetup();
    }

    private void OnEnable()
    {
        EnsureHostSetup();
        BuildRoot();
        InitializePages();
        if (_pageLayer == null)
            return;
        if (_mainView2 != null)
        {
            ShowImmediate(_mainView2);
            SwitchToLibraryWhenStartupImageIsMissingAsync().Forget();
        }
    }

    private void OnDisable()
    {
        CancelStartupNavigation();
        _activePage?.Detach();
        _activePage = null;
        if (_pageLayer != null)
            _pageLayer.Clear();
    }

    private void OnDestroy()
    {
        CancelStartupNavigation();
        foreach (var pair in _textureCache)
        {
            if (pair.Value != null)
                Destroy(pair.Value);
        }
        _textureCache.Clear();
        _textureCacheOrder.Clear();
    }

    public void RequestPageSwitch(BasePageView source, AppPageId target, SwipeDirection direction)
    {
        if (_transitioning || source == null || !ReferenceEquals(source, _activePage))
            return;

        var page = ResolvePage(target);
        if (page == null || ReferenceEquals(page, _activePage))
            return;

        PrepareIncomingPage(page);
        SwitchToAsync(page, direction).Forget();
    }

    public bool OpenLibraryImageInMain(string filePath)
    {
        var ok = ReloadMainImageFromDisk(filePath, false);
        if (!ok)
            return false;
        RequestPageSwitch(_libraryView, AppPageId.MainView2, SwipeDirection.Right);
        return true;
    }

    public void SetLanguage(AppLanguage language)
    {
        if (AppLocalization.CurrentLanguage == language)
            return;

        AppLocalization.SetLanguage(language);
        if (_activePage != null)
            ShowImmediate(_activePage);
    }

    public bool ReloadMainImageFromDisk(string filePath, bool bypassOriginalNameGuard = true)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return false;
        if (_mainView2 == null)
            return false;

        var ok = _mainView2.LoadImageFromPath(filePath, bypassOriginalNameGuard);
        if (!ok)
            return false;

        RememberLastImagePath(filePath);
        _libraryView?.SyncSelectionFromImagePath(filePath);
        SyncDesignFromMain();
        return true;
    }

    public bool TryOpenAdjacentMainImage(int direction)
    {
        if (_mainView2 == null)
            return false;

        var currentPath = _mainView2.CurrentSourcePathForSync;
        if (!ImageNavigationUtility.TryGetAdjacentImagePath(currentPath, direction, out var adjacentPath))
            return false;

        return ReloadMainImageFromDisk(adjacentPath, true);
    }

    public Texture2D LoadTexture(string filePath, bool forceReload)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return null;

        if (!forceReload)
        {
            if (_textureCache.TryGetValue(filePath, out var cached) && cached != null)
                return cached;
        }
        else
        {
            if (_textureCache.TryGetValue(filePath, out var oldCached) && oldCached != null)
                Destroy(oldCached);
            _textureCache.Remove(filePath);
        }

        if (!RawPhotoParser.TryLoadDisplayBytes(filePath, out var data))
            return null;

        var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        if (!tex.LoadImage(data, false))
        {
            Destroy(tex);
            return null;
        }

        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        tex.name = Path.GetFileName(filePath);
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

    public void InvalidateTextureCacheForPath(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return;
        if (_textureCache.TryGetValue(filePath, out var cached) && cached != null)
            Destroy(cached);
        _textureCache.Remove(filePath);
    }

    public string GetLastImagePath()
    {
        return PlayerPrefs.GetString(PrefKeyLastImagePath, string.Empty);
    }

    public void RememberLastImagePath(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return;

        PlayerPrefs.SetString(PrefKeyLastImagePath, filePath);
        PlayerPrefs.Save();
    }

    public UniTask<bool> EnsureModelGroupsAvailableAsync(
        string operationName,
        System.Threading.CancellationToken cancellationToken,
        params AIImageModelGroupId[] groupIds)
    {
        if (_modelDownloadDialog == null)
            return UniTask.FromResult(false);

        var groups = new List<AIImageModelGroup>();
        if (groupIds != null)
        {
            for (var index = 0; index < groupIds.Length; index++)
                groups.Add(AIImageModelDelivery.GetGroup(groupIds[index]));
        }

        return _modelDownloadDialog.EnsureAvailableAsync(operationName, groups, cancellationToken);
    }

    private void BuildRoot()
    {
        if (_uiDocument == null)
        {
            Debug.LogError("AIImagePageHost requires a UIDocument component.", this);
            return;
        }

        var root = _uiDocument.rootVisualElement;
        if (root == null)
        {
            Debug.LogError("AIImagePageHost could not access UIDocument.rootVisualElement.", this);
            return;
        }

        root.Clear();
        root.style.width = Length.Percent(100);
        root.style.height = Length.Percent(100);
        root.style.flexGrow = 1;
        root.style.position = Position.Relative;
        root.style.backgroundColor = new StyleColor(new Color(0.06f, 0.07f, 0.09f, 1f));

        _pageLayer = new VisualElement();
        _pageLayer.style.flexGrow = 1;
        _pageLayer.style.position = Position.Relative;
        root.Add(_pageLayer);
    }

    private void InitializePages()
    {
        EnsureHostSetup();
        if (_mainView2 == null || _libraryView == null || _designView == null)
        {
            Debug.LogError($"AIImagePageHost page bootstrap failed. MainView2={_mainView2 != null}, LibraryView={_libraryView != null}, DesignView={_designView != null}", this);
            return;
        }

        _mainView2.Initialize(this, _uiDocument);
        _libraryView.Initialize(this, _uiDocument);
        _designView.Initialize(this, _uiDocument);
    }

    private void ShowImmediate(BasePageView page)
    {
        if (_pageLayer == null || page == null)
            return;
        _activePage?.Detach();
        _pageLayer.Clear();
        page.AttachTo(_pageLayer);
        page.SetPageOffset(0f, 1f);
        _activePage = page;
        PrepareIncomingPage(page);
    }

    private async UniTaskVoid SwitchToLibraryWhenStartupImageIsMissingAsync()
    {
        CancelStartupNavigation();
        var cts = new System.Threading.CancellationTokenSource();
        _startupNavigationCts = cts;
        try
        {
            await UniTask.Delay(180, cancellationToken: cts.Token);
            if (cts.IsCancellationRequested ||
                !isActiveAndEnabled ||
                _transitioning ||
                !ReferenceEquals(_activePage, _mainView2) ||
                _mainView2 == null ||
                _mainView2.HasCurrentImage)
            {
                return;
            }

            _libraryView?.SelectStartupDefaultDirectory();
            RequestPageSwitch(_mainView2, AppPageId.LibraryView, SwipeDirection.Left);
        }
        catch (System.OperationCanceledException)
        {
        }
        finally
        {
            if (ReferenceEquals(_startupNavigationCts, cts))
            {
                _startupNavigationCts = null;
                cts.Dispose();
            }
        }
    }

    private void CancelStartupNavigation()
    {
        if (_startupNavigationCts == null)
            return;

        _startupNavigationCts.Cancel();
        _startupNavigationCts.Dispose();
        _startupNavigationCts = null;
    }

    private async UniTaskVoid SwitchToAsync(BasePageView incomingPage, SwipeDirection direction)
    {
        if (_pageLayer == null || _activePage == null)
            return;

        _transitioning = true;
        var outgoingPage = _activePage;
        incomingPage.AttachTo(_pageLayer);
        var width = Mathf.Max(1f, _pageLayer.resolvedStyle.width);
        var sign = direction == SwipeDirection.Left ? -1f : 1f;
        incomingPage.SetPageOffset(-sign * width, 0.96f);
        _activePage = incomingPage;

        var duration = 0.22f;
        var elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.unscaledDeltaTime;
            var t = Mathf.Clamp01(elapsed / duration);
            var eased = 1f - Mathf.Pow(1f - t, 3f);
            outgoingPage.SetPageOffset(sign * width * eased, 1f - 0.18f * eased);
            incomingPage.SetPageOffset((-sign * width) * (1f - eased), 0.96f + 0.04f * eased);
            await UniTask.Yield(PlayerLoopTiming.Update);
        }

        outgoingPage.Detach();
        incomingPage.SetPageOffset(0f, 1f);
        _transitioning = false;
    }

    private void PrepareIncomingPage(BasePageView page)
    {
        if (ReferenceEquals(page, _designView) && _mainView2 != null)
            SyncDesignFromMain();
    }

    private void SyncDesignFromMain()
    {
        if (_designView == null || _mainView2 == null)
            return;

        _designView.SyncFromMainView(
            _mainView2.CurrentSourcePathForSync,
            _mainView2.CurrentEditedTextureForSync,
            _mainView2.CurrentOriginalTextureForSync,
            _mainView2.CurrentDisplayLabelForSync);
    }

    private BasePageView ResolvePage(AppPageId id)
    {
        return id switch
        {
            AppPageId.MainView2 => _mainView2,
            AppPageId.LibraryView => _libraryView,
            AppPageId.DesignView => _designView,
            _ => null
        };
    }

    private void EnsurePageComponents()
    {
        _mainView2 = GetComponent<MainView2>();
        if (_mainView2 == null)
            _mainView2 = gameObject.AddComponent<MainView2>();

        _libraryView = GetComponent<LibraryView>();
        if (_libraryView == null)
            _libraryView = gameObject.AddComponent<LibraryView>();

        _designView = GetComponent<DesignView>();
        if (_designView == null)
            _designView = gameObject.AddComponent<DesignView>();
    }

    private void EnsureRuntimeComponents()
    {
        _image2ImageAI = GetOrAdd<Image2ImageAI>();
        _fileDialog = GetOrAdd<CodeOnlyFileDialog>();
        _modelDownloadDialog = GetOrAdd<AIImageModelDownloadDialog>();
        _modelDownloadDialog.Configure(_uiDocument);
        _gpuSharpenRunner = GetOrAdd<GpuSharpenRunner>();
        _faceMaskGenerator = GetOrAdd<FaceMaskGenerator>();
        _realEsrganReproRunner = GetOrAdd<RealEsrganNcnnReproRunner>();
        _gfpganReproRunner = GetOrAdd<GfpganNcnnReproRunner>();
        _codeFormerReproRunner = GetOrAdd<CodeFormerNcnnReproRunner2>();
        _mattingReproRunner = GetOrAdd<MatterNcnnReproRunner>();
        _yoloSegRunner = GetOrAdd<YoloSegNcnnReproRunner>();
        _deepFillV2Runner = GetOrAdd<DeepFillV2Runner>();
        _sdInpaintingRunner = GetOrAdd<SDInpaintingNcnnReproRunner>();
        _clipNcnnReproRunner = GetOrAdd<ClipNcnnReproRunner>();
    }

    private void EnsureHostSetup()
    {
        if (_uiDocument == null)
            _uiDocument = GetComponent<UIDocument>();

        EnsureRuntimeComponents();
        EnsurePageComponents();

        if (_imageProcessingCS == null)
            _imageProcessingCS = Resources.Load<ComputeShader>("ImageProcessing");
    }

    private T GetOrAdd<T>() where T : Component
    {
        var component = GetComponent<T>();
        if (component == null)
            component = gameObject.AddComponent<T>();
        return component;
    }
}
