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

[DisallowMultipleComponent]
[RequireComponent(typeof(UIDocument))]
public class MainView : MonoBehaviour
{
    private const string PrefKeyLastImagePath = "MainView.LastImagePath";
    private const string PrefKeyMaleFacePath = "MainView.Ref.MaleFacePath";
    private const string PrefKeyFemaleFacePath = "MainView.Ref.FemaleFacePath";
    private const string PrefKeyBackgroundPath = "MainView.Ref.BackgroundPath";
    private const string PrefKeyAIProvider = "MainView.AI.Provider";
    private const string PrefKeyGoogleApiKey = "MainView.AI.GoogleApiKey";
    private const string PrefKeyReplicateApiKey = "MainView.AI.ReplicateApiToken";
    private const string PrefKeyDashScopeApiKey = "MainView.AI.DashScopeApiKey";
    private const string PrefKeyDoubaoApiKey = "MainView.AI.DoubaoApiKey";
    private const string PrefKeyHuggingFaceToken = "MainView.AI.HuggingFaceToken";
    private const string PrefKeyRunwareApiKey = "MainView.AI.RunwareApiKey";
    private const string PrefKeyLumenfallApiKey = "MainView.AI.LumenfallApiKey";

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
    private bool _adjustRunning;
    private bool _previewRunning;
    private RenderTexture _previewRt;
    private Texture2D _previewSource;
    private string _previewKernelName;
    private Action<ComputeShader, float> _previewParamSetter;
    private float _previewValue;
    private int _previewPointerId = -1;
    private VisualElement _previewCaptureElement;
    private VisualElement _busyOverlay;
    private VisualElement _busyBarTrack;
    private VisualElement _busyBar;
    private Label _busyText;
    private IVisualElementScheduledItem _busyAnim;
    private float _busyPhase;
    private VisualElement _choiceOverlay;
    private UniTaskCompletionSource<int> _choiceTcs;
    private VisualElement _toastOverlay;
    private Label _toastText;
    private IVisualElementScheduledItem _toastHide;
    private System.Threading.CancellationTokenSource _clipboardCts;

    private Texture2D _maleFaceTexture;
    private Texture2D _femaleFaceTexture;
    private Texture2D _backgroundTexture;
    private Button _maleFaceButton;
    private Button _femaleFaceButton;
    private Button _backgroundButton;
    private DropdownField _providerDropdown;
    private TextField _apiKeyField;
    private string _maleFacePath;
    private string _femaleFacePath;
    private string _backgroundPath;
    private System.Threading.CancellationTokenSource _lifetimeCts;
    private CodeOnlyFileDialog _fileDialog;
    private bool _appendDeGlarePrompt;
    private bool _appendRemoveBgPeoplePrompt = true;
    private ComputeShader _imageProcessingCS;
    private readonly Dictionary<string, int> _imageProcessingKernelIds = new Dictionary<string, int>(StringComparer.Ordinal);
    private long _historyOpSeq;
    private bool _gpuSharpenDumpStages;
    private GpuSharpenRunner _gpuSharpenRunner;
    private FaceMaskGenerator _faceMaskGenerator;
    private System.Threading.CancellationTokenSource _faceMaskCts;
    private System.Threading.CancellationTokenSource _maleFaceMaskCts;
    private System.Threading.CancellationTokenSource _femaleFaceMaskCts;

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

    private static readonly string[] OriginalNameMarkersZh =
    {
        "原图", "原始", "原始图", "原片", "未编辑", "未处理", "直出", "原版"
    };

    private static readonly string[] OriginalNameMarkersEn =
    {
        "original", "originals", "orig", "unedited", "unprocessed", "raw", "source", "camera"
    };

    private void Awake()
    {
        _uiDocument = GetComponent<UIDocument>();
        _image2ImageAI = GetComponent<Image2ImageAI>();
        if (_image2ImageAI == null)
            _image2ImageAI = gameObject.AddComponent<Image2ImageAI>();

        _imageProcessingCS = Resources.Load<ComputeShader>("ImageProcessing");
        _gpuSharpenRunner = GetComponent<GpuSharpenRunner>();
        if (_gpuSharpenRunner == null)
            _gpuSharpenRunner = gameObject.AddComponent<GpuSharpenRunner>();

        _faceMaskGenerator = GetComponent<FaceMaskGenerator>();
        if (_faceMaskGenerator == null)
            _faceMaskGenerator = gameObject.AddComponent<FaceMaskGenerator>();

        _image2ImageAI.SelectResultIndex -= OnSelectAIResultIndex;
        _image2ImageAI.SelectResultIndex += OnSelectAIResultIndex;
        _image2ImageAI.RequestError -= OnAIRequestError;
        _image2ImageAI.RequestError += OnAIRequestError;

        _fileDialog = GetComponent<CodeOnlyFileDialog>();
        if (_fileDialog == null)
            _fileDialog = gameObject.AddComponent<CodeOnlyFileDialog>();
    }

    private void OnEnable()
    {
        if (!Application.isPlaying)
            return;

        if (_lifetimeCts != null)
        {
            _lifetimeCts.Cancel();
            _lifetimeCts.Dispose();
        }
        _lifetimeCts = new System.Threading.CancellationTokenSource();

        BuildUI();
        _fileDialog?.EnsureInitialized();
        RestoreReferencePickersFromPrefs();
        RestoreAISettingsFromPrefs();
        PopulateDrives();
        RestoreLastSelection();
    }

    private void OnDisable()
    {
        if (!Application.isPlaying)
            return;

        StopPreview();
        _imageViewer?.Clear();
        ClearHistory();
        HideBusy();
        HideChoice();
        if (_toastOverlay != null)
            _toastOverlay.style.display = DisplayStyle.None;
        if (_clipboardCts != null)
        {
            _clipboardCts.Cancel();
            _clipboardCts.Dispose();
            _clipboardCts = null;
        }
        if (_lifetimeCts != null)
        {
            _lifetimeCts.Cancel();
            _lifetimeCts.Dispose();
            _lifetimeCts = null;
        }

        if (_maleFaceTexture != null) Destroy(_maleFaceTexture);
        if (_femaleFaceTexture != null) Destroy(_femaleFaceTexture);
        if (_backgroundTexture != null) Destroy(_backgroundTexture);
        _maleFaceTexture = null;
        _femaleFaceTexture = null;
        _backgroundTexture = null;
        _maleFacePath = null;
        _femaleFacePath = null;
        _backgroundPath = null;

        CancelAndDisposeCts(ref _faceMaskCts);
        CancelAndDisposeCts(ref _maleFaceMaskCts);
        CancelAndDisposeCts(ref _femaleFaceMaskCts);
        _faceMaskGenerator?.ClearAllMasks();

        if (_image2ImageAI != null)
        {
            _image2ImageAI.SelectResultIndex -= OnSelectAIResultIndex;
            _image2ImageAI.RequestError -= OnAIRequestError;
        }

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

    private void OnAIRequestError(string message)
    {
        if (string.IsNullOrWhiteSpace(message))
            return;
        ShowToast(message, 3500);
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
        root.RegisterCallback<KeyDownEvent>(OnRootKeyDown, TrickleDown.TrickleDown);
        root.RegisterCallback<PointerUpEvent>(OnAnyPointerUp, TrickleDown.TrickleDown);
        root.RegisterCallback<PointerCancelEvent>(OnAnyPointerCancel, TrickleDown.TrickleDown);

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
        BuildToast(root);
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

    private void BuildToast(VisualElement root)
    {
        _toastOverlay = new VisualElement();
        _toastOverlay.style.position = Position.Absolute;
        _toastOverlay.style.left = 0;
        _toastOverlay.style.right = 0;
        _toastOverlay.style.top = 14;
        _toastOverlay.style.alignItems = Align.Center;
        _toastOverlay.style.justifyContent = Justify.FlexStart;
        _toastOverlay.style.display = DisplayStyle.None;
        _toastOverlay.pickingMode = PickingMode.Ignore;

        var bubble = new VisualElement();
        bubble.style.backgroundColor = new StyleColor(new Color(0.1f, 0.1f, 0.1f, 0.90f));
        bubble.style.borderTopLeftRadius = 10;
        bubble.style.borderTopRightRadius = 10;
        bubble.style.borderBottomLeftRadius = 10;
        bubble.style.borderBottomRightRadius = 10;
        bubble.style.paddingLeft = 14;
        bubble.style.paddingRight = 14;
        bubble.style.paddingTop = 8;
        bubble.style.paddingBottom = 8;
        bubble.style.borderLeftWidth = 1;
        bubble.style.borderRightWidth = 1;
        bubble.style.borderTopWidth = 1;
        bubble.style.borderBottomWidth = 1;
        bubble.style.borderLeftColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        bubble.style.borderRightColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        bubble.style.borderTopColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        bubble.style.borderBottomColor = new StyleColor(new Color(1f, 1f, 1f, 0.12f));
        _toastOverlay.Add(bubble);

        _toastText = new Label();
        _toastText.style.whiteSpace = WhiteSpace.NoWrap;
        _toastText.style.overflow = Overflow.Hidden;
        _toastText.style.textOverflow = TextOverflow.Ellipsis;
        _toastText.style.color = Color.white;
        bubble.Add(_toastText);

        root.Add(_toastOverlay);
    }

    private void ShowToast(string text, int milliseconds = 2000)
    {
        if (_toastOverlay == null) return;
        _toastText.text = text ?? "";
        _toastOverlay.style.display = DisplayStyle.Flex;
        _toastOverlay.BringToFront();

        if (_toastHide != null)
            _toastHide.Pause();

        _toastHide = _toastOverlay.schedule.Execute(() =>
        {
            if (_toastOverlay != null)
                _toastOverlay.style.display = DisplayStyle.None;
        }).StartingIn(Mathf.Max(200, milliseconds));
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
        parent.style.minHeight = 0;
        parent.style.position = Position.Relative;

        var headerContain = new VisualElement();
        headerContain.style.flexDirection = FlexDirection.Row;
        headerContain.style.alignItems = Align.Stretch;
        headerContain.style.paddingLeft = 0;
        headerContain.style.paddingRight = 0;
        headerContain.style.paddingTop = 0;
        headerContain.style.paddingBottom = 0;
        headerContain.style.flexShrink = 0;
        headerContain.style.backgroundColor = new StyleColor(new Color(0.15f, 0.15f, 0.15f, 1f));
        parent.Add(headerContain);

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Column;
        header.style.alignItems = Align.Stretch;
        header.style.paddingLeft = 8;
        header.style.paddingRight = 8;
        header.style.paddingTop = 6;
        header.style.paddingBottom = 6;
        header.style.flexShrink = 0;
        header.style.backgroundColor = new StyleColor(new Color(0.15f, 0.15f, 0.15f, 1f));
        headerContain.Add(header);

        var row0 = new VisualElement();
        row0.style.flexDirection = FlexDirection.Row;
        row0.style.alignItems = Align.Center;
        header.Add(row0);

        row0.Add(new Button(OnFaceSwap) { text = "换脸" });
        row0.Add(new Button(OnSharpen) { text = "清晰" });
        row0.Add(new Button(OnWhiten) { text = "美白" });
        row0.Add(new Button(OnSharpenWhiten) { text = "清晰&美白" });

        row0.Add(new Button(OnChangeBackground) { text = "换背景" });
        row0.Add(new Button(OnDehazeColorGrade) { text = "去霾&调色" });
        row0.Add(new Button(OnColorGrade) { text = "调色" });
        row0.Add(new Button(OnDehaze) { text = "去霾" });

        var deglareLabel = new Label("去反光");
        deglareLabel.style.color = Color.white;
        row0.Add(deglareLabel);

        var deglareToggle = new Toggle();
        deglareToggle.value = false;
        deglareToggle.style.color = Color.white;
        deglareToggle.RegisterValueChangedCallback(evt => _appendDeGlarePrompt = evt.newValue);
        row0.Add(deglareToggle);

        var removeBgPeopleLabel = new Label("去背景人物");
        removeBgPeopleLabel.style.color = Color.white;
        row0.Add(removeBgPeopleLabel);

        var removeBgPeopleToggle = new Toggle();
        removeBgPeopleToggle.value = true;
        removeBgPeopleToggle.style.color = Color.white;
        removeBgPeopleToggle.RegisterValueChangedCallback(evt => _appendRemoveBgPeoplePrompt = evt.newValue);
        row0.Add(removeBgPeopleToggle);

        var row1 = new VisualElement();
        row1.style.flexDirection = FlexDirection.Row;
        row1.style.flexWrap = Wrap.Wrap;
        row1.style.alignItems = Align.Center;
        row1.style.marginTop = 8;
        header.Add(row1);

        var fitButton = new Button(() => _imageViewer.FitToView()) { text = "Fit" };
        row1.Add(fitButton);

        var resetButton = new Button(() => _imageViewer.ResetView()) { text = "Reset" };
        row1.Add(resetButton);

        var saveButton = new Button(OnSaveCurrentImage) { text = "保存" };
        row1.Add(saveButton);

#if !UNITY_WEBGL
        var browseButton = new Button(OnBrowseOriginalImage) { text = "浏览" };
        row1.Add(browseButton);
#endif

        var gpuSharpenButton = new Button(OnGpuSharpen) { text = "GPU清晰化" };
        row1.Add(gpuSharpenButton);

        var gpuSharpenDebugLabel = new Label("调试输出");
        gpuSharpenDebugLabel.style.color = Color.white;
        row1.Add(gpuSharpenDebugLabel);

        var gpuSharpenDebugToggle = new Toggle();
        gpuSharpenDebugToggle.value = false;
        gpuSharpenDebugToggle.style.color = Color.white;
        gpuSharpenDebugToggle.RegisterValueChangedCallback(evt => _gpuSharpenDumpStages = evt.newValue);
        row1.Add(gpuSharpenDebugToggle);

        row1.Add(BuildProviderGroup(out _providerDropdown, out _apiKeyField));

        headerContain.Add(BuildReferencePickerGroup("男脸", "点击设置男人脸", OnPickMaleFace, out _maleFaceButton));
        headerContain.Add(BuildReferencePickerGroup("女脸", "点击设置女人脸", OnPickFemaleFace, out _femaleFaceButton));
        headerContain.Add(BuildReferencePickerGroup("背景", "点击设置背景图", OnPickBackground, out _backgroundButton));

        _imageViewer = new SplitCompareView();
        _imageViewer.style.flexGrow = 1;
        _imageViewer.style.minHeight = 0;
        parent.Add(_imageViewer);

        BuildFloatingAdjustPanel(parent);
    }

    private Button collapseBtn;
    private void BuildFloatingAdjustPanel(VisualElement parent)
    {
        var panel = new VisualElement();
        panel.style.position = Position.Absolute;
        panel.style.width = 360;
        panel.style.backgroundColor = new StyleColor(new Color(0.10f, 0.10f, 0.10f, 0.92f));
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
        panel.style.paddingLeft = 8;
        panel.style.paddingRight = 8;
        panel.style.paddingTop = 8;
        panel.style.paddingBottom = 8;

        var header = new VisualElement();
        header.style.flexDirection = FlexDirection.Row;
        header.style.alignItems = Align.Center;
        header.style.height = 30;
        header.style.marginBottom = 6;
        header.style.backgroundColor = new Color(0.2f,0.2f,0.2f);
        panel.Add(header);

        var title = new Label("图像调节");
        title.style.flexGrow = 1;
        title.style.color = Color.yellow;
        title.style.unityTextAlign = TextAnchor.MiddleLeft;
        header.Add(title);

        var body = new VisualElement();
        body.style.flexDirection = FlexDirection.Column;
        body.style.minHeight = 0;
        panel.Add(body);

        var collapsed = false;
        collapseBtn = new Button(() =>
        {
            collapsed = !collapsed;
            body.style.display = collapsed ? DisplayStyle.None : DisplayStyle.Flex;
            collapseBtn.text = collapsed ? "+" : "-";
        })
        { text = "-" };
        collapseBtn.style.height = 28;
        collapseBtn.style.fontSize = 28;
        header.Add(collapseBtn);

        bool dragging = false;
        int dragPointerId = -1;
        Vector2 startPointer = default;
        Vector2 startPos = default;

        header.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (evt.button != 0) return;
            dragging = true;
            dragPointerId = evt.pointerId;
            startPointer = evt.position;
            startPos = new Vector2(panel.layout.x, panel.layout.y);
            header.CapturePointer(dragPointerId);
            evt.StopPropagation();
        });
        header.RegisterCallback<PointerMoveEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId || !header.HasPointerCapture(dragPointerId))
                return;

            var delta = (Vector2)evt.position - startPointer;
            var newPos = startPos + delta;

            var bounds = parent.contentRect;
            var pw = panel.resolvedStyle.width;
            var ph = panel.resolvedStyle.height;
            var headerH = header.resolvedStyle.height;

            var maxX = Mathf.Max(0f, bounds.width - pw);
            var maxY = Mathf.Max(0f, bounds.height - headerH);

            newPos.x = Mathf.Clamp(newPos.x, 0f, maxX);
            newPos.y = Mathf.Clamp(newPos.y, 0f, maxY);

            panel.style.left = newPos.x;
            panel.style.top = newPos.y;
            evt.StopPropagation();
        });
        header.RegisterCallback<PointerUpEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId) return;
            dragging = false;
            if (header.HasPointerCapture(dragPointerId))
                header.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });
        header.RegisterCallback<PointerCancelEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId) return;
            dragging = false;
            if (header.HasPointerCapture(dragPointerId))
                header.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });

        body.Add(BuildAdjustRow("对比度", -0.5f, 0.5f, 0f, "AdjustContrast", (cs, v) => cs.SetFloat("_Contrast", v), v => $"对比度 {v:0.00}"));
        body.Add(BuildAdjustRow("亮度", -0.5f, 0.5f, 0f, "AdjustBrightness", (cs, v) => cs.SetFloat("_Brightness", v), v => $"亮度 {v:0.00}"));
        body.Add(BuildAdjustRow("自然饱和度", -1f, 1f, 0f, "AdjustVibrance", (cs, v) => cs.SetFloat("_Vibrance", v), v => $"自然饱和度 {v:0.00}"));
        body.Add(BuildAdjustRow("去阴影", 0f, 0.5f, 0f, "AdjustShadows", (cs, v) => cs.SetFloat("_Shadows", v), v => $"去阴影 {v:0.00}"));
        body.Add(BuildAdjustRow("去高光", 0f, 0.5f, 0f, "AdjustHighlights", (cs, v) => cs.SetFloat("_Highlights", v), v => $"去高光 {v:0.00}"));
        body.Add(BuildAdjustRow("加温滤镜", 0f, 1f, 0f, "WarmFilter", (cs, v) => cs.SetFloat("_Warm", v), v => $"加温 {v:0.00}"));
        body.Add(BuildAdjustRow("冷却滤镜", 0f, 1f, 0f, "CoolFilter", (cs, v) => cs.SetFloat("_Cool", v), v => $"冷却 {v:0.00}"));
        body.Add(BuildAdjustRow("锐化", 0f, 4f, 0f, "Sharpen", (cs, v) => cs.SetFloat("_Sharpen", v), v => $"锐化 {v:0.00}"));
        body.Add(BuildAdjustRow("模糊", 0f, 4f, 0f, "Blur", (cs, v) => cs.SetFloat("_Blur", v), v => $"模糊 {v:0.00}"));

        parent.Add(panel);
        panel.BringToFront();

        var placed = false;
        panel.RegisterCallback<GeometryChangedEvent>(_ =>
        {
            if (placed)
                return;
            if (parent.contentRect.width <= 1f || parent.contentRect.height <= 1f)
                return;

            var w = panel.resolvedStyle.width;
            var x = Mathf.Max(0f, parent.contentRect.width - w - 12f);
            panel.style.left = x;
            panel.style.top = 140f;
            placed = true;
        });
    }

    private VisualElement BuildAdjustRow(string name, float min, float max, float defaultValue, string kernelName, Action<ComputeShader, float> paramSetter, Func<float, string> historyLabel)
    {
        var row = new VisualElement();
        row.style.flexDirection = FlexDirection.Row;
        row.style.alignItems = Align.Center;
        row.style.marginBottom = 6;

        var label = new Label(name);
        label.style.width = 100;
        label.style.color = Color.white;
        row.Add(label);

        var slider = new Slider(min, max);
        slider.value = defaultValue;
        slider.style.flexGrow = 1;
        slider.style.marginRight = 8;
        slider.RegisterValueChangedCallback(evt =>
        {
            if (_previewRunning && ReferenceEquals(_previewCaptureElement, slider))
                _previewValue = evt.newValue;
        });
        slider.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (evt.button != 0) return;
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
        row.Add(slider);

        var btn = new Button(() =>
        {
            StopPreview();
            var v = slider.value;
            ApplyComputeAdjustmentAsync(kernelName, cs => paramSetter?.Invoke(cs, v), historyLabel != null ? historyLabel(v) : name).Forget();
        }) { text = "应用" };
        btn.style.height = 28;
        row.Add(btn);

        return row;
    }

    private void Update()
    {
        if (!Application.isPlaying)
            return;
        HandleGlobalShortcuts();

        if (!_previewRunning)
            return;
        if (!Input.GetMouseButton(0))
        {
            StopPreview();
            return;
        }
        if (_previewRt == null || _previewSource == null)
        {
            StopPreview();
            return;
        }
        if (_aiRunning || _adjustRunning)
            return;

        var cs = GetImageProcessingCS();
        if (cs == null)
            return;
        var kernel = GetKernelId(_previewKernelName);
        if (kernel < 0)
            return;

        cs.SetTexture(kernel, "_Source", _previewSource);
        cs.SetTexture(kernel, "_Result", _previewRt);
        _previewParamSetter?.Invoke(cs, _previewValue);

        var gx = Mathf.CeilToInt(_previewSource.width / 8f);
        var gy = Mathf.CeilToInt(_previewSource.height / 8f);
        cs.Dispatch(kernel, gx, gy, 1);

        _imageViewer?.MarkDirtyRepaint();
    }

    private void HandleGlobalShortcuts()
    {
        if (_uiDocument == null)
            return;
        var root = _uiDocument.rootVisualElement;
        var focused = root?.focusController?.focusedElement;
        if (focused is TextField)
            return;

        var ctrl = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);
        var cmd = Input.GetKey(KeyCode.LeftCommand) || Input.GetKey(KeyCode.RightCommand);
        var ctrlOrCmd = ctrl || cmd;
        var shift = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);

        if (ctrlOrCmd && !shift && Input.GetKeyDown(KeyCode.Z))
            UndoLastOperation();

        if (Input.GetKeyDown(KeyCode.Delete))
            DeleteSelectedHistoryEntry();
    }

    private void DeleteSelectedHistoryEntry()
    {
        if (_historyList == null)
            return;
        var index = _historyList.selectedIndex;
        if (index <= 0 || index >= _historyEntries.Count)
            return;

        var removed = _historyEntries[index];
        _historyEntries.RemoveAt(index);
        if (removed.owned && removed.texture != null)
            Destroy(removed.texture);

        var newIndex = Mathf.Clamp(index, 0, _historyEntries.Count - 1);
        if (_historyList != null)
        {
            _historyList.RefreshItems();
            _historyList.SetSelection(newIndex);
            _historyList.ScrollToItem(newIndex);
        }

        var entry = _historyEntries[newIndex];
        var original = GetOriginalHistoryTexture();
        _imageViewer?.SetSources(entry.texture, original, entry.label);
    }

    private void StartPreview(VisualElement captureElement, int pointerId, string kernelName, Action<ComputeShader, float> paramSetter, float initialValue)
    {
        if (_aiRunning || _adjustRunning)
            return;

        var src = GetCurrentHistoryTexture();
        if (src == null) src = GetOriginalHistoryTexture();
        if (src == null)
            return;

        var cs = GetImageProcessingCS();
        if (cs == null)
            return;
        var kernel = GetKernelId(kernelName);
        if (kernel < 0)
            return;

        StopPreview();
        _previewRunning = true;
        _previewKernelName = kernelName;
        _previewParamSetter = paramSetter;
        _previewValue = initialValue;
        _previewSource = src;
        _previewPointerId = pointerId;
        _previewCaptureElement = captureElement;

        _previewRt = new RenderTexture(src.width, src.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        _previewRt.enableRandomWrite = true;
        _previewRt.Create();

        if (_previewCaptureElement != null)
            _previewCaptureElement.CapturePointer(_previewPointerId);

        _imageViewer?.SetPreview(_previewRt);
        _imageViewer?.MarkDirtyRepaint();
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

        _imageViewer?.SetPreview(null);
        _imageViewer?.MarkDirtyRepaint();

        if (_previewRt != null)
        {
            _previewRt.Release();
            Destroy(_previewRt);
            _previewRt = null;
        }
    }

    private void OnAnyPointerUp(PointerUpEvent evt)
    {
        if (!_previewRunning)
            return;
        if (evt.pointerId != _previewPointerId)
            return;
        StopPreview();
    }

    private void OnAnyPointerCancel(PointerCancelEvent evt)
    {
        if (!_previewRunning)
            return;
        if (evt.pointerId != _previewPointerId)
            return;
        StopPreview();
    }

    private void OnRootKeyDown(KeyDownEvent evt)
    {
        if (evt == null)
            return;

        if (evt.target is TextField)
            return;

        var ctrlOrCmd = evt.ctrlKey || evt.commandKey;
        if (!ctrlOrCmd)
            return;

        if (evt.keyCode == KeyCode.Z && !evt.shiftKey)
        {
            UndoLastOperation();
            evt.StopPropagation();
        }
    }

    private void UndoLastOperation()
    {
        if (_historyEntries.Count <= 1)
        {
            ShowToast("没有可撤销的历史记录", 1800);
            return;
        }

        var bestSeq = -1L;
        var bestIndex = -1;
        for (int i = 1; i < _historyEntries.Count; i++)
        {
            var s = _historyEntries[i].opSeq;
            if (s > bestSeq)
            {
                bestSeq = s;
                bestIndex = i;
            }
        }

        if (bestIndex < 0 || bestSeq <= 0)
        {
            ShowToast("没有可撤销的历史记录", 1800);
            return;
        }

        var removed = _historyEntries[bestIndex];
        _historyEntries.RemoveAt(bestIndex);
        if (removed.owned && removed.texture != null)
            Destroy(removed.texture);

        var selectSeq = -1L;
        var selectIndex = 0;
        for (int i = 1; i < _historyEntries.Count; i++)
        {
            var s = _historyEntries[i].opSeq;
            if (s > selectSeq)
            {
                selectSeq = s;
                selectIndex = i;
            }
        }

        _historyList?.RefreshItems();
        if (_historyList != null)
        {
            _historyList.SetSelection(selectIndex);
            _historyList.ScrollToItem(selectIndex);
        }

        var current = _historyEntries[selectIndex].texture;
        var original = GetOriginalHistoryTexture();
        _imageViewer?.SetSources(current, original, _historyEntries[selectIndex].label);
    }

    private ComputeShader GetImageProcessingCS()
    {
        if (_imageProcessingCS == null)
            _imageProcessingCS = Resources.Load<ComputeShader>("ImageProcessing");
        return _imageProcessingCS;
    }

    private int GetKernelId(string kernelName)
    {
        if (string.IsNullOrWhiteSpace(kernelName))
            return -1;
        if (_imageProcessingKernelIds.TryGetValue(kernelName, out var id))
            return id;
        var cs = GetImageProcessingCS();
        if (cs == null)
            return -1;
        try
        {
            id = cs.FindKernel(kernelName);
        }
        catch
        {
            id = -1;
        }
        _imageProcessingKernelIds[kernelName] = id;
        return id;
    }

    private void OnGpuSharpen()
    {
        ApplyGpuSharpenAsync().Forget();
    }

    private async UniTaskVoid ApplyGpuSharpenAsync()
    {
        if (_aiRunning) return;
        if (_adjustRunning) return;
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested) return;

        StopPreview();

        var src = GetCurrentHistoryTexture();
        if (src == null) src = GetOriginalHistoryTexture();
        if (src == null) return;

        _adjustRunning = true;
        ShowBusy("GPU清晰化");

        try
        {
            if (_gpuSharpenRunner == null)
            {
                ShowToast("找不到GpuSharpenRunner", 2200);
                return;
            }

            Texture2D faceMask = _faceMaskGenerator != null ? _faceMaskGenerator.currentImageFaceMask : null;
            if ((faceMask == null || faceMask.width != src.width || faceMask.height != src.height) && _faceMaskGenerator != null)
            {
                CancelAndDisposeCts(ref _faceMaskCts);
                _faceMaskCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
                var gen = await _faceMaskGenerator.GenerateForCurrentAsync(src, false, _faceMaskCts.Token);
                faceMask = string.IsNullOrWhiteSpace(gen.error) ? gen.mask : null;
            }

            var r = await _gpuSharpenRunner.ProcessAsync(src, faceMask, _gpuSharpenDumpStages, _lifetimeCts.Token);
            if (!string.IsNullOrWhiteSpace(r.error))
            {
                ShowToast(r.error, 3500);
                return;
            }
            if (r.texture != null)
                AddHistory(r.texture, "GPU清晰化");
        }
        finally
        {
            _adjustRunning = false;
            HideBusy();
        }
    }

    private async UniTaskVoid ApplyComputeAdjustmentAsync(string kernelName, Action<ComputeShader> setParams, string historyLabel)
    {
        if (_aiRunning) return;
        if (_adjustRunning) return;
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested) return;

        var src = GetCurrentHistoryTexture();
        if (src == null) src = GetOriginalHistoryTexture();
        if (src == null) return;

        var cs = GetImageProcessingCS();
        if (cs == null)
        {
            ShowToast("找不到ImageProcessing.compute", 2200);
            return;
        }

        var kernel = GetKernelId(kernelName);
        if (kernel < 0)
        {
            ShowToast("Compute kernel无效: " + kernelName, 2200);
            return;
        }

        _adjustRunning = true;
        ShowBusy(historyLabel);
        RenderTexture rt = null;
        try
        {
            rt = new RenderTexture(src.width, src.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            rt.enableRandomWrite = true;
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
            _adjustRunning = false;
            HideBusy();
        }
    }

    private async UniTask<Texture2D> ReadbackTextureAsync(RenderTexture rt, int w, int h)
    {
        var tcs = new UniTaskCompletionSource<AsyncGPUReadbackRequest>();
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, req => tcs.TrySetResult(req));
        var r = await tcs.Task;
        if (r.hasError)
            return null;

        var data = r.GetData<byte>();
        var tex = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
        tex.LoadRawTextureData(data);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
    }

    private static GroupBox BuildReferencePickerGroup(string labelText, string buttonText, Action onClick, out Button button)
    {
        var group = new GroupBox();
        group.style.flexDirection = FlexDirection.Row;
        group.style.alignItems = Align.Center;
        group.style.marginRight = 2;
        group.style.paddingLeft = 2;
        group.style.paddingRight = 2;
        group.style.paddingTop = 2;
        group.style.paddingBottom = 2;

        //var label = new Label(labelText);
        //label.style.marginRight = 6;
        //group.Add(label);

        button = new Button(onClick) { text = buttonText };
        button.style.width = 128;
        button.style.height = 128;
        button.style.unityTextAlign = TextAnchor.MiddleCenter;
        button.style.backgroundColor = new StyleColor(new Color(0.18f, 0.18f, 0.18f, 1f));
        button.style.color = Color.white;
        button.style.borderTopLeftRadius = 6;
        button.style.borderTopRightRadius = 6;
        button.style.borderBottomLeftRadius = 6;
        button.style.borderBottomRightRadius = 6;
        group.Add(button);

        return group;
    }

    private static GroupBox BuildProviderGroup(out DropdownField providerDropdown, out TextField apiKeyField)
    {
        var group = new GroupBox();
        group.style.flexDirection = FlexDirection.Row;
        group.style.alignItems = Align.Center;
        group.style.marginRight = 10;
        group.style.paddingLeft = 8;
        group.style.paddingRight = 8;
        group.style.paddingTop = 6;
        group.style.paddingBottom = 6;

        var label = new Label("模型");
        label.style.marginRight = 4;
        group.Add(label);

        providerDropdown = new DropdownField();
        providerDropdown.style.width = 160;
        providerDropdown.style.height = 36;
        providerDropdown.style.marginRight = 4;
        group.Add(providerDropdown);

        var apiLabel = new Label("API Key");
        apiLabel.style.marginRight = 4;
        group.Add(apiLabel);

        apiKeyField = new TextField();
        apiKeyField.style.width = 360;
        apiKeyField.style.height = 36;
        group.Add(apiKeyField);

        return group;
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
                .Where(p => !IsOriginalDefinedPath(p))
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

        var childDirs = EnumerateDirectoriesSafe(parentPath, maxChildrenPerDirectory)
            .Where(p => !IsOriginalDefinedPath(p))
            .ToList();
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

    private void RestoreReferencePickersFromPrefs()
    {
        TryRestoreReference(PrefKeyMaleFacePath, _maleFaceButton, "点击设置男人脸", tex => _maleFaceTexture = tex, () => _maleFaceTexture, p => _maleFacePath = p);
        TryRestoreReference(PrefKeyFemaleFacePath, _femaleFaceButton, "点击设置女人脸", tex => _femaleFaceTexture = tex, () => _femaleFaceTexture, p => _femaleFacePath = p);
        TryRestoreReference(PrefKeyBackgroundPath, _backgroundButton, "点击设置背景图", tex => _backgroundTexture = tex, () => _backgroundTexture, p => _backgroundPath = p);

        if (_maleFaceTexture != null)
            PrepareFaceMaskForReferenceAsync(_maleFaceTexture, true).Forget();
        if (_femaleFaceTexture != null)
            PrepareFaceMaskForReferenceAsync(_femaleFaceTexture, false).Forget();
    }

    private void TryRestoreReference(
        string prefKey,
        Button button,
        string defaultText,
        Action<Texture2D> setTex,
        Func<Texture2D> getTex,
        Action<string> setPath)
    {
        var path = PlayerPrefs.GetString(prefKey, "");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            var old0 = getTex();
            if (old0 != null) Destroy(old0);
            setTex(null);
            setPath(null);
            if (button != null)
            {
                button.style.backgroundImage = StyleKeyword.None;
                button.text = defaultText ?? "";
            }
            return;
        }

        var tex = LoadTextureFromFile(path);
        if (tex == null)
            return;

        var old = getTex();
        if (old != null) Destroy(old);
        setTex(tex);
        setPath(path);
        if (button != null)
        {
            button.style.backgroundImage = new StyleBackground(tex);
            button.style.unityBackgroundScaleMode = ScaleMode.ScaleToFit;
            button.text = "";
        }
    }

    private void RestoreAISettingsFromPrefs()
    {
        if (_image2ImageAI == null)
            return;

        if (PlayerPrefs.HasKey(PrefKeyAIProvider))
        {
            var p = (Image2ImageAI.Provider)PlayerPrefs.GetInt(PrefKeyAIProvider, (int)_image2ImageAI.CurrentProvider);
            _image2ImageAI.CurrentProvider = p;
        }

        var google = PlayerPrefs.GetString(PrefKeyGoogleApiKey, null);
        if (!string.IsNullOrEmpty(google))
            _image2ImageAI.SetApiKeyForProvider(Image2ImageAI.Provider.GoogleAIStudio, google);

        var repl = PlayerPrefs.GetString(PrefKeyReplicateApiKey, null);
        if (!string.IsNullOrEmpty(repl))
            _image2ImageAI.SetApiKeyForProvider(Image2ImageAI.Provider.Replicate, repl);

        var dash = PlayerPrefs.GetString(PrefKeyDashScopeApiKey, null);
        if (!string.IsNullOrEmpty(dash))
            _image2ImageAI.SetApiKeyForProvider(Image2ImageAI.Provider.AliTongyiWanxiang, dash);

        var dou = PlayerPrefs.GetString(PrefKeyDoubaoApiKey, null);
        if (!string.IsNullOrEmpty(dou))
            _image2ImageAI.SetApiKeyForProvider(Image2ImageAI.Provider.Doubao, dou);
 
        var hf = PlayerPrefs.GetString(PrefKeyHuggingFaceToken, null);
        if (!string.IsNullOrEmpty(hf))
            _image2ImageAI.SetApiKeyForProvider(Image2ImageAI.Provider.HuggingFaceInferenceProviders, hf);
 
        var runware = PlayerPrefs.GetString(PrefKeyRunwareApiKey, null);
        if (!string.IsNullOrEmpty(runware))
            _image2ImageAI.SetApiKeyForProvider(Image2ImageAI.Provider.RunwareAI, runware);
 
        var lumenfall = PlayerPrefs.GetString(PrefKeyLumenfallApiKey, null);
        if (!string.IsNullOrEmpty(lumenfall))
            _image2ImageAI.SetApiKeyForProvider(Image2ImageAI.Provider.Lumenfall, lumenfall);

        if (_providerDropdown != null)
        {
            _providerDropdown.choices = Enum.GetNames(typeof(Image2ImageAI.Provider)).ToList();
            var providerName = _image2ImageAI.CurrentProvider.ToString();
            if (_providerDropdown.choices.Contains(providerName))
                _providerDropdown.SetValueWithoutNotify(providerName);

            _providerDropdown.RegisterValueChangedCallback(evt =>
            {
                if (_image2ImageAI == null) return;
                if (!Enum.TryParse(evt.newValue, out Image2ImageAI.Provider newP))
                    return;
                _image2ImageAI.CurrentProvider = newP;
                PlayerPrefs.SetInt(PrefKeyAIProvider, (int)newP);
                PlayerPrefs.Save();
                UpdateApiKeyFieldFromAI();
            });
        }

        if (_apiKeyField != null)
        {
            UpdateApiKeyFieldFromAI();
            _apiKeyField.RegisterValueChangedCallback(evt =>
            {
                if (_image2ImageAI == null) return;
                var p = _image2ImageAI.CurrentProvider;
                _image2ImageAI.SetApiKeyForProvider(p, evt.newValue ?? "");
                var key = GetApiKeyPrefKey(p);
                if (!string.IsNullOrWhiteSpace(key))
                {
                    PlayerPrefs.SetString(key, evt.newValue ?? "");
                    PlayerPrefs.Save();
                }
            });
        }
    }

    private void UpdateApiKeyFieldFromAI()
    {
        if (_apiKeyField == null || _image2ImageAI == null)
            return;
        var key = _image2ImageAI.GetApiKeyForProvider(_image2ImageAI.CurrentProvider) ?? "";
        _apiKeyField.SetValueWithoutNotify(key);
    }

    private static string GetApiKeyPrefKey(Image2ImageAI.Provider p)
    {
        return p switch
        {
            Image2ImageAI.Provider.GoogleAIStudio => PrefKeyGoogleApiKey,
            Image2ImageAI.Provider.Replicate => PrefKeyReplicateApiKey,
            Image2ImageAI.Provider.AliTongyiWanxiang => PrefKeyDashScopeApiKey,
            Image2ImageAI.Provider.Doubao => PrefKeyDoubaoApiKey,
            Image2ImageAI.Provider.HuggingFaceInferenceProviders => PrefKeyHuggingFaceToken,
            Image2ImageAI.Provider.RunwareAI => PrefKeyRunwareApiKey,
            Image2ImageAI.Provider.Lumenfall => PrefKeyLumenfallApiKey,
            _ => null
        };
    }

    private static bool IsOriginalDefinedPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return false;

        var normalized = path.Replace('\\', '/');
        var parts = normalized.Split(new[] { '/' }, StringSplitOptions.RemoveEmptyEntries);
        for (int i = 0; i < parts.Length; i++)
        {
            if (IsOriginalDefinedName(parts[i]))
                return true;
        }
        return false;
    }

    private static bool IsOriginalDefinedName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return false;

        for (int i = 0; i < OriginalNameMarkersZh.Length; i++)
        {
            if (name.IndexOf(OriginalNameMarkersZh[i], StringComparison.OrdinalIgnoreCase) >= 0)
                return true;
        }

        var lower = name.ToLowerInvariant();
        for (int i = 0; i < OriginalNameMarkersEn.Length; i++)
        {
            var marker = OriginalNameMarkersEn[i];
            if (marker.Length <= 4)
            {
                if (ContainsEnglishMarkerWithBoundary(lower, marker))
                    return true;
            }
            else if (lower.Contains(marker))
                return true;
        }

        return false;
    }

    private static bool ContainsEnglishMarkerWithBoundary(string lower, string markerLower)
    {
        var idx = lower.IndexOf(markerLower, StringComparison.Ordinal);
        while (idx >= 0)
        {
            var beforeOk = idx == 0 || !char.IsLetterOrDigit(lower[idx - 1]);
            var end = idx + markerLower.Length;
            var afterOk = end >= lower.Length || !char.IsLetterOrDigit(lower[end]);
            if (beforeOk && afterOk)
                return true;
            idx = lower.IndexOf(markerLower, idx + 1, StringComparison.Ordinal);
        }
        return false;
    }

#if !UNITY_WEBGL
    private void OnBrowseOriginalImage()
    {
        var path = _historyEntries.Count > 0 ? _historyEntries[0].sourcePath : null;
        if (string.IsNullOrWhiteSpace(path))
        {
            ShowToast("没有原路径", 2000);
            return;
        }
        if (!File.Exists(path))
        {
            ShowToast("原文件不存在", 2000);
            return;
        }

        try
        {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            Process.Start(new ProcessStartInfo("explorer.exe", "/open,\"" + path + "\"") { UseShellExecute = true });
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
            Process.Start(new ProcessStartInfo("open", "-R \"" + path + "\"") { UseShellExecute = false });
#elif UNITY_EDITOR
            UnityEditor.EditorUtility.RevealInFinder(path);
#elif UNITY_STANDALONE_LINUX
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(dir))
                Process.Start(new ProcessStartInfo("xdg-open", "\"" + dir + "\"") { UseShellExecute = false });
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
        }
    }
#endif

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

        if (IsOriginalDefinedName(entry.fileName) || IsOriginalDefinedPath(entry.fullPath))
        {
            ShowToast("文件名包含原图字样，如需编辑请改名", 2200);
            _imageList?.ClearSelection();
            return;
        }

        var tex = LoadTexture(entry.fullPath, true);
        if (tex != null)
        {
            ResetHistoryWithOriginal(tex, entry.fileName, entry.fullPath);
            CopySelectionToClipboardAsync(entry.fullPath, tex).Forget();
            PrepareFaceMaskForSelectedImageAsync(entry.fullPath, tex, _gpuSharpenDumpStages).Forget();

            PlayerPrefs.SetString(PrefKeyLastImagePath, entry.fullPath);
            PlayerPrefs.Save();
        }
    }

    private async UniTaskVoid CopySelectionToClipboardAsync(string imageFilePath, Texture2D texture)
    {
        if (_clipboardCts != null)
        {
            _clipboardCts.Cancel();
            _clipboardCts.Dispose();
        }
        _clipboardCts = new System.Threading.CancellationTokenSource();
        var localCts = _clipboardCts;

        await UniTask.NextFrame();

        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
            return;
        if (!ReferenceEquals(_clipboardCts, localCts) || localCts.IsCancellationRequested)
            return;

        if (string.IsNullOrWhiteSpace(imageFilePath))
            return;
        if (texture == null)
            return;

        try
        {
            CopySelectionToClipboard(imageFilePath, texture);
            ShowToast("图片已复制到粘贴板", 2000);
        }
        catch
        {
        }
    }

    private void OnSaveCurrentImage()
    {
        SaveCurrentImageAsync().Forget();
    }

    private async UniTaskVoid SaveCurrentImageAsync()
    {
        await UniTask.NextFrame();

        if (_historyEntries.Count <= 1)
        {
            ShowToast("没有修改", 2000);
            return;
        }

        var path = _historyEntries.Count > 0 ? _historyEntries[0].sourcePath : null;
        if (string.IsNullOrWhiteSpace(path))
            return;

        var tex = GetCurrentHistoryTexture();
        if (tex == null)
            return;

        byte[] bytes = null;
        var ext = (Path.GetExtension(path) ?? "").ToLowerInvariant();
        try
        {
            if (ext == ".png")
                bytes = tex.EncodeToPNG();
            else if (ext == ".jpg" || ext == ".jpeg")
                bytes = tex.EncodeToJPG(95);
            else if (ext == ".tga")
                bytes = tex.EncodeToTGA();
            else if (ext == ".exr")
                bytes = tex.EncodeToEXR(Texture2D.EXRFlags.CompressZIP);
            else
            {
                ShowToast("不支持保存格式: " + ext, 2000);
                return;
            }
        }
        catch
        {
            return;
        }

        if (bytes == null || bytes.Length == 0)
            return;

        try
        {
            await File.WriteAllBytesAsync(path, bytes);
            InvalidateTextureCacheForPath(path);
            ShowToast("已保存到原路径", 2000);
        }
        catch
        {
        }
    }

    private void OnPickMaleFace() => PickReferenceImageAsync("选择男人脸图片", _maleFaceButton, tex => _maleFaceTexture = tex, () => _maleFaceTexture, "点击设置男人脸").Forget();
    private void OnPickFemaleFace() => PickReferenceImageAsync("选择女人脸图片", _femaleFaceButton, tex => _femaleFaceTexture = tex, () => _femaleFaceTexture, "点击设置女人脸").Forget();
    private void OnPickBackground() => PickReferenceImageAsync("选择背景图片", _backgroundButton, tex => _backgroundTexture = tex, () => _backgroundTexture, "点击设置背景图").Forget();

    private async UniTaskVoid PickReferenceImageAsync(string title, Button button, Action<Texture2D> setTex, Func<Texture2D> getTex, string defaultText)
    {
        await UniTask.NextFrame();
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
            return;
        if (_fileDialog == null)
            return;

        _fileDialog.EnsureInitialized();
        var path = await _fileDialog.ShowOpenImageAsync();
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
            return;

        if (string.IsNullOrWhiteSpace(path))
        {
            var old0 = getTex();
            if (old0 != null)
                Destroy(old0);
            setTex(null);

            if (ReferenceEquals(button, _maleFaceButton)) { _maleFacePath = null; PlayerPrefs.DeleteKey(PrefKeyMaleFacePath); }
            if (ReferenceEquals(button, _femaleFaceButton)) { _femaleFacePath = null; PlayerPrefs.DeleteKey(PrefKeyFemaleFacePath); }
            if (ReferenceEquals(button, _backgroundButton)) { _backgroundPath = null; PlayerPrefs.DeleteKey(PrefKeyBackgroundPath); }
            PlayerPrefs.Save();

            if (ReferenceEquals(button, _maleFaceButton))
            {
                CancelAndDisposeCts(ref _maleFaceMaskCts);
                _faceMaskGenerator?.ClearMaleMask();
            }
            if (ReferenceEquals(button, _femaleFaceButton))
            {
                CancelAndDisposeCts(ref _femaleFaceMaskCts);
                _faceMaskGenerator?.ClearFemaleMask();
            }

            if (button != null)
            {
                button.style.backgroundImage = StyleKeyword.None;
                button.text = defaultText ?? "";
            }
            return;
        }

        var tex = LoadTextureFromFile(path);
        if (tex == null)
            return;

        var old = getTex();
        if (old != null)
            Destroy(old);

        setTex(tex);
        if (ReferenceEquals(button, _maleFaceButton)) { _maleFacePath = path; PlayerPrefs.SetString(PrefKeyMaleFacePath, path); }
        if (ReferenceEquals(button, _femaleFaceButton)) { _femaleFacePath = path; PlayerPrefs.SetString(PrefKeyFemaleFacePath, path); }
        if (ReferenceEquals(button, _backgroundButton)) { _backgroundPath = path; PlayerPrefs.SetString(PrefKeyBackgroundPath, path); }
        PlayerPrefs.Save();

        if (button != null)
        {
            button.style.backgroundImage = new StyleBackground(tex);
            button.style.unityBackgroundScaleMode = ScaleMode.ScaleToFit;
            button.text = "";
        }

        if (ReferenceEquals(button, _maleFaceButton))
            PrepareFaceMaskForReferenceAsync(tex, true).Forget();
        if (ReferenceEquals(button, _femaleFaceButton))
            PrepareFaceMaskForReferenceAsync(tex, false).Forget();
    }

    private static void CancelAndDisposeCts(ref System.Threading.CancellationTokenSource cts)
    {
        if (cts == null)
            return;
        try { cts.Cancel(); } catch { }
        try { cts.Dispose(); } catch { }
        cts = null;
    }

    private async UniTaskVoid PrepareFaceMaskForSelectedImageAsync(string imagePath, Texture2D src, bool dumpDebug)
    {
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
            return;
        if (_faceMaskGenerator == null)
            return;
        if (string.IsNullOrWhiteSpace(imagePath) || src == null)
            return;

        CancelAndDisposeCts(ref _faceMaskCts);
        _faceMaskCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
        var localCts = _faceMaskCts;

        var r = await _faceMaskGenerator.GenerateForCurrentAsync(src, dumpDebug, localCts.Token);
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
        {
            return;
        }
        if (!ReferenceEquals(_faceMaskCts, localCts) || localCts.IsCancellationRequested)
        {
            return;
        }
        if (!string.IsNullOrWhiteSpace(r.error) || r.mask == null)
            return;
    }

    private async UniTaskVoid PrepareFaceMaskForReferenceAsync(Texture2D src, bool isMale)
    {
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested)
            return;
        if (_faceMaskGenerator == null || src == null)
            return;

        if (isMale)
        {
            CancelAndDisposeCts(ref _maleFaceMaskCts);
            _maleFaceMaskCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
            var localCts = _maleFaceMaskCts;
            var r = await _faceMaskGenerator.GenerateForMaleAsync(src, false, localCts.Token);
            if (!ReferenceEquals(_maleFaceMaskCts, localCts) || localCts.IsCancellationRequested)
            {
                return;
            }
            if (!string.IsNullOrWhiteSpace(r.error) || r.mask == null)
                return;
        }
        else
        {
            CancelAndDisposeCts(ref _femaleFaceMaskCts);
            _femaleFaceMaskCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(_lifetimeCts.Token);
            var localCts = _femaleFaceMaskCts;
            var r = await _faceMaskGenerator.GenerateForFemaleAsync(src, false, localCts.Token);
            if (!ReferenceEquals(_femaleFaceMaskCts, localCts) || localCts.IsCancellationRequested)
            {
                return;
            }
            if (!string.IsNullOrWhiteSpace(r.error) || r.mask == null)
                return;
        }
    }

    private static Texture2D LoadTextureFromFile(string filePath)
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
            Destroy(tex);
            return null;
        }
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        tex.name = Path.GetFileName(filePath);
        return tex;
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
        _historyOpSeq = 0;

        _historyEntries.Add(new HistoryEntry
        {
            label = "原图: " + (label ?? originalTexture.name),
            texture = originalTexture,
            owned = false,
            sourcePath = fullPath,
            opSeq = 0
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
            sourcePath = null,
            opSeq = ++_historyOpSeq
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
    private void OnSharpenWhiten() => ApplyOperation(ImageOp.SharpenWhiten);
    private void OnChangeBackground() => ApplyOperation(ImageOp.ChangeBackground);
    private void OnDehazeColorGrade() => ApplyOperation(ImageOp.DehazeColorGrade);
    private void OnColorGrade() => ApplyOperation(ImageOp.ColorGrade);
    private void OnDehaze() => ApplyOperation(ImageOp.Dehaze);

    private async UniTaskVoid RunAIForOperation(ImageOp op)
    {
        if (_aiRunning) return;
        if (_image2ImageAI == null) return;
        if (_lifetimeCts == null || _lifetimeCts.IsCancellationRequested) return;

        var src = GetCurrentHistoryTexture();
        if (src == null) src = GetOriginalHistoryTexture();
        var original = GetOriginalHistoryTexture();
        if (src == null || original == null) return;

        _aiRunning = true;
        ShowBusy(OpLabel(op) + "处理中…");
        try
        {
            var useChinesePrompt = _image2ImageAI.CurrentProvider == Image2ImageAI.Provider.Doubao ||
                                   _image2ImageAI.CurrentProvider == Image2ImageAI.Provider.AliTongyiWanxiang;
            var prompt = BuildPromptForOp(op, useChinesePrompt);

            if ((op == ImageOp.Sharpen || op == ImageOp.SharpenWhiten) && _image2ImageAI.CurrentProvider != Image2ImageAI.Provider.Doubao)
                prompt += useChinesePrompt
                    ? "调整构图，聚焦前景人物，适当放大前景人物，让前景人物更突出，"
                    : "Adjust the framing to focus on the main subject. Slightly zoom in so the subject stands out. ";

            var refs = new List<Texture2D> { src };
            if (op == ImageOp.FaceSwap)
            {
                var hasMale = _maleFaceTexture != null;
                var hasFemale = _femaleFaceTexture != null;

                if (hasMale && hasFemale)
                {
                    var same = ReferenceEquals(_maleFaceTexture, _femaleFaceTexture) ||
                               (!string.IsNullOrWhiteSpace(_maleFacePath) &&
                                string.Equals(_maleFacePath, _femaleFacePath, StringComparison.OrdinalIgnoreCase));

                    if (same)
                    {
                        prompt = useChinesePrompt
                            ? "以图1的构图和人物为基础，用图2的男角色人脸替换掉图1中男角色的人脸，用图2的女角色人脸替换掉图1中女角色的人脸，保留图1的构图与内容，只改变对应人脸；保持光照、肤色与细节一致，结果真实自然无明显伪影。"
                            : "Replace the male face in image 1 with the face from image 2, and replace the female face in image 1 with the face from image 2. Keep image 1 as the base: preserve composition and content, only change the specified faces. Match lighting, skin tone and fine details. Natural and artifact-free.";
                        refs.Add(_maleFaceTexture);
                    }
                    else
                    {
                        prompt = useChinesePrompt
                            ? "以图1的构图和人物为基础，用图2的男角色人脸替换掉图1中男角色的人脸，用图3的女角色人脸替换掉图1中女角色的人脸，保留图1的构图与内容，只改变对应人脸；保持光照、肤色与细节一致，结果真实自然无明显伪影。"
                            : "Replace the male face in image 1 with the face from image 2, and replace the female face in image 1 with the face from image 3. Keep image 1 as the base: preserve composition and content, only change the specified faces. Match lighting, skin tone and fine details. Natural and artifact-free.";
                        refs.Add(_maleFaceTexture);
                        refs.Add(_femaleFaceTexture);
                    }
                }
                else if (hasMale)
                {
                    prompt = useChinesePrompt
                        ? "以图1的构图和人物为基础，用图2的男角色人脸替换掉图1中男角色的人脸，保留图1的构图与内容，只改变该人脸；保持光照、肤色与细节一致，结果真实自然无明显伪影。"
                        : "Replace the male face in image 1 with the face from image 2. Keep image 1 as the base: preserve composition and content, only change that face. Match lighting, skin tone and fine details. Natural and artifact-free.";
                    refs.Add(_maleFaceTexture);
                }
                else if (hasFemale)
                {
                    prompt = useChinesePrompt
                        ? "以图1的构图和人物为基础，用图2的女角色人脸替换掉图1中女角色的人脸，保留图1的构图与内容，只改变该人脸；保持光照、肤色与细节一致，结果真实自然无明显伪影。"
                        : "Replace the female face in image 1 with the face from image 2. Keep image 1 as the base: preserve composition and content, only change that face. Match lighting, skin tone and fine details. Natural and artifact-free.";
                    refs.Add(_femaleFaceTexture);
                }
                else
                {
                    if (original != null && !ReferenceEquals(original, src))
                        refs.Add(original);
                }
            }
            else if (op == ImageOp.ChangeBackground)
            {
                if (_backgroundTexture != null)
                    refs.Add(_backgroundTexture);
            }

            if (ShouldAppendPromptToggles(op))
                prompt = AppendPromptToggles(prompt, useChinesePrompt);

            var result = await _image2ImageAI.ImageToImageAsync(
                refs,
                prompt,
                original.width,
                original.height,
                _lifetimeCts.Token);

            if (result != null)
                AddHistory(result, OpLabel(op));
        }
        finally
        {
            _aiRunning = false;
            HideBusy();
        }
    }

    private static bool ShouldAppendPromptToggles(ImageOp op)
    {
        return op == ImageOp.Sharpen ||
               op == ImageOp.Whiten ||
               op == ImageOp.SharpenWhiten ||
               op == ImageOp.DehazeColorGrade ||
               op == ImageOp.ColorGrade ||
               op == ImageOp.Dehaze;
    }

    private string AppendPromptToggles(string prompt, bool useChinesePrompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
            prompt = "";

        if (_appendDeGlarePrompt)
            prompt += (prompt.Length > 0 ? " " : "") + (useChinesePrompt
                ? "并进行去反光/去高光处理，降低镜面反射与眩光，保留细节与真实质感，"
                : "Reduce glare and specular highlights. Minimize reflections and flare while preserving fine details and realistic texture. ");
        if (_appendRemoveBgPeoplePrompt)
            prompt += (prompt.Length > 0 ? " " : "") + (useChinesePrompt
                ? "并移除画面中的背景人物，自动补全背景，纹理连贯自然，无明显修补痕迹，"
                : "Remove background people and inpaint the background seamlessly. Keep textures coherent and avoid obvious retouching artifacts. ");

        return prompt;
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

    private static string BuildPromptForOp(ImageOp op, bool useChinesePrompt)
    {
        if (useChinesePrompt)
        {
            return op switch
            {
                ImageOp.FaceSwap => "将输入图片中的前景人物脸部进行自然的换脸处理，保持光照、肤色和细节一致，结果真实且无明显伪影,",
                ImageOp.Sharpen => "对输入图片严格保持前景人物脸容发型和五官不变，提高前景人物清晰度,",
                ImageOp.Whiten => "对输入图片严格保持前景人物脸容发型和五官不变，对前景人物进行轻微美白与肤色优化，保持肤质真实，避免假白和过度磨皮,",
                ImageOp.SharpenWhiten => "对输入图片在严格保持前景人物脸容发型和五官不变，提高前景人物清晰度，并进行轻微美白与肤色优化，保持肤质真实，避免假白和过度磨皮,",
                ImageOp.ChangeBackground => "在保持主体完整的前提下替换背景，边缘自然干净，主体与背景融合自然,",
                ImageOp.DehazeColorGrade => "对输入图片进行去霾与对比度提升，增强通透感，保留细节避免色偏，并进行调色优化，提升整体观感与色彩层次，保持自然不过饱和,",
                ImageOp.ColorGrade => "对输入图片进行调色，提升整体观感与色彩层次，保持自然不过饱和,",
                ImageOp.Dehaze => "对输入图片进行去霾与对比度提升，增强通透感，保留细节避免色偏,",
                _ => op.ToString()
            };
        }

        return op switch
        {
            ImageOp.FaceSwap => "Perform a natural face replacement on the foreground person. Preserve lighting, skin tone and fine details. Realistic result with minimal artifacts.",
            ImageOp.Sharpen => "Keep the subject's face, hairstyle and facial features strictly unchanged. Increase clarity and sharpness on the subject while keeping everything else consistent.",
            ImageOp.Whiten => "Keep the subject's face, hairstyle and facial features strictly unchanged. Apply subtle whitening and skin tone enhancement. Keep skin texture realistic; avoid over-smoothing and unnatural whitening.",
            ImageOp.SharpenWhiten => "Keep the subject's face, hairstyle and facial features strictly unchanged. Increase clarity/sharpness and apply subtle whitening and skin tone enhancement. Keep skin texture realistic; avoid over-smoothing and unnatural whitening.",
            ImageOp.ChangeBackground => "Replace the background while keeping the subject intact. Clean natural edges and seamless blending between subject and background.",
            ImageOp.DehazeColorGrade => "Remove haze and boost contrast to improve clarity while preserving details and avoiding color cast. Then apply natural color grading to enhance overall look and color depth without over-saturation.",
            ImageOp.ColorGrade => "Apply natural color grading to enhance overall look and color depth without over-saturation.",
            ImageOp.Dehaze => "Remove haze and boost contrast to improve clarity while preserving details and avoiding color cast.",
            _ => op.ToString()
        };
    }

    private static string OpLabel(ImageOp op)
    {
        return op switch
        {
            ImageOp.FaceSwap => "换脸",
            ImageOp.Sharpen => "清晰",
            ImageOp.Whiten => "美白",
            ImageOp.SharpenWhiten => "清晰&美白",
            ImageOp.ChangeBackground => "换背景",
            ImageOp.DehazeColorGrade => "去霾&调色",
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

    private void InvalidateTextureCacheForPath(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return;
        if (_textureCache.TryGetValue(filePath, out var cached) && cached != null)
            Destroy(cached);
        _textureCache.Remove(filePath);
    }

    private Texture2D LoadTexture(string filePath, bool forceReload)
    {
        if (string.IsNullOrWhiteSpace(filePath)) return null;

        if (!forceReload)
        {
            if (_textureCache.TryGetValue(filePath, out var cached0) && cached0 != null)
                return cached0;
        }
        else
        {
            if (_textureCache.TryGetValue(filePath, out var cached0) && cached0 != null)
                Destroy(cached0);
            _textureCache.Remove(filePath);
        }

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
        public long opSeq;
    }

    private enum ImageOp
    {
        FaceSwap,
        Sharpen,
        Whiten,
        SharpenWhiten,
        ChangeBackground,
        DehazeColorGrade,
        ColorGrade,
        Dehaze
    }

    private sealed class SplitCompareView : VisualElement
    {
        private readonly Label _info;

        private Texture _texA;
        private Texture _texB;
        private Texture _previewTex;
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

        public void SetPreview(Texture preview)
        {
            _previewTex = preview;
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
                var viewRect = GetViewRect();
                var viewportSize = viewRect.size;
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

            var viewRect = GetViewRect();
            var viewportSize = viewRect.size;
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

            var refTex0 = _texA != null ? _texA : _texB;
            var viewRect0 = GetViewRect();
            if (viewRect0.width <= 1f || viewRect0.height <= 1f)
                return;
            if (!viewRect0.Contains(evt.localMousePosition))
                return;

            var imgRect0 = GetImageRect(refTex0, _zoom, viewRect0.position + _pan);
            var drawRect0 = IntersectRect(viewRect0, imgRect0);
            if (!drawRect0.Contains(evt.localMousePosition))
                return;

            var viewportPos = evt.localMousePosition - viewRect0.position;
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
            var viewRect = GetViewRect();
            if (viewRect.width <= 1f || viewRect.height <= 1f)
                return;
            if (!viewRect.Contains(evt.localPosition))
                return;

            var imgRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
            if (imgRect.width <= 1f || imgRect.height <= 1f)
                return;
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
                var viewRect = GetViewRect();
                var imgRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
                var w = Mathf.Max(1f, imgRect.width);
                var h = Mathf.Max(1f, imgRect.height);
                var deltaLocal = evt.localPosition - _splitDragStartLocal;
                var deltaUv = new Vector2(deltaLocal.x / w, -deltaLocal.y / h);
                var drawRect = IntersectRect(viewRect, imgRect);

                if (evt.shiftKey)
                {
                    var deltaAngle = (deltaLocal.x / w) * Mathf.PI * 2f;
                    angleRad = _splitDragStartAngle + deltaAngle;
                    ClampOffsetToDrawRect(drawRect, imgRect);
                }
                else
                {
                    var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
                    offset = _splitDragStartOffset - Vector2.Dot(n, deltaUv);
                    ClampOffsetToDrawRect(drawRect, imgRect);
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

        private Rect GetViewRect()
        {
            var bounds = contentRect;
            if (bounds.width <= 1f || bounds.height <= 1f)
                return new Rect(0, 0, 0, 0);

            var infoBottom = _info.layout.y + _info.layout.height;
            infoBottom = Mathf.Clamp(infoBottom, bounds.yMin, bounds.yMax);
            return Rect.MinMaxRect(bounds.xMin, infoBottom, bounds.xMax, bounds.yMax);
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

        private void ClampOffsetToDrawRect(Rect drawRect, Rect imageRect)
        {
            if (drawRect.width <= 1f || drawRect.height <= 1f)
                return;
            if (imageRect.width <= 1f || imageRect.height <= 1f)
                return;

            var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
            if (n.sqrMagnitude <= 1e-6f)
                return;
            n.Normalize();

            Vector2 UvOf(Vector2 pLocal)
            {
                return new Vector2(
                    (pLocal.x - imageRect.xMin) / imageRect.width,
                    1f - ((pLocal.y - imageRect.yMin) / imageRect.height)
                );
            }

            var p0 = new Vector2(drawRect.xMin, drawRect.yMin);
            var p1 = new Vector2(drawRect.xMax, drawRect.yMin);
            var p2 = new Vector2(drawRect.xMax, drawRect.yMax);
            var p3 = new Vector2(drawRect.xMin, drawRect.yMax);

            var c = new Vector2(0.5f, 0.5f);
            float D(Vector2 uv) => Vector2.Dot(n, uv - c);

            var d0 = D(UvOf(p0));
            var d1 = D(UvOf(p1));
            var d2 = D(UvOf(p2));
            var d3 = D(UvOf(p3));

            var minD = Mathf.Min(Mathf.Min(d0, d1), Mathf.Min(d2, d3));
            var maxD = Mathf.Max(Mathf.Max(d0, d1), Mathf.Max(d2, d3));

            var minOffset = -maxD;
            var maxOffset = -minD;
            if (minOffset > maxOffset)
            {
                var t = minOffset;
                minOffset = maxOffset;
                maxOffset = t;
            }

            offset = Mathf.Clamp(offset, minOffset, maxOffset);
        }

        private void OnGenerateVisualContent(MeshGenerationContext mgc)
        {
            var refTex = _texA != null ? _texA : _texB;
            if (refTex == null)
                return;

            var viewRect = GetViewRect();
            if (viewRect.width <= 1f || viewRect.height <= 1f)
                return;

            var imageRect = GetImageRect(refTex, _zoom, viewRect.position + _pan);
            var drawRect = IntersectRect(viewRect, imageRect);
            if (drawRect.width <= 1f || drawRect.height <= 1f)
                return;

            ClampOffsetToDrawRect(drawRect, imageRect);

            if (_previewTex != null)
            {
                DrawFullRect(mgc, _previewTex, drawRect, imageRect);
                return;
            }
            float SignedDist(Vector2 p) => SignedDistUv(p, imageRect);

            if (_texA != null)
                DrawHalfPlane(mgc, _texA, drawRect, imageRect, SignedDist, keepNegative: true);
            if (_texB != null)
                DrawHalfPlane(mgc, _texB, drawRect, imageRect, SignedDist, keepNegative: false);

            DrawSplitLine(mgc, drawRect, imageRect);
        }

        private static void DrawFullRect(MeshGenerationContext mgc, Texture tex, Rect drawRect, Rect imageRect)
        {
            var mesh = mgc.Allocate(4, 6, tex);
            var p0 = new Vector2(drawRect.xMin, drawRect.yMin);
            var p1 = new Vector2(drawRect.xMax, drawRect.yMin);
            var p2 = new Vector2(drawRect.xMax, drawRect.yMax);
            var p3 = new Vector2(drawRect.xMin, drawRect.yMax);

            Vector2 Uv(Vector2 p)
            {
                return new Vector2(
                    (p.x - imageRect.xMin) / imageRect.width,
                    1f - ((p.y - imageRect.yMin) / imageRect.height)
                );
            }

            mesh.SetNextVertex(new Vertex { position = p0, uv = Uv(p0), tint = Color.white });
            mesh.SetNextVertex(new Vertex { position = p1, uv = Uv(p1), tint = Color.white });
            mesh.SetNextVertex(new Vertex { position = p2, uv = Uv(p2), tint = Color.white });
            mesh.SetNextVertex(new Vertex { position = p3, uv = Uv(p3), tint = Color.white });

            mesh.SetNextIndex(0);
            mesh.SetNextIndex(1);
            mesh.SetNextIndex(2);
            mesh.SetNextIndex(0);
            mesh.SetNextIndex(2);
            mesh.SetNextIndex(3);
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
            if (imageRect.width <= 1f || imageRect.height <= 1f)
                return;

            var n = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
            var w = imageRect.width;
            var h = imageRect.height;
            var x0 = imageRect.xMin;
            var y0 = imageRect.yMin;
            var y1 = imageRect.yMax;
            var c = 0.5f * (n.x + n.y) - offset;
            var A = n.x / w;
            var B = -n.y / h;
            var C = (-n.x * x0 / w) + (n.y * y1 / h) - c;

            float F(Vector2 p) => A * p.x + B * p.y + C;

            var corners = new[]
            {
                new Vector2(drawRect.xMin, drawRect.yMin),
                new Vector2(drawRect.xMax, drawRect.yMin),
                new Vector2(drawRect.xMax, drawRect.yMax),
                new Vector2(drawRect.xMin, drawRect.yMax)
            };

            var points = new List<Vector2>(4);
            for (int i = 0; i < 4; i++)
            {
                var p0 = corners[i];
                var p1 = corners[(i + 1) % 4];
                var f0 = F(p0);
                var f1 = F(p1);

                if (Mathf.Abs(f0) <= 1e-6f)
                    points.Add(p0);

                if ((f0 > 0f && f1 < 0f) || (f0 < 0f && f1 > 0f))
                {
                    var t = f0 / (f0 - f1);
                    points.Add(p0 + (p1 - p0) * t);
                }
            }

            if (points.Count < 2)
                return;

            var segA = points[0];
            var segB = points[1];
            float bestDist = (segB - segA).sqrMagnitude;
            for (int i = 0; i < points.Count; i++)
            {
                for (int j = i + 1; j < points.Count; j++)
                {
                    var d = (points[j] - points[i]).sqrMagnitude;
                    if (d > bestDist)
                    {
                        bestDist = d;
                        segA = points[i];
                        segB = points[j];
                    }
                }
            }

            var dir = segB - segA;
            if (dir.sqrMagnitude < 1e-4f)
                return;
            dir.Normalize();
            var perp = new Vector2(-dir.y, dir.x) * (thicknessPx * 0.5f);

            var v0p = segA + perp;
            var v1p = segA - perp;
            var v2p = segB - perp;
            var v3p = segB + perp;

            var quad = new List<Vector2> { v0p, v1p, v2p, v3p };
            var clipped = ClipToRect(quad, drawRect);
            if (clipped.Count < 3)
                return;

            var vCount = clipped.Count;
            var iCount = (vCount - 2) * 3;
            var mesh = mgc.Allocate(vCount, iCount, Texture2D.whiteTexture);
            for (int i = 0; i < vCount; i++)
                mesh.SetNextVertex(new Vertex { position = clipped[i], uv = Vector2.zero, tint = lineColor });
            for (int i = 0; i < vCount - 2; i++)
            {
                mesh.SetNextIndex(0);
                mesh.SetNextIndex((ushort)(i + 1));
                mesh.SetNextIndex((ushort)(i + 2));
            }
        }

        private static List<Vector2> ClipToRect(List<Vector2> poly, Rect rect)
        {
            List<Vector2> p = poly;
            p = ClipPolygon(p, v => rect.xMin - v.x, true);
            if (p.Count < 3) return p;
            p = ClipPolygon(p, v => v.x - rect.xMax, true);
            if (p.Count < 3) return p;
            p = ClipPolygon(p, v => rect.yMin - v.y, true);
            if (p.Count < 3) return p;
            p = ClipPolygon(p, v => v.y - rect.yMax, true);
            return p;
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

