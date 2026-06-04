using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

/// <summary>
/// 所有页面的基类 - 改为MonoBehaviour
/// </summary>
public abstract class BasePageView : MonoBehaviour
{
    protected UIDocument _uiDocument;
    protected VisualElement _root;
    protected VisualElement _pageContainer;
    protected VisualElement _pageIndicator;

    protected System.Threading.CancellationTokenSource _lifetimeCts;

    public enum PageType
    {
        MainView2,
        LibraryView,
        DesignView
    }

    public event Action<PageType> OnRequestPageSwitch;

    protected virtual void Awake()
    {
        _uiDocument = GetComponent<UIDocument>();
        if (_uiDocument == null)
        {
            Debug.LogError($"[{GetType().Name}] UIDocument component not found!");
        }

    }

    protected virtual void OnEnable()
    {
        if (!Application.isPlaying)
            return;

        _lifetimeCts = new System.Threading.CancellationTokenSource();
        _root = _uiDocument?.rootVisualElement;
    }

    protected virtual void OnDisable()
    {
        if (!Application.isPlaying)
            return;

        if (_lifetimeCts != null)
        {
            _lifetimeCts.Cancel();
            _lifetimeCts.Dispose();
            _lifetimeCts = null;
        }
    }

    public virtual void Show()
    {
        if (_pageContainer != null)
        {
            if (!_root.Contains(_pageContainer))
            {
                _root.Add(_pageContainer);
            }
            _pageContainer.style.display = DisplayStyle.Flex;
            _pageContainer.BringToFront();
        }


    }

    public virtual void Hide()
    {
        if (_pageContainer != null)
        {
            _pageContainer.style.display = DisplayStyle.None;
            // 注意：不要Remove，只是隐藏，避免重建开销
        }

    }

    public abstract void BuildPage();

    protected void BuildPageIndicator(PageType currentPage)
    {
        _pageIndicator = new VisualElement();
        _pageIndicator.name = "page-indicator";
        _pageIndicator.style.position = Position.Absolute;
        _pageIndicator.style.bottom = 20; // 提高位置，避免被遮挡
        _pageIndicator.style.left = 0;
        _pageIndicator.style.right = 0;
        _pageIndicator.style.height = 60;
        _pageIndicator.style.alignItems = Align.Center;
        _pageIndicator.style.flexDirection = FlexDirection.Row;
        _pageIndicator.style.justifyContent = Justify.Center;
        _pageIndicator.pickingMode = PickingMode.Ignore; // 不阻挡其他元素

        var touchArea = new VisualElement();
        touchArea.style.width = 200;
        touchArea.style.height = 60;
        touchArea.style.alignItems = Align.Center;
        touchArea.style.justifyContent = Justify.Center;

        var indicatorBar = new VisualElement();
        indicatorBar.style.width = 120;
        indicatorBar.style.height = 6;
        indicatorBar.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.3f));
        indicatorBar.style.borderTopLeftRadius = 3;
        indicatorBar.style.borderTopRightRadius = 3;
        indicatorBar.style.borderBottomLeftRadius = 3;
        indicatorBar.style.borderBottomRightRadius = 3;
        indicatorBar.style.overflow = Overflow.Hidden;

        var activeIndicator = new VisualElement();
        activeIndicator.name = "active-indicator";
        activeIndicator.style.width = 40;
        activeIndicator.style.height = Length.Percent(100);
        activeIndicator.style.backgroundColor = new StyleColor(new Color(1f, 1f, 1f, 0.9f));
        activeIndicator.style.position = Position.Absolute;
        activeIndicator.style.left = currentPage == PageType.MainView2 ? 40 : (currentPage == PageType.LibraryView ? 0 : 80);

        // CSS动画
        var propertyNames = new List<StylePropertyName> { new StylePropertyName("left") };
        activeIndicator.style.transitionProperty = new StyleList<StylePropertyName>(propertyNames);

        var durations = new List<TimeValue> { new TimeValue(0.3f, TimeUnit.Second) };
        activeIndicator.style.transitionDuration = new StyleList<TimeValue>(durations);

        var timingFunctions = new List<EasingFunction> { new EasingFunction(EasingMode.EaseInOut) };
        activeIndicator.style.transitionTimingFunction = new StyleList<EasingFunction>(timingFunctions);

        indicatorBar.Add(activeIndicator);
        touchArea.Add(indicatorBar);
        _pageIndicator.Add(touchArea);

        // 触摸拖动逻辑
        SetupIndicatorDragging(touchArea, indicatorBar, currentPage);

        _pageContainer?.Add(_pageIndicator);
        _pageIndicator.BringToFront(); // 确保在最上层
    }

    private void SetupIndicatorDragging(VisualElement touchArea, VisualElement indicatorBar, PageType startPage)
    {
        bool dragging = false;
        int dragPointerId = -1;
        float startX = 0;
        float currentDragX = 0;

        touchArea.RegisterCallback<PointerDownEvent>(evt =>
        {
            if (evt.button != 0) return;
            dragging = true;
            dragPointerId = evt.pointerId;
            startX = evt.position.x;
            currentDragX = 0;
            touchArea.CapturePointer(dragPointerId);
            evt.StopPropagation();
        });

        touchArea.RegisterCallback<PointerMoveEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId) return;

            currentDragX = evt.position.x - startX;
            var indicator = indicatorBar.Q<VisualElement>("active-indicator");
            if (indicator != null)
            {
                var basePos = startPage == PageType.MainView2 ? 40f : (startPage == PageType.LibraryView ? 0f : 80f);
                var newPos = Mathf.Clamp(basePos - currentDragX * 0.2f, 0f, 80f);
                indicator.style.left = newPos;
            }
            evt.StopPropagation();
        });

        touchArea.RegisterCallback<PointerUpEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId) return;

            float deltaX = currentDragX;
            bool shouldSwitch = Mathf.Abs(deltaX) > 20;

            if (shouldSwitch)
            {
                if (deltaX > 0)
                {
                    var targetPage = GetRightPage(startPage);
                    if (targetPage != startPage)
                        RequestPageSwitch(targetPage);
                }
                else
                {
                    var targetPage = GetLeftPage(startPage);
                    if (targetPage != startPage)
                        RequestPageSwitch(targetPage);
                }
            }
            else
            {
                var indicator = indicatorBar.Q<VisualElement>("active-indicator");
                if (indicator != null)
                {
                    var originalPos = startPage == PageType.MainView2 ? 40f : (startPage == PageType.LibraryView ? 0f : 80f);
                    indicator.style.left = originalPos;
                }
            }

            dragging = false;
            if (touchArea.HasPointerCapture(dragPointerId))
                touchArea.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });

        touchArea.RegisterCallback<PointerCancelEvent>(evt =>
        {
            if (!dragging || dragPointerId != evt.pointerId) return;

            var indicator = indicatorBar.Q<VisualElement>("active-indicator");
            if (indicator != null)
            {
                var originalPos = startPage == PageType.MainView2 ? 40f : (startPage == PageType.LibraryView ? 0f : 80f);
                indicator.style.left = originalPos;
            }

            dragging = false;
            if (touchArea.HasPointerCapture(dragPointerId))
                touchArea.ReleasePointer(dragPointerId);
            evt.StopPropagation();
        });
    }

    protected PageType GetLeftPage(PageType current)
    {
        return current switch
        {
            PageType.MainView2 => PageType.DesignView,
            PageType.LibraryView => PageType.LibraryView,
            PageType.DesignView => PageType.MainView2,
            _ => current
        };
    }

    protected PageType GetRightPage(PageType current)
    {
        return current switch
        {
            PageType.MainView2 => PageType.LibraryView,
            PageType.LibraryView => PageType.MainView2,
            PageType.DesignView => PageType.DesignView,
            _ => current
        };
    }

    protected void RequestPageSwitch(PageType targetPage)
    {
        OnRequestPageSwitch?.Invoke(targetPage);
    }

    protected VisualElement BuildToast()
    {
        var toastOverlay = new VisualElement();
        toastOverlay.name = "toast-overlay";
        toastOverlay.style.position = Position.Absolute;
        toastOverlay.style.left = 0;
        toastOverlay.style.right = 0;
        toastOverlay.style.top = 14;
        toastOverlay.style.alignItems = Align.Center;
        toastOverlay.style.justifyContent = Justify.FlexStart;
        toastOverlay.style.display = DisplayStyle.None;
        toastOverlay.pickingMode = PickingMode.Ignore;

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
        toastOverlay.Add(bubble);

        var toastText = new Label();
        toastText.name = "toast-text";
        toastText.style.whiteSpace = WhiteSpace.NoWrap;
        toastText.style.overflow = Overflow.Hidden;
        toastText.style.textOverflow = TextOverflow.Ellipsis;
        toastText.style.color = Color.white;
        bubble.Add(toastText);

        return toastOverlay;
    }

    protected void ShowToast(VisualElement toastOverlay, string text, int milliseconds = 2000)
    {
        if (toastOverlay == null) return;

        var toastText = toastOverlay.Q<Label>("toast-text");
        if (toastText != null)
            toastText.text = text ?? "";

        toastOverlay.style.display = DisplayStyle.Flex;
        toastOverlay.BringToFront();

        toastOverlay.schedule.Execute(() =>
        {
            if (toastOverlay != null)
                toastOverlay.style.display = DisplayStyle.None;
        }).StartingIn(Mathf.Max(200, milliseconds));
    }

    protected bool IsLandscape()
    {
        return Screen.width > Screen.height;
    }
}
