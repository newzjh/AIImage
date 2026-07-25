using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Aexis.Samples.Async;
using UnityEngine;
using UnityEngine.UIElements;

[DisallowMultipleComponent]
public sealed class AIImageModelDownloadDialog : MonoBehaviour
{
    private UIDocument _document;
    private VisualElement _overlay;
    private Label _title;
    private Label _detail;
    private ProgressBar _progress;
    private Button _primaryButton;
    private Button _cancelButton;
    private UniTaskCompletionSource<bool> _confirmation;
    private CancellationTokenSource _downloadCts;
    private bool _downloading;

    public void Configure(UIDocument document)
    {
        _document = document;
    }

    public async UniTask<bool> EnsureAvailableAsync(
        string operationName,
        IEnumerable<AIImageModelGroup> requestedGroups,
        CancellationToken cancellationToken = default)
    {
        var missing = new List<AIImageModelGroup>();
        if (requestedGroups != null)
        {
            foreach (var group in requestedGroups)
            {
                if (group == null || missing.Any(existing => existing.Id == group.Id))
                    continue;
                var materializationFailed = false;
                if (Application.platform == RuntimePlatform.Android
                    && !AIImageModelDelivery.IsInstalled(group)
                    && await AIImageModelDelivery.IsAvailableAsync(group, cancellationToken))
                {
                    try
                    {
                        await AIImageModelDelivery.MaterializeBundledGroupAsync(group, null, cancellationToken);
                    }
                    catch (Exception exception) when (!(exception is OperationCanceledException))
                    {
                        materializationFailed = true;
                        Debug.LogWarning("Unable to prepare bundled " + group.DisplayName + ". Falling back to download.\n" + exception);
                    }
                }

                if (materializationFailed || !await AIImageModelDelivery.IsAvailableAsync(group, cancellationToken))
                    missing.Add(group);
            }
        }

        if (missing.Count == 0)
            return true;

        EnsureUi();
        if (_overlay == null)
            return false;

        var names = string.Join("\n", missing.Select(group => "- " + group.DisplayName));
        _title.text = string.IsNullOrWhiteSpace(operationName) ? "Model download required" : operationName;
        _detail.text = "This action requires model files that are not installed.\n" + names;
        _progress.value = 0f;
        _progress.title = "";
        _primaryButton.text = "Download";
        _primaryButton.SetEnabled(true);
        _cancelButton.text = "Cancel";
        _cancelButton.SetEnabled(true);
        _downloading = false;
        _overlay.style.display = DisplayStyle.Flex;
        _overlay.BringToFront();

        _confirmation = new UniTaskCompletionSource<bool>();
        _primaryButton.clicked -= OnPrimaryClicked;
        _primaryButton.clicked += OnPrimaryClicked;
        _cancelButton.clicked -= OnCancelClicked;
        _cancelButton.clicked += OnCancelClicked;

        void OnPrimaryClicked()
        {
            if (_downloading)
                return;
            DownloadMissingAsync(missing, cancellationToken).Forget();
        }

        void OnCancelClicked()
        {
            if (_downloading)
            {
                try { _downloadCts?.Cancel(); } catch { }
                _detail.text = "Cancelling download...";
                _cancelButton.SetEnabled(false);
                return;
            }
            Complete(false);
        }

        try
        {
            return await _confirmation.Task;
        }
        finally
        {
            _primaryButton.clicked -= OnPrimaryClicked;
            _cancelButton.clicked -= OnCancelClicked;
        }
    }

    private async UniTaskVoid DownloadMissingAsync(IReadOnlyList<AIImageModelGroup> groups, CancellationToken callerToken)
    {
        _downloading = true;
        _primaryButton.SetEnabled(false);
        _cancelButton.text = "Cancel download";
        _downloadCts?.Dispose();
        _downloadCts = CancellationTokenSource.CreateLinkedTokenSource(callerToken);
        try
        {
            for (var index = 0; index < groups.Count; index++)
            {
                var group = groups[index];
                var groupStart = index / (float)groups.Count;
                var groupSpan = 1f / groups.Count;
                await AIImageModelDelivery.DownloadGroupAsync(
                    group,
                    progress => UpdateProgress(groupStart + progress.Progress01 * groupSpan, progress),
                    _downloadCts.Token);
            }
            Complete(true);
        }
        catch (OperationCanceledException)
        {
            Complete(false);
        }
        catch (Exception exception)
        {
            _detail.text = "Download failed: " + exception.Message;
            _progress.title = "Download failed";
            _cancelButton.text = "Close";
            _cancelButton.SetEnabled(true);
            _downloading = false;
            _primaryButton.text = "Retry";
            _primaryButton.SetEnabled(true);
            Debug.LogException(exception);
        }
        finally
        {
            _downloadCts?.Dispose();
            _downloadCts = null;
        }
    }

    private void UpdateProgress(float value, AIImageModelDownloadProgress progress)
    {
        if (_progress == null)
            return;

        _progress.value = Mathf.Clamp01(value) * 100f;
        _progress.title = Mathf.RoundToInt(_progress.value) + "%";
        _detail.text = progress.Group.DisplayName + "\n" + progress.Detail;
    }

    private void Complete(bool value)
    {
        _downloading = false;
        _overlay.style.display = DisplayStyle.None;
        var completion = _confirmation;
        _confirmation = null;
        completion?.TrySetResult(value);
    }

    private void EnsureUi()
    {
        if (_overlay != null)
            return;
        if (_document == null)
            _document = GetComponent<UIDocument>();
        var root = _document?.rootVisualElement;
        if (root == null)
            return;

        _overlay = new VisualElement();
        _overlay.style.position = Position.Absolute;
        _overlay.style.left = 0;
        _overlay.style.top = 0;
        _overlay.style.right = 0;
        _overlay.style.bottom = 0;
        _overlay.style.backgroundColor = new StyleColor(new Color(0f, 0f, 0f, 0.68f));
        _overlay.style.alignItems = Align.Center;
        _overlay.style.justifyContent = Justify.Center;
        _overlay.style.display = DisplayStyle.None;

        var panel = new VisualElement();
        panel.style.width = 460;
        panel.style.maxWidth = Length.Percent(90);
        panel.style.backgroundColor = new StyleColor(new Color(0.10f, 0.11f, 0.14f, 1f));
        panel.style.borderTopLeftRadius = 8;
        panel.style.borderTopRightRadius = 8;
        panel.style.borderBottomLeftRadius = 8;
        panel.style.borderBottomRightRadius = 8;
        panel.style.paddingLeft = 18;
        panel.style.paddingRight = 18;
        panel.style.paddingTop = 16;
        panel.style.paddingBottom = 16;
        panel.style.flexDirection = FlexDirection.Column;
        _overlay.Add(panel);

        _title = new Label();
        _title.style.color = Color.white;
        _title.style.fontSize = 17;
        _title.style.unityFontStyleAndWeight = FontStyle.Bold;
        _title.style.marginBottom = 10;
        panel.Add(_title);

        _detail = new Label();
        _detail.style.whiteSpace = WhiteSpace.Normal;
        _detail.style.color = new Color(0.84f, 0.87f, 0.92f, 1f);
        _detail.style.marginBottom = 12;
        panel.Add(_detail);

        _progress = new ProgressBar { lowValue = 0f, highValue = 100f };
        _progress.style.height = 18;
        _progress.style.marginBottom = 14;
        panel.Add(_progress);

        var buttons = new VisualElement();
        buttons.style.flexDirection = FlexDirection.Row;
        buttons.style.justifyContent = Justify.FlexEnd;
        panel.Add(buttons);

        _cancelButton = new Button { text = "Cancel" };
        _cancelButton.style.height = 32;
        _cancelButton.style.marginRight = 8;
        buttons.Add(_cancelButton);

        _primaryButton = new Button { text = "Download" };
        _primaryButton.style.height = 32;
        _primaryButton.style.backgroundColor = new StyleColor(new Color(0.16f, 0.48f, 0.86f, 1f));
        _primaryButton.style.color = Color.white;
        buttons.Add(_primaryButton);

        root.Add(_overlay);
    }

    private void OnDestroy()
    {
        try { _downloadCts?.Cancel(); } catch { }
        _downloadCts?.Dispose();
        _downloadCts = null;
    }
}
