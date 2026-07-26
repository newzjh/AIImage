#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Threading;
using Aexis.Samples.Async;
using UnityEditor;
using UnityEngine;

public sealed class AIImageModelDownloadWindow : EditorWindow
{
    private readonly Dictionary<AIImageModelGroupId, AIImageModelDownloadProgress> _progress =
        new Dictionary<AIImageModelGroupId, AIImageModelDownloadProgress>();
    private CancellationTokenSource _downloadCts;
    private AIImageModelGroupId? _activeGroup;
    private string _error;

    [MenuItem("Aexis/Examples/Models/Download Models...")]
    public static void Open()
    {
        var window = GetWindow<AIImageModelDownloadWindow>();
        window.titleContent = new GUIContent("Download Models");
        window.minSize = new Vector2(620f, 360f);
        window.Show();
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("AIImage Model Delivery", EditorStyles.boldLabel);
        EditorGUILayout.LabelField("Downloads are installed outside the project StreamingAssets tree.", EditorStyles.wordWrappedMiniLabel);
        EditorGUILayout.Space(8f);

        foreach (var group in AIImageModelDelivery.AllGroups)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(group.DisplayName, EditorStyles.boldLabel);
            GUILayout.FlexibleSpace();
            var installed = AIImageModelDelivery.IsInstalled(group);
            EditorGUILayout.LabelField(installed ? "Installed" : group.BundledByDefault ? "Bundled by release build" : "Optional", GUILayout.Width(150f));
            EditorGUI.BeginDisabledGroup(_activeGroup.HasValue);
            if (GUILayout.Button(installed ? "Download again" : "Download", GUILayout.Width(110f)))
                DownloadAsync(group).Forget();
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.EndHorizontal();

            if (_progress.TryGetValue(group.Id, out var progress))
            {
                var rect = GUILayoutUtility.GetRect(1f, 18f);
                EditorGUI.ProgressBar(rect, progress.Progress01, progress.Detail);
            }
            EditorGUILayout.EndVertical();
        }

        if (_activeGroup.HasValue && GUILayout.Button("Cancel current download"))
            _downloadCts?.Cancel();
        if (!string.IsNullOrWhiteSpace(_error))
            EditorGUILayout.HelpBox(_error, MessageType.Error);
    }

    private async UniTaskVoid DownloadAsync(AIImageModelGroup group)
    {
        _error = null;
        _activeGroup = group.Id;
        _downloadCts?.Dispose();
        _downloadCts = new CancellationTokenSource();
        try
        {
            await AIImageModelDelivery.DownloadGroupAsync(
                group,
                progress =>
                {
                    _progress[group.Id] = progress;
                    Repaint();
                },
                _downloadCts.Token);
        }
        catch (OperationCanceledException)
        {
            _error = "Download cancelled.";
        }
        catch (Exception exception)
        {
            _error = exception.Message;
            Debug.LogException(exception);
        }
        finally
        {
            _activeGroup = null;
            _downloadCts?.Dispose();
            _downloadCts = null;
            Repaint();
        }
    }

    private void OnDisable()
    {
        try { _downloadCts?.Cancel(); } catch { }
        _downloadCts?.Dispose();
        _downloadCts = null;
    }
}
#endif
