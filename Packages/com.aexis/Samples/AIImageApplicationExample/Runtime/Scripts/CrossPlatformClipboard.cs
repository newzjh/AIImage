using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UIElements;

public static class CrossPlatformClipboard
{
    public static void Copy(string text)
    {
        text ??= string.Empty;
        GUIUtility.systemCopyBuffer = text;

#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            using var unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
            using var activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
            using var clipboard = activity.Call<AndroidJavaObject>("getSystemService", "clipboard");
            using var clipData = new AndroidJavaClass("android.content.ClipData");
            using var clip = clipData.CallStatic<AndroidJavaObject>("newPlainText", "AIImage", text);
            clipboard.Call("setPrimaryClip", clip);
        }
        catch
        {
        }
#elif UNITY_IOS && !UNITY_EDITOR
        try
        {
            AIImageClipboard_SetText(text);
        }
        catch
        {
        }
#endif
    }

    public static string Paste()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            using var unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
            using var activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
            using var clipboard = activity.Call<AndroidJavaObject>("getSystemService", "clipboard");
            if (clipboard == null || !clipboard.Call<bool>("hasPrimaryClip"))
                return GUIUtility.systemCopyBuffer;

            using var clip = clipboard.Call<AndroidJavaObject>("getPrimaryClip");
            if (clip == null || clip.Call<int>("getItemCount") < 1)
                return GUIUtility.systemCopyBuffer;

            using var item = clip.Call<AndroidJavaObject>("getItemAt", 0);
            using var value = item.Call<AndroidJavaObject>("coerceToText", activity);
            return value?.Call<string>("toString") ?? GUIUtility.systemCopyBuffer;
        }
        catch
        {
            return GUIUtility.systemCopyBuffer;
        }
#elif UNITY_IOS && !UNITY_EDITOR
        try
        {
            var pointer = AIImageClipboard_GetText();
            if (pointer == IntPtr.Zero)
                return GUIUtility.systemCopyBuffer;

            try
            {
                return Marshal.PtrToStringUTF8(pointer) ?? string.Empty;
            }
            finally
            {
                AIImageClipboard_FreeText(pointer);
            }
        }
        catch
        {
            return GUIUtility.systemCopyBuffer;
        }
#else
        return GUIUtility.systemCopyBuffer;
#endif
    }

    public static void EnableTextFieldClipboard(TextField field)
    {
        if (field == null)
            return;

        field.RegisterCallback<KeyDownEvent>(evt =>
        {
            if (!evt.ctrlKey && !evt.commandKey)
            {
                if (evt.keyCode == KeyCode.Backspace || evt.keyCode == KeyCode.Delete)
                {
                    DeleteSelection(field, evt.keyCode == KeyCode.Backspace);
                    evt.StopImmediatePropagation();
                }
                return;
            }

            if (evt.keyCode == KeyCode.V)
            {
                ReplaceSelection(field, Paste());
                evt.StopImmediatePropagation();
            }
            else if (evt.keyCode == KeyCode.C)
            {
                Copy(GetSelection(field));
                evt.StopImmediatePropagation();
            }
            else if (evt.keyCode == KeyCode.X)
            {
                Copy(GetSelection(field));
                ReplaceSelection(field, string.Empty);
                evt.StopImmediatePropagation();
            }
        }, TrickleDown.TrickleDown);

        field.RegisterCallback<ContextualMenuPopulateEvent>(evt =>
        {
            evt.menu.AppendAction(
                AppLocalization.Text("Paste", "\u7c98\u8d34"),
                _ => ReplaceSelection(field, Paste()));
            evt.menu.AppendAction(
                AppLocalization.Text("Copy", "\u590d\u5236"),
                _ => Copy(GetSelection(field)));
            evt.menu.AppendAction(
                AppLocalization.Text("Cut", "\u526a\u5207"),
                _ =>
                {
                    Copy(GetSelection(field));
                    ReplaceSelection(field, string.Empty);
                });
        });
    }

    private static string GetSelection(TextField field)
    {
        var value = field.value ?? string.Empty;
        var start = Mathf.Clamp(Mathf.Min(field.cursorIndex, field.selectIndex), 0, value.Length);
        var end = Mathf.Clamp(Mathf.Max(field.cursorIndex, field.selectIndex), start, value.Length);
        return value.Substring(start, end - start);
    }

    private static void ReplaceSelection(TextField field, string replacement)
    {
        if (field == null)
            return;

        var value = field.value ?? string.Empty;
        var start = Mathf.Clamp(Mathf.Min(field.cursorIndex, field.selectIndex), 0, value.Length);
        var end = Mathf.Clamp(Mathf.Max(field.cursorIndex, field.selectIndex), start, value.Length);
        replacement ??= string.Empty;
        field.value = value.Remove(start, end - start).Insert(start, replacement);
        field.cursorIndex = start + replacement.Length;
        field.selectIndex = field.cursorIndex;
    }

    private static void DeleteSelection(TextField field, bool backspace)
    {
        if (field == null)
            return;

        var value = field.value ?? string.Empty;
        var start = Mathf.Clamp(Mathf.Min(field.cursorIndex, field.selectIndex), 0, value.Length);
        var end = Mathf.Clamp(Mathf.Max(field.cursorIndex, field.selectIndex), start, value.Length);
        if (start == end)
        {
            if (backspace && start > 0)
                start--;
            else if (!backspace && end < value.Length)
                end++;
        }

        if (start == end)
            return;

        field.value = value.Remove(start, end - start);
        field.cursorIndex = start;
        field.selectIndex = start;
    }

#if UNITY_IOS && !UNITY_EDITOR
    [DllImport("__Internal")]
    private static extern void AIImageClipboard_SetText(string text);

    [DllImport("__Internal")]
    private static extern IntPtr AIImageClipboard_GetText();

    [DllImport("__Internal")]
    private static extern void AIImageClipboard_FreeText(IntPtr text);
#endif
}
