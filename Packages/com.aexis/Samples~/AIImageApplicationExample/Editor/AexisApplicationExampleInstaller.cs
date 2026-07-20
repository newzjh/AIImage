#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

internal static class AexisApplicationExampleInstaller
{
    private const string InstallerScriptName = "AexisApplicationExampleInstaller";
    private const string Destination = "Assets/StreamingAssets";

    [MenuItem("Aexis/Examples/Install Main2 Application StreamingAssets")]
    private static void InstallStreamingAssets()
    {
        var root = ResolveSampleRoot();
        CopyDirectory(Path.Combine(root, "StreamingAssets"), Path.GetFullPath(Destination));
        AssetDatabase.Refresh();
        Debug.Log("Installed AIImage application configuration and permitted default models to " + Destination + ".");
    }

    [MenuItem("Aexis/Examples/Open Main2 Application Scene")]
    private static void OpenMain2Scene()
    {
        var root = ResolveSampleRoot();
        EditorSceneManager.OpenScene(ToAssetPath(Path.Combine(root, "Scenes", "Main2.unity")), OpenSceneMode.Single);
    }

    private static string ResolveSampleRoot()
    {
        var guids = AssetDatabase.FindAssets(InstallerScriptName);
        if (guids.Length != 1)
            throw new InvalidOperationException("Could not resolve the AIImage application sample root.");
        var scriptPath = AssetDatabase.GUIDToAssetPath(guids[0]);
        return Path.GetFullPath(Path.Combine(Path.GetDirectoryName(scriptPath), ".."));
    }

    private static string ToAssetPath(string absolutePath)
    {
        var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        return absolutePath.Substring(projectRoot.Length + 1).Replace('\\', '/');
    }

    private static void CopyDirectory(string source, string destination)
    {
        if (!Directory.Exists(source))
            return;
        Directory.CreateDirectory(destination);
        foreach (var directory in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
            Directory.CreateDirectory(directory.Replace(source, destination));
        foreach (var file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
        {
            if (string.Equals(Path.GetExtension(file), ".meta", StringComparison.OrdinalIgnoreCase))
                continue;
            File.Copy(file, file.Replace(source, destination), true);
        }
    }
}
#endif
