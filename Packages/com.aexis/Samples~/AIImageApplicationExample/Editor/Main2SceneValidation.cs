#if UNITY_EDITOR
using System;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class Main2SceneValidation
{
    public static void ValidateMain2SceneBatch()
    {
        const string scenePath = "Assets/Scenes/Main2.unity";

        try
        {
            var scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
            if (!scene.IsValid())
                throw new InvalidOperationException("Failed to open Main2 scene.");

            var host = UnityEngine.Object.FindFirstObjectByType<AIImagePageHost>(FindObjectsInactive.Exclude);
            if (host == null)
                throw new InvalidOperationException("AIImagePageHost not found in Main2 scene.");

            var document = host.GetComponent<UnityEngine.UIElements.UIDocument>();
            if (document == null)
                throw new InvalidOperationException("UIDocument missing on AIImagePageHost object.");

            var awake = typeof(AIImagePageHost).GetMethod("Awake", BindingFlags.Instance | BindingFlags.NonPublic);
            var onEnable = typeof(AIImagePageHost).GetMethod("OnEnable", BindingFlags.Instance | BindingFlags.NonPublic);
            awake?.Invoke(host, null);
            onEnable?.Invoke(host, null);

            EditorApplication.QueuePlayerLoopUpdate();
            AssetDatabase.Refresh();

            Debug.Log("[Main2SceneValidation] Main2 scene loaded and AIImagePageHost resolved successfully.");
            EditorApplication.Exit(0);
        }
        catch (Exception ex)
        {
            Debug.LogError("[Main2SceneValidation] Validation failed: " + ex.Message);
            Debug.LogException(ex);
            EditorApplication.Exit(1);
        }
    }
}
#endif
