#if ENABLE_INPUT_SYSTEM
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.UI;

internal static class AIImageInputSystemSupport
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Register()
    {
        AIImageInputCompatibility.RegisterNewInputSystem(IsPrimaryPointerPressed, ConfigureUiInputModule);
    }

    private static bool IsPrimaryPointerPressed()
    {
        if (Mouse.current?.leftButton.isPressed == true)
            return true;

        var touchScreen = Touchscreen.current;
        if (touchScreen == null)
            return false;

        foreach (var touch in touchScreen.touches)
            if (touch.press.isPressed)
                return true;
        return false;
    }

    private static void ConfigureUiInputModule()
    {
        var eventSystem = EventSystem.current;
        if (eventSystem == null)
            eventSystem = Object.FindFirstObjectByType<EventSystem>();
        if (eventSystem == null)
            eventSystem = new GameObject("EventSystem", typeof(EventSystem)).GetComponent<EventSystem>();

        var inputSystemModule = eventSystem.GetComponent<InputSystemUIInputModule>();
        if (inputSystemModule == null)
        {
            inputSystemModule = eventSystem.gameObject.AddComponent<InputSystemUIInputModule>();
            inputSystemModule.AssignDefaultActions();
        }

        // EventSystem selects one active input module. This module handles mouse,
        // touch, keyboard, and navigation in both New and Both configurations.
        var legacyModule = eventSystem.GetComponent<StandaloneInputModule>();
        if (legacyModule != null)
            legacyModule.enabled = false;
        inputSystemModule.enabled = true;
    }
}
#endif
