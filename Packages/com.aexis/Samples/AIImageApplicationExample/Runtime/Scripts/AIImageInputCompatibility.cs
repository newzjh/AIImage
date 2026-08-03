using System;
using UnityEngine;

/// <summary>Provides optional input backends without making the application sample depend on Input System.</summary>
public static class AIImageInputCompatibility
{
    private static Func<bool> _newInputSystemPointerPressed;
    private static Action _newInputSystemUiConfigurator;

    public static void RegisterNewInputSystem(Func<bool> pointerPressed, Action configureUiInputModule)
    {
        _newInputSystemPointerPressed = pointerPressed;
        _newInputSystemUiConfigurator = configureUiInputModule;
    }

    public static void ConfigureUiInputModule()
    {
        _newInputSystemUiConfigurator?.Invoke();
    }

    public static bool IsPrimaryPointerPressed()
    {
        if (_newInputSystemPointerPressed != null && _newInputSystemPointerPressed())
            return true;

#if ENABLE_LEGACY_INPUT_MANAGER
        return Input.GetMouseButton(0) || Input.touchCount > 0;
#else
        return false;
#endif
    }
}
