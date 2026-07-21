using System;
using UnityEngine;

public static class MonaiLowPowerModeState
{
    private const string PlayerPrefsKey = "Monai.LowPowerMode";
    private const string ProjectEnvironmentVariableName = "AIIMAGE_MONAI_LOW_POWER_MODE";
    private const string LegacyEnvironmentVariableName = "MONAI_LOW_POWER_MODE";

    private static bool _hasCachedValue;
    private static bool _cachedValue;

    public static event Action<bool> Changed;

    public static bool IsEnabled
    {
        get
        {
            if (TryReadEnvironmentOverride(out var enabled))
                return enabled;

            EnsureLoaded();
            return _cachedValue;
        }
    }

    public static void SetEnabled(bool enabled)
    {
        EnsureLoaded();
        if (_cachedValue == enabled)
            return;

        _cachedValue = enabled;
        PlayerPrefs.SetInt(PlayerPrefsKey, enabled ? 1 : 0);
        PlayerPrefs.Save();
        Changed?.Invoke(enabled);
    }

    private static void EnsureLoaded()
    {
        if (_hasCachedValue)
            return;

        _cachedValue = PlayerPrefs.GetInt(PlayerPrefsKey, 0) != 0;
        _hasCachedValue = true;
    }

    private static bool TryReadEnvironmentOverride(out bool enabled)
    {
        enabled = false;
        return TryReadBooleanEnvironmentVariable(ProjectEnvironmentVariableName, out enabled)
            || TryReadBooleanEnvironmentVariable(LegacyEnvironmentVariableName, out enabled);
    }

    private static bool TryReadBooleanEnvironmentVariable(string name, out bool enabled)
    {
        enabled = false;
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
            return false;

        raw = raw.Trim();
        if (raw == "1"
            || raw.Equals("true", StringComparison.OrdinalIgnoreCase)
            || raw.Equals("yes", StringComparison.OrdinalIgnoreCase)
            || raw.Equals("on", StringComparison.OrdinalIgnoreCase))
        {
            enabled = true;
            return true;
        }

        if (raw == "0"
            || raw.Equals("false", StringComparison.OrdinalIgnoreCase)
            || raw.Equals("no", StringComparison.OrdinalIgnoreCase)
            || raw.Equals("off", StringComparison.OrdinalIgnoreCase))
        {
            enabled = false;
            return true;
        }

        return false;
    }
}
