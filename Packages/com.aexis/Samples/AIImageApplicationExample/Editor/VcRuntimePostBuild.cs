#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEngine;

public static class VcRuntimePostBuild
{
    [PostProcessBuild(20)]
    public static void OnPostProcessBuild(BuildTarget target, string pathToBuiltProject)
    {
        if (target != BuildTarget.StandaloneWindows && target != BuildTarget.StandaloneWindows64)
            return;

        var exePath = pathToBuiltProject;
        var buildDir = Path.GetDirectoryName(exePath);
        if (string.IsNullOrWhiteSpace(buildDir) || !Directory.Exists(buildDir))
            return;

        var srcDir = Path.Combine(Application.dataPath, "Plugins", "x86_64");
        if (!Directory.Exists(srcDir))
            return;

        CopyIfExists(srcDir, buildDir, "MSVCP140.dll");
        CopyIfExists(srcDir, buildDir, "VCRUNTIME140.dll");
        CopyIfExists(srcDir, buildDir, "VCRUNTIME140_1.dll");
        CopyIfExists(srcDir, buildDir, "CONCRT140.dll");
    }

    private static void CopyIfExists(string srcDir, string dstDir, string fileName)
    {
        try
        {
            var src = Path.Combine(srcDir, fileName);
            if (!File.Exists(src))
                return;
            var dst = Path.Combine(dstDir, fileName);
            File.Copy(src, dst, true);
        }
        catch (Exception e)
        {
            Debug.LogWarning("VC runtime copy failed: " + fileName + " " + e.Message);
        }
    }
}
#endif

