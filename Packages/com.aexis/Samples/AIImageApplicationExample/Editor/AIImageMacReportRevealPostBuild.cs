#if UNITY_EDITOR
using System;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

internal sealed class AIImageMacReportRevealPostBuild : IPostprocessBuildWithReport
{
    public int callbackOrder => 1000;

    public void OnPostprocessBuild(BuildReport report)
    {
        if (report.summary.platform != BuildTarget.StandaloneOSX)
            return;
        if (Application.platform != RuntimePlatform.OSXEditor)
            throw new BuildFailedException("The Aexis macOS report-reveal plug-in must be built on macOS.");

        var appPath = report.summary.outputPath;
        var sourcePath = Path.Combine(Application.dataPath, "Plugins", "macOS", "AIImageReportReveal.mm");
        var pluginDirectory = Path.Combine(appPath, "Contents", "Plugins");
        var pluginPath = Path.Combine(pluginDirectory, "libAIImageReportReveal.dylib");
        if (!File.Exists(sourcePath))
            throw new BuildFailedException("Aexis macOS report-reveal source is missing: " + sourcePath);

        Directory.CreateDirectory(pluginDirectory);
        Run("/usr/bin/xcrun", "--sdk macosx clang++ -dynamiclib -fobjc-arc -arch x86_64 -arch arm64 -framework Cocoa "
            + Quote(sourcePath) + " -o " + Quote(pluginPath));
        Run("/usr/bin/codesign", "--force --sign - --timestamp=none " + Quote(pluginPath));
        UnityEngine.Debug.Log("Aexis macOS runner-report reveal plug-in created: " + pluginPath);
    }

    private static void Run(string executable, string arguments)
    {
        using (var process = Process.Start(new ProcessStartInfo
        {
            FileName = executable,
            Arguments = arguments,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        }))
        {
            if (process == null)
                throw new BuildFailedException("Could not start macOS report-reveal build command: " + executable);
            var standardOutput = process.StandardOutput.ReadToEnd();
            var standardError = process.StandardError.ReadToEnd();
            process.WaitForExit();
            if (process.ExitCode != 0)
                throw new BuildFailedException(
                    "Could not build macOS report-reveal plug-in (exit " + process.ExitCode + ").\n"
                    + standardOutput + "\n" + standardError);
        }
    }

    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }
}
#endif
