using System;
using System.IO;
using System.Threading;
using Aexis.Execution;
using Aexis.Samples.Async;
using UnityEngine;
using UnityEngine.Networking;
#if UNITY_EDITOR
using UnityEditor.PackageManager;
#endif

namespace Aexis.Samples
{
    public static class AexisSampleStreamingAssets
    {
        private const string PackageName = "com.aexis";
        private const string SampleStreamingAssetsRelativePath = "Samples/AIImageApplicationExample/StreamingAssets";

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void RegisterManifestPathResolverAtRuntime()
        {
            RegisterManifestPathResolver();
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
        private static void RegisterManifestPathResolverInEditor()
        {
            RegisterManifestPathResolver();
        }
#endif

        public static void RegisterManifestPathResolver()
        {
            AexisModelManifestLoader.DefaultStreamingAssetsPathResolver = relativePath =>
                TryResolveFilePath(relativePath, out var path) ? path : null;
        }

        public static string ResolveFilePath(string relativePath)
        {
            if (TryResolveFilePath(relativePath, out var path))
                return path;
            throw NewMissingPathException(relativePath, expectDirectory: false);
        }

        public static string ResolveDirectoryPath(string relativePath = null)
        {
            if (TryResolveDirectoryPath(relativePath, out var path))
                return path;
            throw NewMissingPathException(relativePath, expectDirectory: true);
        }

        public static bool TryResolveFilePath(string relativePath, out string path)
        {
            return TryResolvePath(relativePath, expectDirectory: false, out path);
        }

        public static bool TryResolveDirectoryPath(string relativePath, out string path)
        {
            return TryResolvePath(relativePath, expectDirectory: true, out path);
        }

        public static async UniTask<byte[]> ReadBytesAsync(string relativePath, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(relativePath))
                throw new ArgumentException("A StreamingAssets-relative path is required.", nameof(relativePath));

            var path = ResolveFilePath(relativePath);
            var requestPath = Path.IsPathRooted(path) ? new Uri(path).AbsoluteUri : path;
            using var request = UnityWebRequest.Get(requestPath);
            var operation = request.SendWebRequest();
            while (!operation.isDone)
            {
                cancellationToken.ThrowIfCancellationRequested();
                await UniTask.NextFrame();
            }

            if (request.result != UnityWebRequest.Result.Success)
                throw new IOException("Unable to load StreamingAssets file '" + relativePath + "': " + request.error);
            return request.downloadHandler.data;
        }

        public static async UniTask<string> ReadTextAsync(string relativePath, CancellationToken cancellationToken = default)
        {
            var bytes = await ReadBytesAsync(relativePath, cancellationToken);
            return System.Text.Encoding.UTF8.GetString(bytes);
        }

        private static bool TryResolvePath(string relativePath, bool expectDirectory, out string path)
        {
            if (Path.IsPathRooted(relativePath ?? string.Empty))
            {
                path = relativePath;
                return true;
            }

            var normalized = (relativePath ?? string.Empty).TrimStart('/', '\\').Replace('/', Path.DirectorySeparatorChar);
            if (!string.IsNullOrWhiteSpace(normalized))
            {
                var persistentPath = AIImageModelDelivery.GetPersistentPath(normalized);
                if (PathExists(persistentPath, expectDirectory))
                {
                    path = persistentPath;
                    return true;
                }
            }

            var playerRoot = Application.streamingAssetsPath.TrimEnd('/', '\\');
            var playerPath = string.IsNullOrEmpty(normalized) ? playerRoot : Path.Combine(playerRoot, normalized);

#if UNITY_EDITOR
            if (PathExists(playerPath, expectDirectory))
            {
                path = playerPath;
                return true;
            }

            var packageRoot = ResolveEditorPackageStreamingAssetsRoot();
            var packagePath = string.IsNullOrEmpty(packageRoot)
                ? null
                : string.IsNullOrEmpty(normalized) ? packageRoot : Path.Combine(packageRoot, normalized);
            if (PathExists(packagePath, expectDirectory))
            {
                path = packagePath;
                return true;
            }

            path = null;
            return false;
#else
            path = playerPath;
            return true;
#endif
        }

        private static FileNotFoundException NewMissingPathException(string relativePath, bool expectDirectory)
        {
            var normalized = (relativePath ?? string.Empty).TrimStart('/', '\\').Replace('/', Path.DirectorySeparatorChar);
            var playerRoot = Application.streamingAssetsPath.TrimEnd('/', '\\');
            var playerPath = string.IsNullOrEmpty(normalized) ? playerRoot : Path.Combine(playerRoot, normalized);
#if UNITY_EDITOR
            var packageRoot = ResolveEditorPackageStreamingAssetsRoot();
            var packagePath = string.IsNullOrEmpty(packageRoot)
                ? "<package unavailable>"
                : string.IsNullOrEmpty(normalized) ? packageRoot : Path.Combine(packageRoot, normalized);
            return new FileNotFoundException(
                "Aexis sample StreamingAssets " + (expectDirectory ? "directory" : "file") + " was not found. "
                + "Checked Assets path '" + playerPath + "' and package sample path '" + packagePath + "'.",
                relativePath);
#else
            return new FileNotFoundException(
                "Aexis player StreamingAssets " + (expectDirectory ? "directory" : "file") + " was not found in the built player path '" + playerPath + "'.",
                relativePath);
#endif
        }

        private static bool PathExists(string path, bool expectDirectory)
        {
            return !string.IsNullOrWhiteSpace(path) && (expectDirectory ? Directory.Exists(path) : File.Exists(path));
        }

#if UNITY_EDITOR
        private static string ResolveEditorPackageStreamingAssetsRoot()
        {
            var packages = PackageInfo.GetAllRegisteredPackages();
            for (var i = 0; i < packages.Length; i++)
            {
                var package = packages[i];
                if (package == null || !string.Equals(package.name, PackageName, StringComparison.Ordinal))
                    continue;
                var root = Path.Combine(package.resolvedPath, SampleStreamingAssetsRelativePath);
                if (Directory.Exists(root))
                    return root;
            }
            return null;
        }
#endif
    }
}
