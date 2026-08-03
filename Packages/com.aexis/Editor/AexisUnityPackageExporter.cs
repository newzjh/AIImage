#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace Aexis.Editor
{
    /// <summary>
    /// Exports com.aexis as a normal UnityPackage file list. The package files retain
    /// their Packages/com.aexis paths; no opaque model or source archive is embedded.
    /// </summary>
    public static class AexisUnityPackageExporter
    {
        private const string ExportOutputEnvironmentVariable = "AEXIS_UNITYPACKAGE_OUTPUT";
        private const string PackageName = "com.aexis";
        private const string PackageRelativePath = "Packages/" + PackageName;
        private const string BootstrapRelativePath = "Assets/AexisPackageBootstrap/Editor/AexisUnityPackageImportBootstrap.cs";
        private const string BootstrapScriptGuid = "86b5d2cfd2bd4a14a35f157d6f9da1ce";

        [MenuItem("Aexis/Release/Export Complete UnityPackage")]
        public static void ExportCompleteUnityPackage()
        {
            var root = ResolveAexisRoot();
            var version = ReadPackageVersion(root);
            ExportCompleteUnityPackage(root, ResolveOutputPath(version));
        }

        /// <summary>Exports the complete package to an explicit UnityPackage path.</summary>
        public static void ExportCompleteUnityPackage(string outputPath)
        {
            ExportCompleteUnityPackage(ResolveAexisRoot(), outputPath);
        }

        private static void ExportCompleteUnityPackage(string root, string outputPath)
        {
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException("An output path is required.", nameof(outputPath));

            outputPath = Path.GetFullPath(outputPath);
            var outputDirectory = Path.GetDirectoryName(outputPath);
            if (string.IsNullOrWhiteSpace(outputDirectory))
                throw new ArgumentException("The output path must include a directory.", nameof(outputPath));

            Directory.CreateDirectory(outputDirectory);
            ExportPackageFileList(root, outputPath);
            Debug.Log("Aexis UnityPackage exported: " + outputPath);
        }

        internal static void ExportPackageFileList(string packageRoot, string outputPath)
        {
            if (string.IsNullOrWhiteSpace(packageRoot))
                throw new ArgumentException("A package root is required.", nameof(packageRoot));
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException("An output path is required.", nameof(outputPath));
            if (!File.Exists(Path.Combine(packageRoot, "package.json")))
                throw new FileNotFoundException("The Aexis package manifest was not found.", packageRoot);

            var entries = CollectPackageEntries(packageRoot);
            entries.Add(CreateBootstrapEntry());

            using (var output = File.Create(outputPath))
            using (var gzip = new GZipStream(output, System.IO.Compression.CompressionLevel.Optimal, leaveOpen: false))
            using (var writer = new UnityPackageTarWriter(gzip))
            {
                foreach (var entry in entries.OrderBy(candidate => candidate.PathName, StringComparer.Ordinal))
                    writer.Write(entry);
            }
        }

        private static List<UnityPackageEntry> CollectPackageEntries(string packageRoot)
        {
            var files = Directory.GetFiles(packageRoot, "*", SearchOption.AllDirectories)
                .Where(path => !IsUnityPackageExportSupportFile(packageRoot, path))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            var metaFiles = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            var assetFiles = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

            foreach (var file in files)
            {
                var relativePath = ToPackageRelativePath(packageRoot, file);
                if (relativePath.EndsWith(".meta", StringComparison.OrdinalIgnoreCase))
                    metaFiles[relativePath.Substring(0, relativePath.Length - ".meta".Length)] = file;
                else
                    assetFiles[relativePath] = file;
            }

            var entryPaths = new HashSet<string>(assetFiles.Keys, StringComparer.OrdinalIgnoreCase);
            foreach (var metaPath in metaFiles.Keys)
                entryPaths.Add(metaPath);

            var entries = new List<UnityPackageEntry>(entryPaths.Count + 1);
            foreach (var relativePath in entryPaths.OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            {
                metaFiles.TryGetValue(relativePath, out var metaPath);
                assetFiles.TryGetValue(relativePath, out var assetPath);
                entries.Add(new UnityPackageEntry(
                    PackageRelativePath + "/" + relativePath,
                    ReadGuid(metaPath, PackageRelativePath + "/" + relativePath),
                    assetPath,
                    metaPath,
                    null,
                    null));
            }

            return entries;
        }

        private static UnityPackageEntry CreateBootstrapEntry()
        {
            var script = Encoding.UTF8.GetBytes(GetBootstrapSource());
            var meta = Encoding.UTF8.GetBytes(
                "fileFormatVersion: 2\n"
                + "guid: " + BootstrapScriptGuid + "\n"
                + "MonoImporter:\n"
                + "  externalObjects: {}\n"
                + "  serializedVersion: 2\n"
                + "  defaultReferences: []\n"
                + "  executionOrder: 0\n"
                + "  icon: {instanceID: 0}\n"
                + "  userData: \n"
                + "  assetBundleName: \n"
                + "  assetBundleVariant: \n");
            return new UnityPackageEntry(BootstrapRelativePath, BootstrapScriptGuid, null, null, script, meta);
        }

        private static bool IsUnityPackageExportSupportFile(string packageRoot, string file)
        {
            var relativePath = file.Substring(packageRoot.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return relativePath.StartsWith("UnityPackage~/", StringComparison.OrdinalIgnoreCase)
                || relativePath.StartsWith("UnityPackage~\\", StringComparison.OrdinalIgnoreCase);
        }

        private static string ToPackageRelativePath(string packageRoot, string path)
        {
            return path.Substring(packageRoot.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Replace('\\', '/');
        }

        private static string ReadGuid(string metaPath, string fallbackKey)
        {
            if (!string.IsNullOrEmpty(metaPath) && File.Exists(metaPath))
            {
                foreach (var line in File.ReadLines(metaPath))
                {
                    const string prefix = "guid: ";
                    if (line.StartsWith(prefix, StringComparison.Ordinal) && line.Length == prefix.Length + 32)
                        return line.Substring(prefix.Length);
                }
            }

            using (var md5 = MD5.Create())
            {
                var hash = md5.ComputeHash(Encoding.UTF8.GetBytes("AexisUnityPackage/" + fallbackKey));
                var builder = new StringBuilder(hash.Length * 2);
                foreach (var value in hash)
                    builder.Append(value.ToString("x2"));
                return builder.ToString();
            }
        }

        private static string ResolveAexisRoot()
        {
            var guids = AssetDatabase.FindAssets(nameof(AexisUnityPackageExporter));
            if (guids.Length != 1)
                throw new InvalidOperationException("Could not resolve the Aexis UnityPackage exporter asset.");

            var scriptPath = AssetDatabase.GUIDToAssetPath(guids[0]);
            var editorDirectory = Path.GetDirectoryName(Path.GetFullPath(scriptPath));
            return Directory.GetParent(editorDirectory).FullName;
        }

        private static string ReadPackageVersion(string root)
        {
            var packageJson = Path.Combine(root, "package.json");
            if (!File.Exists(packageJson))
                return "local";

            var package = JsonUtility.FromJson<PackageMetadata>(File.ReadAllText(packageJson));
            return string.IsNullOrWhiteSpace(package.version) ? "local" : package.version;
        }

        private static string ResolveOutputPath(string version)
        {
            var configured = Environment.GetEnvironmentVariable(ExportOutputEnvironmentVariable);
            if (!string.IsNullOrWhiteSpace(configured))
                return Path.GetFullPath(configured);

            var projectRoot = Directory.GetParent(Application.dataPath).FullName;
            return Path.Combine(projectRoot, "output", "Aexis-" + version + ".unitypackage");
        }

        private static string GetBootstrapSource()
        {
            return @"#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;
using UnityEngine;

[InitializeOnLoad]
internal static class AexisUnityPackageImportBootstrap
{
    private const string PackageName = ""com.aexis"";
    private const string PackageManifestPath = ""Packages/com.aexis/package.json"";
    private const string PackageFileReference = ""file:Packages/com.aexis"";
    private const string BootstrapAssetPath = ""Assets/AexisPackageBootstrap"";
    private static AddRequest _addRequest;
    private static double _waitStarted;

    static AexisUnityPackageImportBootstrap()
    {
        EditorApplication.delayCall += RegisterPackage;
    }

    private static void RegisterPackage()
    {
        try
        {
            if (UnityEditor.PackageManager.PackageInfo.FindForAssetPath(PackageManifestPath) != null)
            {
                CleanupBootstrap();
                return;
            }

            var manifestPath = Path.Combine(ProjectRoot, PackageManifestPath);
            if (!File.Exists(manifestPath))
                throw new FileNotFoundException(""Aexis package files were not imported to Packages/com.aexis. Reimport the UnityPackage with all Aexis entries selected."", manifestPath);

            _waitStarted = EditorApplication.timeSinceStartup;
            _addRequest = Client.Add(PackageFileReference);
            EditorApplication.update += AwaitPackageRegistration;
        }
        catch (Exception exception)
        {
            Debug.LogError(""Aexis UnityPackage registration failed: "" + exception);
        }
    }

    private static void AwaitPackageRegistration()
    {
        if (_addRequest != null && _addRequest.IsCompleted)
        {
            EditorApplication.update -= AwaitPackageRegistration;
            if (_addRequest.Status == StatusCode.Success && _addRequest.Result != null && _addRequest.Result.name == PackageName)
            {
                Debug.Log(""Aexis package imported and registered at Packages/com.aexis."");
                CleanupBootstrap();
            }
            else
            {
                Debug.LogError(""Aexis UnityPackage registration failed: "" + (_addRequest.Error == null ? ""unknown Package Manager error"" : _addRequest.Error.message));
            }
            return;
        }

        if (EditorApplication.timeSinceStartup - _waitStarted > 120d)
        {
            EditorApplication.update -= AwaitPackageRegistration;
            Debug.LogError(""Aexis UnityPackage registration timed out."");
        }
    }

    private static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ""..""));

    private static void CleanupBootstrap()
    {
        EditorApplication.delayCall += () =>
        {
            if (AssetDatabase.IsValidFolder(BootstrapAssetPath))
                AssetDatabase.DeleteAsset(BootstrapAssetPath);
        };
    }
}
#endif
";
        }

        private sealed class UnityPackageEntry
        {
            public UnityPackageEntry(string pathName, string guid, string assetPath, string metaPath, byte[] assetBytes, byte[] metaBytes)
            {
                PathName = pathName;
                Guid = guid;
                AssetPath = assetPath;
                MetaPath = metaPath;
                AssetBytes = assetBytes;
                MetaBytes = metaBytes;
            }

            public string PathName { get; }
            public string Guid { get; }
            public string AssetPath { get; }
            public string MetaPath { get; }
            public byte[] AssetBytes { get; }
            public byte[] MetaBytes { get; }
        }

        private sealed class UnityPackageTarWriter : IDisposable
        {
            private const int TarBlockSize = 512;
            private readonly Stream _stream;
            private bool _finished;

            public UnityPackageTarWriter(Stream stream)
            {
                _stream = stream ?? throw new ArgumentNullException(nameof(stream));
            }

            public void Write(UnityPackageEntry entry)
            {
                WriteDirectory(entry.Guid);
                if (entry.AssetBytes != null)
                    WriteFile(entry.Guid + "/asset", entry.AssetBytes);
                else if (!string.IsNullOrEmpty(entry.AssetPath))
                    WriteFile(entry.Guid + "/asset", entry.AssetPath);
                if (entry.MetaBytes != null)
                    WriteFile(entry.Guid + "/asset.meta", entry.MetaBytes);
                else if (!string.IsNullOrEmpty(entry.MetaPath))
                    WriteFile(entry.Guid + "/asset.meta", entry.MetaPath);
                WriteFile(entry.Guid + "/pathname", Encoding.UTF8.GetBytes(entry.PathName));
            }

            public void Dispose()
            {
                if (_finished)
                    return;

                _stream.Write(new byte[TarBlockSize * 2], 0, TarBlockSize * 2);
                _finished = true;
            }

            private void WriteDirectory(string path)
            {
                WriteHeader(path + "/", 0, (byte)'5');
            }

            private void WriteFile(string path, byte[] content)
            {
                WriteHeader(path, content.LongLength, (byte)'0');
                _stream.Write(content, 0, content.Length);
                WritePadding(content.LongLength);
            }

            private void WriteFile(string path, string sourcePath)
            {
                var length = new FileInfo(sourcePath).Length;
                WriteHeader(path, length, (byte)'0');
                using (var input = File.OpenRead(sourcePath))
                    input.CopyTo(_stream);
                WritePadding(length);
            }

            private void WriteHeader(string path, long length, byte typeFlag)
            {
                var header = new byte[TarBlockSize];
                WriteAscii(header, 0, 100, path);
                WriteOctal(header, 100, 8, typeFlag == (byte)'5' ? 493 : 420);
                WriteOctal(header, 108, 8, 0);
                WriteOctal(header, 116, 8, 0);
                WriteOctal(header, 124, 12, length);
                WriteOctal(header, 136, 12, 0);
                for (var index = 148; index < 156; index++)
                    header[index] = 0x20;
                header[156] = typeFlag;
                WriteAscii(header, 257, 6, "ustar");
                header[262] = 0;
                WriteAscii(header, 263, 2, "00");
                WriteAscii(header, 265, 32, "Aexis");
                WriteAscii(header, 297, 32, "Aexis");
                var checksum = 0;
                foreach (var value in header)
                    checksum += value;
                WriteOctal(header, 148, 8, checksum);
                _stream.Write(header, 0, header.Length);
            }

            private void WritePadding(long length)
            {
                var padding = (int)((TarBlockSize - length % TarBlockSize) % TarBlockSize);
                if (padding > 0)
                    _stream.Write(new byte[padding], 0, padding);
            }

            private static void WriteAscii(byte[] destination, int offset, int length, string value)
            {
                var bytes = Encoding.ASCII.GetBytes(value ?? string.Empty);
                if (bytes.Length > length)
                    throw new InvalidOperationException("Tar header field is too long: " + value);
                Buffer.BlockCopy(bytes, 0, destination, offset, bytes.Length);
            }

            private static void WriteOctal(byte[] destination, int offset, int length, long value)
            {
                var octal = Convert.ToString(value, 8);
                if (octal.Length > length - 2)
                    throw new InvalidOperationException("Tar numeric field is too large: " + value);
                var start = offset + length - 2 - octal.Length;
                for (var index = offset; index < offset + length - 2; index++)
                    destination[index] = (byte)'0';
                for (var index = 0; index < octal.Length; index++)
                    destination[start + index] = (byte)octal[index];
                destination[offset + length - 2] = 0;
                destination[offset + length - 1] = (byte)' ';
            }
        }

        [Serializable]
        private sealed class PackageMetadata
        {
            public string version;
        }
    }
}
#endif
