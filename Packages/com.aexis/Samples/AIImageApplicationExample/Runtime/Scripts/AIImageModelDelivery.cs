using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;
using Aexis.Samples;
using Aexis.Samples.Async;
using Aexis.Samples.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

public enum AIImageModelGroupId
{
    ClipMobileClipS0,
    CodeFormerDefault,
    RealEsrganX4PlusAnime,
    GfpganDefault,
    YoloV8PersonSegmentation,
    DeepFillV2Case1Onnx,
    Qwen35MobileQ4,
    Qwen35MobileQ8,
    DeepFillV2Case1Ncnn,
    DeepFillV2HiFill,
    Yolo11PersonSegmentation,
    RealEsrganOptionalModels,
    Matting,
    StableDiffusion,
    GfpganNativeExtras,
    Qwen35FullPrecision
}

public sealed class AIImageModelRemoteFile
{
    public AIImageModelRemoteFile(string destinationPath, string releaseTag, string assetName)
    {
        DestinationPath = AIImageModelDelivery.NormalizeRelativePath(destinationPath);
        ReleaseTag = releaseTag?.Trim() ?? string.Empty;
        AssetName = assetName?.Trim() ?? string.Empty;
    }

    public string DestinationPath { get; }
    public string ReleaseTag { get; }
    public string AssetName { get; }
}

public sealed class AIImageModelGroup
{
    public AIImageModelGroup(
        AIImageModelGroupId id,
        string displayName,
        bool bundledByDefault,
        string[] files,
        string[] prefixes = null,
        string flatReleaseTag = null,
        AIImageModelRemoteFile[] remoteFiles = null)
    {
        Id = id;
        DisplayName = displayName;
        BundledByDefault = bundledByDefault;
        Files = files ?? Array.Empty<string>();
        Prefixes = prefixes ?? Array.Empty<string>();
        RemoteFiles = remoteFiles
            ?? (string.IsNullOrWhiteSpace(flatReleaseTag)
                ? Array.Empty<AIImageModelRemoteFile>()
                : Files.Select(file => new AIImageModelRemoteFile(
                    file,
                    flatReleaseTag,
                    Path.GetFileName(file))).ToArray());
    }

    public AIImageModelGroupId Id { get; }
    public string DisplayName { get; }
    public bool BundledByDefault { get; }
    public IReadOnlyList<string> Files { get; }
    public IReadOnlyList<string> Prefixes { get; }
    public IReadOnlyList<AIImageModelRemoteFile> RemoteFiles { get; }
    public string ArchiveName => "AIImageModels." + Id + ".zip";

    public bool Covers(string relativePath)
    {
        var normalized = AIImageModelDelivery.NormalizeRelativePath(relativePath);
        for (var index = 0; index < Files.Count; index++)
        {
            if (string.Equals(Files[index], normalized, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        for (var index = 0; index < Prefixes.Count; index++)
        {
            if (normalized.StartsWith(Prefixes[index], StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }
}

public readonly struct AIImageModelDownloadProgress
{
    public AIImageModelDownloadProgress(AIImageModelGroup group, float progress01, string detail)
    {
        Group = group;
        Progress01 = Mathf.Clamp01(progress01);
        Detail = detail ?? string.Empty;
    }

    public AIImageModelGroup Group { get; }
    public float Progress01 { get; }
    public string Detail { get; }
}

public static class AIImageModelDelivery
{
    public const string PersistentDirectoryName = "AIImageModels";
    public const string ReleaseOwner = "newzjh";
    public const string ReleaseRepository = "AIImage";
    public const string DefaultReleaseTag = "model";
    public const string BundledManifestFileName = "AIImageModelDelivery.bundled.json";

    private const string ReleaseTagEnvironmentVariable = "AIIMAGE_MODEL_RELEASE_TAG";
    private const string ReleaseBaseUrlEnvironmentVariable = "AIIMAGE_MODEL_RELEASE_BASE_URL";
    private const string InferenceManifestDirectoryName = "InferenceManifests";

    private static readonly AIImageModelGroup[] Groups =
    {
        new AIImageModelGroup(
            AIImageModelGroupId.ClipMobileClipS0,
            "CLIP MobileCLIP S0",
            true,
            new[]
            {
                "Clip/bpe_simple_vocab_16e6.txt",
                "Clip/vocab.txt",
                "Clip/mobileclip_s0_export.label_embeddings.json",
                "Clip/mobileclip_s0_export/image_encoder.ncnn.param",
                "Clip/mobileclip_s0_export/image_encoder.ncnn.bin"
            }),
        new AIImageModelGroup(
            AIImageModelGroupId.CodeFormerDefault,
            "CodeFormer",
            true,
            new[]
            {
                "CodeFormer/models/encoder.param",
                "CodeFormer/models/encoder.bin",
                "CodeFormer/models/generator.param",
                "CodeFormer/models/generator.bin",
                "CodeFormer/models/yolov7-lite-e.param",
                "CodeFormer/models/yolov7-lite-e.bin"
            }),
        new AIImageModelGroup(
            AIImageModelGroupId.RealEsrganX4PlusAnime,
            "Real-ESRGAN AnimeVideo v3 x4",
            true,
            new[]
            {
                "RealESRGAN/models/realesr-animevideov3-x4.param",
                "RealESRGAN/models/realesr-animevideov3-x4.bin"
            },
            flatReleaseTag: "realesr"),
        new AIImageModelGroup(
            AIImageModelGroupId.GfpganDefault,
            "GFPGAN",
            true,
            new[]
            {
                "GFPGAN/models/encoder.param",
                "GFPGAN/models/encoder.bin",
                "GFPGAN/models/style.bin"
            },
            flatReleaseTag: "gfpgan"),
        new AIImageModelGroup(
            AIImageModelGroupId.YoloV8PersonSegmentation,
            "YOLOv8 person segmentation",
            true,
            new[]
            {
                "Yolo/yolov8n_seg.ncnn.param",
                "Yolo/yolov8n_seg.ncnn.bin"
            }),
        new AIImageModelGroup(
            AIImageModelGroupId.DeepFillV2Case1Onnx,
            "DeepFillV2 case1 (ONNX)",
            false,
            new[]
            {
                "DeepFileV2/deepfillv2_case1.source.onnx",
                "DeepFileV2/deepfillv2_case1.ncnn.param"
            },
            flatReleaseTag: "DeepFileV2"),
        new AIImageModelGroup(
            AIImageModelGroupId.Qwen35MobileQ4,
            "Qwen3.5 0.8B mobile Q4",
            true,
            new[]
            {
                "QWEN35/qwen3.5_0.8b_mobile_q4/model.json",
                "QWEN35/qwen3.5_0.8b_mobile_q4/vocab.txt",
                "QWEN35/qwen3.5_0.8b_mobile_q4/merges.txt",
                "QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_decoder.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_embed_token.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_proj_out.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_vision_embed_patch.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_vision_embed_pos.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_vision_encoder.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_mobile_q4.model.json",
                "QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_mobile_q4_projection.model.json",
                "QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_mobile_q4_assets.json",
                "QWEN35/qwen3.5_0.8b_mobile_q4/weights/qwen3.5_decoder.ncnn.bin.q4.part000",
                "QWEN35/qwen3.5_0.8b_mobile_q4/weights/qwen3.5_decoder.ncnn.bin.q4.part001",
                "QWEN35/qwen3.5_0.8b_mobile_q4/weights/qwen3.5_embed_token.ncnn.bin.q4.part000",
                "QWEN35/qwen3.5_0.8b_mobile_q4/weights/qwen3.5_vision_embed_patch.ncnn.bin.q4.part000",
                "QWEN35/qwen3.5_0.8b_mobile_q4/weights/qwen3.5_vision_embed_pos.ncnn.bin.q4.part000",
                "QWEN35/qwen3.5_0.8b_mobile_q4/weights/qwen3.5_vision_encoder.ncnn.bin.q4.part000"
            },
            flatReleaseTag: "qwen3.5_0.8b_mobile_q4"),
        new AIImageModelGroup(
            AIImageModelGroupId.Qwen35MobileQ8,
            "Qwen3.5 0.8B mobile Q8",
            false,
            new[]
            {
                "QWEN35/qwen3.5_0.8b_mobile_q8/model.json",
                "QWEN35/qwen3.5_0.8b_mobile_q8/vocab.txt",
                "QWEN35/qwen3.5_0.8b_mobile_q8/merges.txt",
                "QWEN35/qwen3.5_0.8b_mobile_q8/qwen3.5_decoder.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q8/qwen3.5_embed_token.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q8/qwen3.5_proj_out.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q8/qwen3.5_vision_embed_patch.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q8/qwen3.5_vision_embed_pos.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q8/qwen3.5_vision_encoder.ncnn.param",
                "QWEN35/qwen3.5_0.8b_mobile_q8/qwen3.5_mobile_q8.model.json",
                "QWEN35/qwen3.5_0.8b_mobile_q8/qwen3.5_mobile_q8_assets.json",
                "QWEN35/qwen3.5_0.8b_mobile_q8/weights/qwen3.5_decoder.ncnn.bin.q8.part000",
                "QWEN35/qwen3.5_0.8b_mobile_q8/weights/qwen3.5_decoder.ncnn.bin.q8.part001",
                "QWEN35/qwen3.5_0.8b_mobile_q8/weights/qwen3.5_embed_token.ncnn.bin.q8.part000",
                "QWEN35/qwen3.5_0.8b_mobile_q8/weights/qwen3.5_vision_embed_patch.ncnn.bin.q8.part000",
                "QWEN35/qwen3.5_0.8b_mobile_q8/weights/qwen3.5_vision_embed_pos.ncnn.bin.q8.part000",
                "QWEN35/qwen3.5_0.8b_mobile_q8/weights/qwen3.5_vision_encoder.ncnn.bin.q8.part000"
            },
            flatReleaseTag: "qwen3.5_0.8b_mobile_q8"),
        new AIImageModelGroup(
            AIImageModelGroupId.DeepFillV2Case1Ncnn,
            "DeepFillV2 case1 (NCNN)",
            true,
            new[]
            {
                "DeepFileV2/deepfillv2_case1.ncnn.param",
                "DeepFileV2/deepfillv2_case1.ncnn.bin"
            },
            flatReleaseTag: "DeepFileV2"),
        new AIImageModelGroup(
            AIImageModelGroupId.DeepFillV2HiFill,
            "DeepFillV2 HiFill",
            false,
            new[]
            {
                "DeepFileV2/deepfillv2_hifill.source.onnx",
                "DeepFileV2/deepfillv2_hifill.ncnn.param",
                "DeepFileV2/deepfillv2_hifill.ncnn.bin"
            },
            flatReleaseTag: "DeepFileV2"),
        new AIImageModelGroup(
            AIImageModelGroupId.Yolo11PersonSegmentation,
            "YOLO11 person segmentation",
            false,
            new[]
            {
                "Yolo/yolo11n_seg.ncnn.param",
                "Yolo/yolo11n_seg.ncnn.bin"
            }),
        new AIImageModelGroup(
            AIImageModelGroupId.RealEsrganOptionalModels,
            "Real-ESRGAN alternate models",
            false,
            new[]
            {
                "RealESRGAN/models/realesrgan-x4plus.param",
                "RealESRGAN/models/realesrgan-x4plus.bin",
                "RealESRGAN/models/realesr-animevideov3-x2.param",
                "RealESRGAN/models/realesr-animevideov3-x2.bin",
                "RealESRGAN/models/realesr-animevideov3-x3.param",
                "RealESRGAN/models/realesr-animevideov3-x3.bin"
            },
            flatReleaseTag: "realesr"),
        new AIImageModelGroup(
            AIImageModelGroupId.Matting,
            "Matting",
            true,
            new[] { "Matting/matting.param", "Matting/matting.bin" }),
        new AIImageModelGroup(
            AIImageModelGroupId.StableDiffusion,
            "SD 1.5 inpainting",
            false,
            new[]
            {
                "StableDiffusion/FrozenCLIPEmbedder-fp16.param",
                "StableDiffusion/FrozenCLIPEmbedder-fp16.bin",
                "StableDiffusion/AutoencoderKL-512-512-fp16-opt.param",
                "StableDiffusion/AutoencoderKL-fp16.bin",
                "StableDiffusion/AutoencoderKL-encoder-512-512-fp16.param",
                "StableDiffusion/AutoencoderKL-encoder-512-512-fp16.bin",
                "sdinpainting/unet.param",
                "sdinpainting/unet.bin"
            }),
        new AIImageModelGroup(
            AIImageModelGroupId.GfpganNativeExtras,
            "GFPGAN native extras",
            false,
            new[]
            {
                "GFPGAN/models/real_esrgan.param",
                "GFPGAN/models/real_esrgan.bin",
                "GFPGAN/models/yolov5-blazeface.param",
                "GFPGAN/models/yolov5-blazeface.bin"
            },
            flatReleaseTag: "gfpgan"),
        new AIImageModelGroup(
            AIImageModelGroupId.Qwen35FullPrecision,
            "Qwen3.5 0.8B full precision",
            false,
            new[] { "QWEN35/qwen3.5_0.8b/model.json" },
            new[] { "QWEN35/qwen3.5_0.8b/" })
    };

    public static IReadOnlyList<AIImageModelGroup> AllGroups => Groups;
    public static string PersistentRoot => Path.Combine(Application.persistentDataPath, PersistentDirectoryName);

    public static IEnumerable<AIImageModelGroup> DefaultGroups => Groups.Where(group => group.BundledByDefault);

    public static AIImageModelGroup GetGroup(AIImageModelGroupId id)
    {
        for (var index = 0; index < Groups.Length; index++)
        {
            if (Groups[index].Id == id)
                return Groups[index];
        }

        throw new ArgumentOutOfRangeException(nameof(id), id, "Unknown AIImage model group.");
    }

    public static AIImageModelGroup FindGroupForPath(string relativePath)
    {
        for (var index = 0; index < Groups.Length; index++)
        {
            if (Groups[index].Covers(relativePath))
                return Groups[index];
        }

        return null;
    }

    public static IEnumerable<AIImageModelGroup> FindGroupsForPaths(IEnumerable<string> relativePaths)
    {
        var unique = new HashSet<AIImageModelGroupId>();
        if (relativePaths == null)
            yield break;

        foreach (var path in relativePaths)
        {
            var group = FindGroupForPath(path);
            if (group != null && unique.Add(group.Id))
                yield return group;
        }
    }

    public static string GetPersistentPath(string relativePath)
    {
        var normalized = NormalizeRelativePath(relativePath);
        if (string.IsNullOrWhiteSpace(normalized))
            throw new ArgumentException("A model relative path is required.", nameof(relativePath));

        var root = Path.GetFullPath(PersistentRoot);
        var path = Path.GetFullPath(Path.Combine(root, normalized));
        var rootWithSeparator = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;
        if (!path.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Model path escapes the persistent model directory: " + relativePath);
        return path;
    }

    public static bool IsInstalled(AIImageModelGroup group)
    {
        if (group == null)
            return false;

        var files = GetInstalledRequiredFiles(group);
        for (var index = 0; index < files.Count; index++)
        {
            if (!File.Exists(GetPersistentPath(files[index])))
                return false;
        }

        return files.Count > 0;
    }

    public static bool IsAvailableLocally(AIImageModelGroup group)
    {
        if (IsInstalled(group))
            return true;

        if (group == null || Application.platform == RuntimePlatform.Android)
            return false;

        for (var index = 0; index < group.Files.Count; index++)
        {
            if (!AexisSampleStreamingAssets.TryResolveFilePath(group.Files[index], out var path)
                || !File.Exists(path))
                return false;
        }

        return group.Files.Count > 0;
    }

    public static async UniTask<bool> IsAvailableAsync(AIImageModelGroup group, CancellationToken cancellationToken = default)
    {
        if (IsAvailableLocally(group))
            return true;

        if (Application.platform != RuntimePlatform.Android)
            return false;

        var bundled = await LoadAndroidBundledGroupIdsAsync(cancellationToken);
        return bundled.Contains(group.Id.ToString());
    }

    public static async UniTask MaterializeBundledGroupAsync(
        AIImageModelGroup group,
        Action<AIImageModelDownloadProgress> onProgress,
        CancellationToken cancellationToken = default)
    {
        if (group == null)
            throw new ArgumentNullException(nameof(group));
        if (Application.platform != RuntimePlatform.Android)
            return;
        if (IsInstalled(group))
        {
            await MaterializeBundledInferenceManifestsAsync(group, onProgress, cancellationToken);
            return;
        }

        var bundled = await LoadAndroidBundledGroupIdsAsync(cancellationToken);
        if (!bundled.Contains(group.Id.ToString()))
            throw new FileNotFoundException("The requested model group is not bundled in this Android player.", group.DisplayName);

        // Copy the static manifest first. Qwen's physical shard list is encoded
        // inside it, so a fixed part000/part001 list cannot be release-safe.
        await CopyBundledFilesAsync(group, group.Files, onProgress, cancellationToken);
        await CopyBundledFilesAsync(group, GetInstalledRequiredFiles(group), onProgress, cancellationToken);

        await MaterializeBundledInferenceManifestsAsync(group, onProgress, cancellationToken);

        if (!IsInstalled(group))
            throw new InvalidDataException("Bundled Android model files are incomplete for " + group.DisplayName + ".");
        onProgress?.Invoke(new AIImageModelDownloadProgress(group, 1f, "Bundled model files ready"));
    }

    private static async UniTask CopyBundledFilesAsync(
        AIImageModelGroup group,
        IReadOnlyList<string> files,
        Action<AIImageModelDownloadProgress> onProgress,
        CancellationToken cancellationToken)
    {
        if (files == null)
            return;

        for (var index = 0; index < files.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var relative = files[index];
            var destination = GetPersistentPath(relative);
            if (File.Exists(destination))
                continue;

            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            var staging = destination + ".part";
            Cleanup(staging);
            try
            {
                var source = Application.streamingAssetsPath.TrimEnd('/', '\\') + "/" + NormalizeRelativePath(relative);
                using (var request = new UnityWebRequest(source, UnityWebRequest.kHttpVerbGET))
                {
                    request.downloadHandler = new DownloadHandlerFile(staging, false);
                    var operation = request.SendWebRequest();
                    while (!operation.isDone)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var completed = (index + request.downloadProgress) / Mathf.Max(1, files.Count);
                        onProgress?.Invoke(new AIImageModelDownloadProgress(group, completed, "Preparing bundled model files"));
                        await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                    }

                    cancellationToken.ThrowIfCancellationRequested();
                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        throw new IOException(
                            "Failed to read bundled Android model file '" + relative + "': " + request.error);
                    }
                }

                File.Move(staging, destination);
            }
            finally
            {
                Cleanup(staging);
            }
        }
    }

    private static IReadOnlyList<string> GetInstalledRequiredFiles(AIImageModelGroup group)
    {
        var files = new List<string>(group?.Files ?? Array.Empty<string>());
        if (group?.Id == AIImageModelGroupId.Qwen35MobileQ4)
        {
            var manifest = GetPersistentPath("QWEN35/qwen3.5_0.8b_mobile_q4/qwen3.5_mobile_q4_assets.json");
            AppendQwen35ShardFiles(manifest, files);
        }
        return files
            .Select(NormalizeRelativePath)
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static void AppendQwen35ShardFiles(string manifestPath, ICollection<string> files)
    {
        if (files == null || string.IsNullOrWhiteSpace(manifestPath) || !File.Exists(manifestPath))
            return;

        var document = JObject.Parse(File.ReadAllText(manifestPath));
        var logicalFiles = document["logical_files"] as JObject;
        if (logicalFiles == null)
            throw new InvalidDataException("Qwen3.5 mobile asset manifest has no logical_files object.");

        const string prefix = "QWEN35/qwen3.5_0.8b_mobile_q4/";
        foreach (var logical in logicalFiles.Properties())
        {
            var parts = logical.Value["parts"] as JArray;
            if (parts == null)
                throw new InvalidDataException("Qwen3.5 mobile logical asset has no parts array: " + logical.Name);
            foreach (var part in parts)
            {
                var path = NormalizeRelativePath((string)part["file"]);
                if (string.IsNullOrWhiteSpace(path))
                    throw new InvalidDataException("Qwen3.5 mobile shard path is empty: " + logical.Name);
                files.Add(prefix + path);
            }
        }
    }

    private static async UniTask MaterializeBundledInferenceManifestsAsync(
        AIImageModelGroup group,
        Action<AIImageModelDownloadProgress> onProgress,
        CancellationToken cancellationToken)
    {
        var manifestFileNames = GetBundledInferenceManifestFileNames(group.Id);
        for (var index = 0; index < manifestFileNames.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var relative = InferenceManifestDirectoryName + "/" + manifestFileNames[index];
            var destination = GetPersistentPath(relative);
            if (File.Exists(destination))
                continue;

            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            var staging = destination + ".part";
            Cleanup(staging);
            try
            {
                var source = Application.streamingAssetsPath.TrimEnd('/', '\\') + "/" + NormalizeRelativePath(relative);
                using (var request = new UnityWebRequest(source, UnityWebRequest.kHttpVerbGET))
                {
                    request.downloadHandler = new DownloadHandlerFile(staging, false);
                    var operation = request.SendWebRequest();
                    while (!operation.isDone)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var completed = (index + request.downloadProgress) / Mathf.Max(1, manifestFileNames.Length);
                        onProgress?.Invoke(new AIImageModelDownloadProgress(
                            group,
                            Mathf.Lerp(0.94f, 0.99f, completed),
                            "Preparing inference contract"));
                        await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                    }

                    cancellationToken.ThrowIfCancellationRequested();
                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        throw new IOException(
                            "Failed to read bundled inference manifest '" + relative + "': " + request.error);
                    }
                }

                File.Move(staging, destination);
            }
            finally
            {
                Cleanup(staging);
            }
        }
    }

    private static string[] GetBundledInferenceManifestFileNames(AIImageModelGroupId groupId)
    {
        switch (groupId)
        {
            case AIImageModelGroupId.ClipMobileClipS0:
                return new[] { "clip-mobileclip-s0.fp16.model.json" };
            case AIImageModelGroupId.RealEsrganX4PlusAnime:
            case AIImageModelGroupId.RealEsrganOptionalModels:
                return new[] { "esrgan-realesrgan-x4plus.fp16.model.json" };
            case AIImageModelGroupId.CodeFormerDefault:
                return new[]
                {
                    "codeformer.fp16.model.json",
                    "codeformer.fp32.model.json"
                };
            case AIImageModelGroupId.GfpganDefault:
                return new[]
                {
                    "gfpgan.fp16.model.json",
                    "gfpgan.fp32.model.json"
                };
            case AIImageModelGroupId.YoloV8PersonSegmentation:
            case AIImageModelGroupId.Yolo11PersonSegmentation:
                return new[] { "yolo-seg.fp16.model.json" };
            case AIImageModelGroupId.Matting:
                return new[]
                {
                    "matting.fp16.model.json",
                    "matting.fp32.model.json"
                };
            case AIImageModelGroupId.StableDiffusion:
                return new[] { "sd-inpainting.fp16.model.json" };
            default:
                return Array.Empty<string>();
        }
    }

    public static async UniTask DownloadGroupAsync(
        AIImageModelGroup group,
        Action<AIImageModelDownloadProgress> onProgress,
        CancellationToken cancellationToken = default)
    {
        if (group == null)
            throw new ArgumentNullException(nameof(group));
        if (IsInstalled(group))
        {
            if (Application.platform == RuntimePlatform.Android)
                await MaterializeBundledInferenceManifestsAsync(group, onProgress, cancellationToken);
            onProgress?.Invoke(new AIImageModelDownloadProgress(group, 1f, "Already installed"));
            return;
        }

        var stagingRoot = Path.Combine(PersistentRoot, ".staging", group.Id.ToString());
        Cleanup(stagingRoot);
        Directory.CreateDirectory(Path.GetDirectoryName(stagingRoot));

        try
        {
            if (group.RemoteFiles.Count > 0)
            {
                await DownloadRemoteFilesAsync(group, stagingRoot, onProgress, cancellationToken);
                cancellationToken.ThrowIfCancellationRequested();
                onProgress?.Invoke(new AIImageModelDownloadProgress(group, 0.97f, "Installing"));
                CommitExtractedFiles(group, stagingRoot);
            }
            else
            {
                await DownloadArchiveAsync(group, stagingRoot, onProgress, cancellationToken);
            }

            if (!IsInstalled(group))
                throw new InvalidDataException("Downloaded files do not contain every required file for " + group.DisplayName + ".");
            if (Application.platform == RuntimePlatform.Android)
                await MaterializeBundledInferenceManifestsAsync(group, onProgress, cancellationToken);
            onProgress?.Invoke(new AIImageModelDownloadProgress(group, 1f, "Installed"));
        }
        finally
        {
            Cleanup(stagingRoot);
        }
    }

    public static string GetReleaseAssetUrl(string assetName, string releaseTag = null)
    {
        if (string.IsNullOrWhiteSpace(assetName))
            throw new ArgumentException("A release asset name is required.", nameof(assetName));

        var tag = string.IsNullOrWhiteSpace(releaseTag)
            ? Environment.GetEnvironmentVariable(ReleaseTagEnvironmentVariable)
            : releaseTag;
        if (string.IsNullOrWhiteSpace(tag))
            tag = DefaultReleaseTag;

        var configuredBaseUrl = Environment.GetEnvironmentVariable(ReleaseBaseUrlEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(configuredBaseUrl))
        {
            var baseUrl = configuredBaseUrl.TrimEnd('/');
            if (baseUrl.IndexOf("{tag}", StringComparison.OrdinalIgnoreCase) >= 0)
                baseUrl = baseUrl.Replace("{tag}", Uri.EscapeDataString(tag));
            return baseUrl + "/" + Uri.EscapeDataString(assetName);
        }

        return "https://github.com/" + ReleaseOwner + "/" + ReleaseRepository + "/releases/download/"
            + Uri.EscapeDataString(tag) + "/" + Uri.EscapeDataString(assetName);
    }

    private static async UniTask DownloadRemoteFilesAsync(
        AIImageModelGroup group,
        string stagingRoot,
        Action<AIImageModelDownloadProgress> onProgress,
        CancellationToken cancellationToken)
    {
        var files = ResolveRemoteFiles(group);
        for (var index = 0; index < files.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var remote = files[index];
            var stagingPath = ResolveContainedPath(stagingRoot, remote.DestinationPath);
            Directory.CreateDirectory(Path.GetDirectoryName(stagingPath));
            var url = GetReleaseAssetUrl(remote.AssetName, remote.ReleaseTag);
            using (var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbGET))
            {
                request.downloadHandler = new DownloadHandlerFile(stagingPath, false);
                request.disposeDownloadHandlerOnDispose = true;
                var operation = request.SendWebRequest();
                while (!operation.isDone)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var completed = (index + request.downloadProgress) / Mathf.Max(1, files.Count);
                    onProgress?.Invoke(new AIImageModelDownloadProgress(group, completed * 0.95f,
                        "Downloading " + remote.AssetName));
                    await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                }

                cancellationToken.ThrowIfCancellationRequested();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    throw new IOException("Failed to download " + group.DisplayName + " asset '"
                        + remote.AssetName + "' from release '" + remote.ReleaseTag + "': "
                        + request.error + " (" + url + ")");
                }
            }
        }
    }

    private static async UniTask DownloadArchiveAsync(
        AIImageModelGroup group,
        string stagingRoot,
        Action<AIImageModelDownloadProgress> onProgress,
        CancellationToken cancellationToken)
    {
        var archivePath = stagingRoot + ".zip";
        var extractRoot = stagingRoot + ".extract";
        Cleanup(archivePath);
        Cleanup(extractRoot);
        try
        {
            var url = GetReleaseAssetUrl(group.ArchiveName);
            onProgress?.Invoke(new AIImageModelDownloadProgress(group, 0f, "Downloading"));
            using (var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbGET))
            {
                request.downloadHandler = new DownloadHandlerFile(archivePath, false);
                request.disposeDownloadHandlerOnDispose = true;
                var operation = request.SendWebRequest();
                while (!operation.isDone)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    onProgress?.Invoke(new AIImageModelDownloadProgress(group, request.downloadProgress * 0.9f, "Downloading"));
                    await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                }

                cancellationToken.ThrowIfCancellationRequested();
                if (request.result != UnityWebRequest.Result.Success)
                    throw new IOException("Failed to download " + group.DisplayName + ": " + request.error + " (" + url + ")");
            }

            onProgress?.Invoke(new AIImageModelDownloadProgress(group, 0.91f, "Extracting"));
            await UniTask.RunOnThreadPool(() => ExtractArchive(group, archivePath, extractRoot, cancellationToken), cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            CommitExtractedFiles(group, extractRoot);
        }
        finally
        {
            Cleanup(archivePath);
            Cleanup(extractRoot);
        }
    }

    private static IReadOnlyList<AIImageModelRemoteFile> ResolveRemoteFiles(AIImageModelGroup group)
    {
        if (group.Prefixes.Count > 0)
            throw new InvalidOperationException("Direct release-file downloads require an explicit file list: " + group.DisplayName);

        var mapped = new Dictionary<string, AIImageModelRemoteFile>(StringComparer.OrdinalIgnoreCase);
        for (var index = 0; index < group.RemoteFiles.Count; index++)
        {
            var remote = group.RemoteFiles[index];
            if (string.IsNullOrWhiteSpace(remote.DestinationPath)
                || string.IsNullOrWhiteSpace(remote.ReleaseTag)
                || string.IsNullOrWhiteSpace(remote.AssetName))
                throw new InvalidDataException("Invalid remote release mapping for " + group.DisplayName + ".");
            if (mapped.ContainsKey(remote.DestinationPath))
                throw new InvalidDataException("Duplicate remote release mapping for " + remote.DestinationPath + ".");
            mapped.Add(remote.DestinationPath, remote);
        }

        var result = new List<AIImageModelRemoteFile>(group.Files.Count);
        for (var index = 0; index < group.Files.Count; index++)
        {
            var relative = group.Files[index];
            if (!mapped.TryGetValue(relative, out var remote))
                throw new InvalidDataException("No remote release mapping exists for " + relative + " in " + group.DisplayName + ".");
            result.Add(remote);
        }

        if (mapped.Count != result.Count)
            throw new InvalidDataException("Remote release mapping contains an unexpected file for " + group.DisplayName + ".");
        return result;
    }

    public static string NormalizeRelativePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return string.Empty;
        return path.Trim().TrimStart('/', '\\').Replace('\\', '/');
    }

    private static async UniTask<HashSet<string>> LoadAndroidBundledGroupIdsAsync(CancellationToken cancellationToken)
    {
        var result = new HashSet<string>(StringComparer.Ordinal);
        try
        {
            var text = await AexisSampleStreamingAssets.ReadTextAsync(BundledManifestFileName, cancellationToken);
            var values = JsonUtility.FromJson<BundledGroupManifest>(text)?.groups;
            if (values == null)
                return result;
            for (var index = 0; index < values.Length; index++)
            {
                if (!string.IsNullOrWhiteSpace(values[index]))
                    result.Add(values[index]);
            }
        }
        catch
        {
            // A player built before reduced delivery support has no manifest.
        }
        return result;
    }

    private static void ExtractArchive(
        AIImageModelGroup group,
        string archivePath,
        string extractRoot,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(extractRoot);
        using (var archive = ZipFile.OpenRead(archivePath))
        {
            foreach (var entry in archive.Entries)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (string.IsNullOrEmpty(entry.Name))
                    continue;
                var relative = NormalizeRelativePath(entry.FullName);
                if (!group.Covers(relative))
                    throw new InvalidDataException("Unexpected file in " + group.ArchiveName + ": " + relative);
                var destination = ResolveContainedPath(extractRoot, relative);
                Directory.CreateDirectory(Path.GetDirectoryName(destination));
                entry.ExtractToFile(destination, true);
            }
        }
    }

    private static void CommitExtractedFiles(AIImageModelGroup group, string extractRoot)
    {
        for (var index = 0; index < group.Files.Count; index++)
        {
            var relative = group.Files[index];
            var source = ResolveContainedPath(extractRoot, relative);
            if (!File.Exists(source))
                continue;
            var destination = GetPersistentPath(relative);
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            File.Copy(source, destination, true);
        }

        foreach (var prefix in group.Prefixes)
        {
            var prefixRoot = ResolveContainedPath(extractRoot, prefix);
            if (!Directory.Exists(prefixRoot))
                continue;
            foreach (var source in Directory.GetFiles(prefixRoot, "*", SearchOption.AllDirectories))
            {
                var relative = NormalizeRelativePath(source.Substring(extractRoot.Length));
                var destination = GetPersistentPath(relative);
                Directory.CreateDirectory(Path.GetDirectoryName(destination));
                File.Copy(source, destination, true);
            }
        }
    }

    private static string ResolveContainedPath(string root, string relative)
    {
        var fullRoot = Path.GetFullPath(root);
        var full = Path.GetFullPath(Path.Combine(fullRoot, NormalizeRelativePath(relative)));
        var prefix = fullRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;
        if (!full.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Archive entry escapes its extraction directory: " + relative);
        return full;
    }

    private static void Cleanup(string path)
    {
        try
        {
            if (File.Exists(path)) File.Delete(path);
            if (Directory.Exists(path)) Directory.Delete(path, true);
        }
        catch
        {
            // Cleanup must not hide a download or validation error.
        }
    }

    [Serializable]
    private sealed class BundledGroupManifest
    {
        public string[] groups;
    }
}
