using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using NcnnCompute;
using Newtonsoft.Json.Linq;
using UnityEditor;
using AIImage.Qwen35;
using UnityEngine;

public static class Qwen35MobileQ8AssetBuilder
{
    private const string SourceModelVariable = "AIIMAGE_QWEN35_SOURCE_MODEL";
    private const string OutputModelVariable = "AIIMAGE_QWEN35_Q8_OUTPUT";
    private const string ShardBytesVariable = "AIIMAGE_QWEN35_Q8_SHARD_BYTES";
    private const string RebuildLogicalFilesVariable = "AIIMAGE_QWEN35_Q8_REBUILD_LOGICAL_FILES";
    private const int DefaultShardBytes = 256 * 1024 * 1024;

    private static readonly (string Param, string Bin)[] QuantizedNetworks =
    {
        ("qwen3.5_decoder.ncnn.param", "qwen3.5_decoder.ncnn.bin"),
        ("qwen3.5_vision_embed_patch.ncnn.param", "qwen3.5_vision_embed_patch.ncnn.bin"),
        ("qwen3.5_vision_embed_pos.ncnn.param", "qwen3.5_vision_embed_pos.ncnn.bin"),
        ("qwen3.5_vision_encoder.ncnn.param", "qwen3.5_vision_encoder.ncnn.bin")
    };

    private static readonly string[] DeliveryFiles =
    {
        "model.json",
        "vocab.txt",
        "merges.txt",
        "qwen3.5_decoder.ncnn.param",
        "qwen3.5_embed_token.ncnn.param",
        "qwen3.5_proj_out.ncnn.param",
        "qwen3.5_vision_embed_patch.ncnn.param",
        "qwen3.5_vision_embed_pos.ncnn.param",
        "qwen3.5_vision_encoder.ncnn.param"
    };

    [MenuItem("AIImage/Qwen3.5/Build Mobile Q8 Assets")]
    public static void BuildInteractive()
    {
        Build(
            Environment.GetEnvironmentVariable(SourceModelVariable),
            Environment.GetEnvironmentVariable(OutputModelVariable),
            ResolveShardBytes());
    }

    public static void BuildBatch()
    {
        try
        {
            Build(
                Environment.GetEnvironmentVariable(SourceModelVariable),
                Environment.GetEnvironmentVariable(OutputModelVariable),
                ResolveShardBytes());
        }
        catch (Exception error)
        {
            UnityEngine.Debug.LogError("QWEN35_Q8_BUILD_FAILED\n" + error);
            throw;
        }
    }

    [MenuItem("AIImage/Qwen3.5/Build Mobile INT4 GPU Manifest")]
    public static void BuildInt4GpuManifestInteractive()
    {
        BuildInt4GpuManifest(Environment.GetEnvironmentVariable(OutputModelVariable));
    }

    public static void BuildInt4GpuManifestBatch()
    {
        try
        {
            BuildInt4GpuManifest(Environment.GetEnvironmentVariable(OutputModelVariable));
        }
        catch (Exception error)
        {
            UnityEngine.Debug.LogError("QWEN35_INT4_GPU_MANIFEST_BUILD_FAILED\n" + error);
            throw;
        }
    }

    private static void Build(string sourceModel, string outputModel, int shardBytes)
    {
        if (string.IsNullOrWhiteSpace(sourceModel)) throw new ArgumentException(SourceModelVariable + " is required.");
        if (string.IsNullOrWhiteSpace(outputModel)) throw new ArgumentException(OutputModelVariable + " is required.");
        var source = Path.GetFullPath(sourceModel);
        var output = Path.GetFullPath(outputModel);
        if (!Directory.Exists(source)) throw new DirectoryNotFoundException(source);
        if (string.Equals(source.TrimEnd('\\', '/'), output.TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Qwen3.5 Q8 output must differ from the FP32 source directory.");
        if (shardBytes < 16 * 1024 * 1024) throw new ArgumentOutOfRangeException(nameof(shardBytes), "Shard size must be at least 16 MiB.");

        var contract = Qwen35ModelContract.Validate(source, requireWeights: true);
        if (!contract.IsValid) throw new InvalidDataException("Qwen3.5 source contract failed: " + string.Join(" | ", contract.Errors));

        Directory.CreateDirectory(output);
        var weightsDirectory = Path.Combine(output, "weights");
        Directory.CreateDirectory(weightsDirectory);
        foreach (var file in DeliveryFiles)
            File.Copy(Path.Combine(source, file), Path.Combine(output, file), true);

        var precisionManifest = BuildPrecisionManifest();
        File.WriteAllText(Path.Combine(output, Qwen35MobileAssetSet.PrecisionManifestFileName), precisionManifest.ToString());
        File.WriteAllText(
            Path.Combine(output, Qwen35MobileAssetSet.Int4GpuPrecisionManifestFileName),
            BuildInt4GpuPrecisionManifest().ToString());
        var runtimeManifest = NcnnModelManifestLoader.LoadFromJson(precisionManifest.ToString(), "Qwen3.5 mobile Q8 builder");
        var logicalFiles = new JObject();
        var sourceFiles = new JObject();
        var buildWatch = Stopwatch.StartNew();
        var rebuild = ResolveRebuildSet();
        JObject existing = null;
        var existingManifestPath = Path.Combine(output, Qwen35MobileAssetSet.ManifestFileName);
        if (rebuild != null && File.Exists(existingManifestPath)) existing = JObject.Parse(File.ReadAllText(existingManifestPath));

        const string sharedLogicalName = "qwen3.5_embed_token.ncnn.bin";
        if (rebuild == null || rebuild.Contains(sharedLogicalName))
            ConvertSharedEmbedding(source, output, weightsDirectory, shardBytes, logicalFiles, sourceFiles);
        else
            CopyExistingEntry(existing, sharedLogicalName, logicalFiles, sourceFiles);
        foreach (var network in QuantizedNetworks)
        {
            if (rebuild == null || rebuild.Contains(network.Bin))
                ConvertNetwork(source, output, weightsDirectory, network.Param, network.Bin, shardBytes, runtimeManifest, logicalFiles, sourceFiles);
            else
                CopyExistingEntry(existing, network.Bin, logicalFiles, sourceFiles);
        }

        buildWatch.Stop();
        long storedBytes = 0;
        foreach (var property in logicalFiles.Properties()) storedBytes += (long)property.Value["stored_bytes"];
        var sourceBytes = 0L;
        foreach (var property in sourceFiles.Properties()) sourceBytes += (long)property.Value["bytes"];
        var manifest = new JObject
        {
            ["schema"] = Qwen35MobileAssetSet.ManifestSchema,
            ["model_id"] = "qwen3.5_0.8b",
            ["format"] = "aiimage-q8-block-symmetric/v1",
            ["weight_only"] = true,
            ["source_base_url"] = "https://mirrors.sdu.edu.cn/ncnn_modelzoo/",
            ["source_model_directory"] = source,
            ["source_bytes"] = sourceBytes,
            ["stored_weight_bytes"] = storedBytes,
            ["compression_ratio"] = sourceBytes > 0 ? storedBytes / (double)sourceBytes : 0.0,
            ["shard_limit_bytes"] = shardBytes,
            ["build_elapsed_seconds"] = buildWatch.Elapsed.TotalSeconds,
            ["source_files"] = sourceFiles,
            ["logical_files"] = logicalFiles
        };
        var manifestPath = Path.Combine(output, Qwen35MobileAssetSet.ManifestFileName);
        File.WriteAllText(manifestPath, manifest.ToString());

        var verified = Qwen35MobileAssetSet.TryLoad(output, verifyHashes: true);
        if (verified == null || verified.StoredWeightBytes != storedBytes || !verified.WeightOnly)
            throw new InvalidDataException("Generated Qwen3.5 mobile Q8 manifest failed self-validation.");
        var reportPath = Path.Combine(output, "qwen3.5_mobile_q8_build_report.json");
        File.WriteAllText(reportPath, new JObject
        {
            ["schema"] = "qwen35.mobile-q8-build-report/v1",
            ["valid"] = true,
            ["source_model"] = source,
            ["output_model"] = output,
            ["manifest"] = manifestPath,
            ["stored_weight_bytes"] = storedBytes,
            ["source_bytes"] = sourceBytes,
            ["elapsed_seconds"] = buildWatch.Elapsed.TotalSeconds
        }.ToString());
        UnityEngine.Debug.Log("QWEN35_Q8_BUILD_OK report=" + reportPath + " stored_bytes=" + storedBytes);
        AssetDatabase.Refresh();
    }

    private static void BuildInt4GpuManifest(string outputModel)
    {
        if (string.IsNullOrWhiteSpace(outputModel)) throw new ArgumentException(OutputModelVariable + " is required.");
        var output = Path.GetFullPath(outputModel);
        var mobile = Qwen35MobileAssetSet.TryLoad(output, verifyHashes: true);
        if (mobile == null || !mobile.WeightOnly)
            throw new InvalidDataException("A validated Qwen3.5 Q8 mobile asset set is required before creating the INT4 GPU manifest.");
        var path = Path.Combine(output, Qwen35MobileAssetSet.Int4GpuPrecisionManifestFileName);
        File.WriteAllText(path, BuildInt4GpuPrecisionManifest().ToString());
        UnityEngine.Debug.Log("QWEN35_INT4_GPU_MANIFEST_BUILD_OK path=" + path);
        AssetDatabase.Refresh();
    }

    private static void ConvertSharedEmbedding(
        string source,
        string output,
        string weightsDirectory,
        int shardBytes,
        JObject logicalFiles,
        JObject sourceFiles)
    {
        const string logicalName = "qwen3.5_embed_token.ncnn.bin";
        var sourcePath = Path.Combine(source, logicalName);
        var tempPath = Path.Combine(weightsDirectory, logicalName + ".q8.building");
        var sourceInfo = DescribeSource(sourcePath);
        using (var input = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan))
        using (var reader = new BinaryReader(input))
        using (var archive = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024, FileOptions.SequentialScan))
        using (var writer = new NcnnQ8ArchiveWriter(archive, input.Length, Qwen35DecoderSession.HiddenSize, 0))
        {
            var flag = reader.ReadUInt32();
            if (flag != 0u && flag != 0x0002C056u)
                throw new InvalidDataException("Shared token embedding must use raw FP32 ncnn storage. flag=0x" + flag.ToString("X8"));
            var rowCount = Qwen35SharedTokenEmbeddingWeights.ExpectedElementCount / Qwen35DecoderSession.HiddenSize;
            writer.WriteQ8FromFloat32Rows(reader, rowCount, Qwen35DecoderSession.HiddenSize);
            if (input.Position != input.Length)
                throw new InvalidDataException("Shared token embedding source was not consumed exactly. remaining=" + (input.Length - input.Position));
        }
        logicalFiles[logicalName] = FinalizeArchive(output, weightsDirectory, logicalName, tempPath, shardBytes, "row-wise-int8", Qwen35DecoderSession.HiddenSize);
        sourceFiles[logicalName] = sourceInfo;
    }

    private static void ConvertNetwork(
        string source,
        string output,
        string weightsDirectory,
        string paramName,
        string logicalName,
        int shardBytes,
        AIImage.Inference.Core.ModelManifest runtimeManifest,
        JObject logicalFiles,
        JObject sourceFiles)
    {
        var sourcePath = Path.Combine(source, logicalName);
        var tempPath = Path.Combine(weightsDirectory, logicalName + ".q8.building");
        var sourceInfo = DescribeSource(sourcePath);
        using (var input = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan))
        using (var archive = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024, FileOptions.SequentialScan))
        using (var writer = new NcnnQ8ArchiveWriter(archive, input.Length, defaultBlockSize: 256, fp32Threshold: 4096))
        using (var reader = new NcnnBinReader(input, writer))
        using (var ops = new NcnnOps())
        using (var repro = new NcnnRepro(ops))
        {
            repro.ApplyModelManifest(runtimeManifest);
            repro.LoadModel(File.ReadAllText(Path.Combine(source, paramName)), reader);
            if (input.Position != input.Length)
                throw new InvalidDataException(logicalName + " source was not consumed exactly. remaining=" + (input.Length - input.Position));
        }
        logicalFiles[logicalName] = FinalizeArchive(output, weightsDirectory, logicalName, tempPath, shardBytes, "block-int8-or-fp32-small-constant", 256);
        sourceFiles[logicalName] = sourceInfo;
    }

    private static JObject FinalizeArchive(
        string output,
        string weightsDirectory,
        string logicalName,
        string archivePath,
        int shardBytes,
        string encoding,
        int blockSize)
    {
        var parts = new JArray();
        var buffer = new byte[1024 * 1024];
        long storedBytes = 0;
        using (var source = new FileStream(archivePath, FileMode.Open, FileAccess.Read, FileShare.Read, buffer.Length, FileOptions.SequentialScan))
        {
            for (var partIndex = 0; source.Position < source.Length; partIndex++)
            {
                var fileName = logicalName + ".q8.part" + partIndex.ToString("D3");
                var partPath = Path.Combine(weightsDirectory, fileName);
                var remaining = Math.Min((long)shardBytes, source.Length - source.Position);
                using (var part = new FileStream(partPath, FileMode.Create, FileAccess.Write, FileShare.None, buffer.Length, FileOptions.SequentialScan))
                {
                    while (remaining > 0)
                    {
                        var requested = (int)Math.Min(buffer.Length, remaining);
                        var read = source.Read(buffer, 0, requested);
                        if (read <= 0) throw new EndOfStreamException("Unexpected end while splitting " + logicalName);
                        part.Write(buffer, 0, read);
                        remaining -= read;
                    }
                }
                var bytes = new FileInfo(partPath).Length;
                storedBytes += bytes;
                parts.Add(new JObject
                {
                    ["file"] = NormalizeRelativePath(Path.GetRelativePath(output, partPath)),
                    ["bytes"] = bytes,
                    ["sha256"] = ComputeSha256(partPath)
                });
            }
        }
        File.Delete(archivePath);
        return new JObject
        {
            ["encoding"] = encoding,
            ["block_size"] = blockSize,
            ["stored_bytes"] = storedBytes,
            ["parts"] = parts
        };
    }

    private static JObject DescribeSource(string path)
    {
        var info = new FileInfo(path);
        if (!info.Exists) throw new FileNotFoundException("Qwen3.5 source bin is missing.", path);
        return new JObject { ["bytes"] = info.Length, ["sha256"] = ComputeSha256(path) };
    }

    private static JObject BuildPrecisionManifest()
    {
        return new JObject
        {
            ["schemaVersion"] = "aiimage.model-manifest/v1",
            ["modelId"] = "qwen3.5_0.8b",
            ["precision"] = new JObject
            {
                ["activationDtype"] = "FP32",
                ["weightDtype"] = "INT8",
                ["sensitiveOutputDtype"] = "FP32",
                ["requireStrictTexturePlan"] = true
            },
            ["quantization"] = new JObject
            {
                ["quantizationVersion"] = "aiimage.qwen35-mobile-q8/v1",
                ["calibrationVersion"] = "qwen35-sdu-fp32-weight-absmax-v1",
                ["calibrationMethod"] = "symmetric-weight-absmax-per-output-channel",
                ["weightScheme"] = "INT8_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC",
                ["outputChannelAxis"] = 0,
                ["symmetric"] = true,
                ["zeroPoint"] = 0,
                ["accumulationDtype"] = "FP32",
                ["activationQuantized"] = false,
                ["quantizedOperators"] = new JArray("Gemm", "Convolution", "InnerProduct"),
                ["nodePlans"] = new JArray(),
                ["unquantizedWeightDtype"] = "FP32"
            }
        };
    }

    private static JObject BuildInt4GpuPrecisionManifest()
    {
        return new JObject
        {
            ["schemaVersion"] = "aiimage.model-manifest/v1",
            ["modelId"] = "qwen3.5_0.8b",
            ["precision"] = new JObject
            {
                ["activationDtype"] = "FP32",
                ["weightDtype"] = "INT4",
                ["sensitiveOutputDtype"] = "FP32",
                ["requireStrictTexturePlan"] = true
            },
            ["quantization"] = new JObject
            {
                ["quantizationVersion"] = "aiimage.qwen35-mobile-q4gpu/v1",
                ["calibrationVersion"] = "qwen35-q8-to-int4-weight-absmax-v1",
                ["calibrationMethod"] = "symmetric-weight-absmax-per-output-channel",
                ["weightScheme"] = "INT4_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC",
                ["outputChannelAxis"] = 0,
                ["symmetric"] = true,
                ["zeroPoint"] = 0,
                ["accumulationDtype"] = "FP32",
                ["activationQuantized"] = false,
                ["quantizedOperators"] = new JArray("Gemm", "Convolution", "InnerProduct"),
                ["nodePlans"] = new JArray(),
                ["unquantizedWeightDtype"] = "FP32"
            }
        };
    }

    private static int ResolveShardBytes()
    {
        var value = Environment.GetEnvironmentVariable(ShardBytesVariable);
        return int.TryParse(value, out var parsed) && parsed > 0 ? parsed : DefaultShardBytes;
    }

    private static HashSet<string> ResolveRebuildSet()
    {
        var value = Environment.GetEnvironmentVariable(RebuildLogicalFilesVariable);
        if (string.IsNullOrWhiteSpace(value)) return null;
        var result = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in value.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
            result.Add(item.Trim());
        if (result.Count == 0) throw new ArgumentException(RebuildLogicalFilesVariable + " did not select any logical assets.");
        return result;
    }

    private static void CopyExistingEntry(JObject existing, string logicalName, JObject logicalFiles, JObject sourceFiles)
    {
        var logical = existing?["logical_files"]?[logicalName];
        var source = existing?["source_files"]?[logicalName];
        if (logical == null || source == null)
            throw new InvalidDataException("Targeted Q8 rebuild requires an existing validated manifest entry: " + logicalName);
        logicalFiles[logicalName] = logical.DeepClone();
        sourceFiles[logicalName] = source.DeepClone();
    }

    private static string ComputeSha256(string path)
    {
        using (var sha = SHA256.Create())
        using (var stream = File.OpenRead(path))
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty).ToLowerInvariant();
    }

    private static string NormalizeRelativePath(string path) => path.Replace(Path.DirectorySeparatorChar, '/');
}
