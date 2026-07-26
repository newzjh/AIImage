#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using Aexis.Execution;
using Aexis.Ncnn;
using Aexis.Samples.Json.Linq;
using AIImage.Qwen35;
using UnityEditor;
using UnityEngine;

public static class Qwen35MobileQ4AssetBuilder
{
    private const string SourceModelVariable = "AIIMAGE_QWEN35_Q4_SOURCE_MODEL";
    private const string OutputModelVariable = "AIIMAGE_QWEN35_Q4_OUTPUT";
    private const string ShardBytesVariable = "AIIMAGE_QWEN35_Q4_SHARD_BYTES";
    private const string PlayerQ4ModelVariable = "AIIMAGE_QWEN35_Q4_MODEL_DIR";
    private const string PlayerQ8ModelVariable = "AIIMAGE_QWEN35_Q8_MODEL_DIR";
    private const int DefaultShardBytes = 128 * 1024 * 1024;
    // Decoder Q4 needs fine groups for Qwen3.5's recurrent blocks.  The tied
    // token embedding/output projection stays in its source Q8 representation:
    // empirical validation shows RTN Q4 changes the first greedy token there.
    private const int Q4GroupSize = 16;

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

    [MenuItem("AIImage/Qwen3.5/Build Mobile Q4 Assets")]
    public static void BuildInteractive()
    {
        Build(
            ResolveSourceDirectory(),
            ResolveOutputDirectory(),
            ResolveShardBytes());
    }

    public static void BuildBatch()
    {
        try
        {
            Build(
                ResolveSourceDirectory(),
                ResolveOutputDirectory(),
                ResolveShardBytes());
        }
        catch (Exception error)
        {
            UnityEngine.Debug.LogError("QWEN35_Q4_BUILD_FAILED\n" + error);
            throw;
        }
    }

    internal static string ResolveQ4SourceDirectory()
    {
        var configured = Environment.GetEnvironmentVariable(PlayerQ4ModelVariable);
        if (!string.IsNullOrWhiteSpace(configured)
            && Qwen35MobileAssetSet.TryLoad(Path.GetFullPath(configured), verifyHashes: false)?.QuantizationBits == 4)
        {
            return Path.GetFullPath(configured);
        }

        var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        var generated = Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "_models", "qwen3.5_0.8b_mobile_q4");
        return Qwen35MobileAssetSet.TryLoad(generated, verifyHashes: false)?.QuantizationBits == 4 ? generated : null;
    }

    private static void Build(string sourceModel, string outputModel, int shardBytes)
    {
        if (string.IsNullOrWhiteSpace(sourceModel)) throw new ArgumentException(SourceModelVariable + " is required.");
        if (string.IsNullOrWhiteSpace(outputModel)) throw new ArgumentException(OutputModelVariable + " is required.");
        if (shardBytes < 16 * 1024 * 1024) throw new ArgumentOutOfRangeException(nameof(shardBytes));

        var source = Path.GetFullPath(sourceModel);
        var output = Path.GetFullPath(outputModel);
        if (string.Equals(source.TrimEnd('\\', '/'), output.TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Qwen3.5 Q4 output must differ from the Q8 source directory.");
        var sourceAssets = Qwen35MobileAssetSet.TryLoad(source, verifyHashes: true)
            ?? throw new InvalidDataException("Qwen3.5 Q4 build requires a validated Q8 source asset set: " + source);
        if (sourceAssets.QuantizationBits != 8)
            throw new InvalidDataException("Qwen3.5 Q4 build source must be native Q8, but was Q" + sourceAssets.QuantizationBits + ".");

        Directory.CreateDirectory(output);
        var weightsDirectory = Path.Combine(output, "weights");
        Directory.CreateDirectory(weightsDirectory);
        foreach (var file in DeliveryFiles)
            File.Copy(Path.Combine(source, file), Path.Combine(output, file), true);

        var precisionManifest = BuildPrecisionManifest();
        File.WriteAllText(Path.Combine(output, Qwen35MobileAssetSet.Q4PrecisionManifestFileName), precisionManifest.ToString());
        var runtimeManifest = AexisModelManifestLoader.LoadFromJson(precisionManifest.ToString(), "Qwen3.5 mobile Q4 builder");
        var projectionPrecisionManifest = BuildProjectionPrecisionManifest();
        File.WriteAllText(
            Path.Combine(output, Qwen35MobileAssetSet.Q4ProjectionPrecisionManifestFileName),
            projectionPrecisionManifest.ToString());
        AexisModelManifestLoader.LoadFromJson(
            projectionPrecisionManifest.ToString(),
            "Qwen3.5 mobile Q4 projection builder");
        var logicalFiles = new JObject();
        var sourceFiles = new JObject();
        var stopwatch = Stopwatch.StartNew();

        CopySharedEmbeddingQ8(source, output, weightsDirectory, shardBytes, logicalFiles, sourceFiles);
        foreach (var network in QuantizedNetworks)
            ConvertNetwork(source, output, weightsDirectory, network.Param, network.Bin, shardBytes, runtimeManifest, logicalFiles, sourceFiles);

        stopwatch.Stop();
        long storedBytes = 0;
        long sourceBytes = 0;
        foreach (var property in logicalFiles.Properties()) storedBytes += (long)property.Value["stored_bytes"];
        foreach (var property in sourceFiles.Properties()) sourceBytes += (long)property.Value["bytes"];
        var manifestPath = Path.Combine(output, Qwen35MobileAssetSet.Q4ManifestFileName);
        File.WriteAllText(manifestPath, new JObject
        {
            ["schema"] = Qwen35MobileAssetSet.Q4ManifestSchema,
            ["model_id"] = "qwen3.5_0.8b",
            ["format"] = "aiimage-q4-decoder-q8-tied-embedding/v1",
            ["quantization_bits"] = 4,
            ["weight_only"] = true,
            ["tied_embedding_quantization_bits"] = 8,
            ["projection_quantization_bits"] = 8,
            ["source_model_directory"] = source,
            ["source_bytes"] = sourceBytes,
            ["stored_weight_bytes"] = storedBytes,
            ["compression_ratio"] = sourceBytes > 0 ? storedBytes / (double)sourceBytes : 0d,
            ["shard_limit_bytes"] = shardBytes,
            ["build_elapsed_seconds"] = stopwatch.Elapsed.TotalSeconds,
            ["source_files"] = sourceFiles,
            ["logical_files"] = logicalFiles
        }.ToString());

        var verified = Qwen35MobileAssetSet.TryLoad(output, verifyHashes: true);
        if (verified == null || verified.QuantizationBits != 4 || verified.StoredWeightBytes != storedBytes)
            throw new InvalidDataException("Generated Qwen3.5 Q4 manifest failed self-validation.");
        File.WriteAllText(Path.Combine(output, "qwen3.5_mobile_q4_build_report.json"), new JObject
        {
            ["schema"] = "qwen35.mobile-q4-build-report/v1",
            ["valid"] = true,
            ["source_model"] = source,
            ["output_model"] = output,
            ["stored_weight_bytes"] = storedBytes,
            ["source_bytes"] = sourceBytes,
            ["elapsed_seconds"] = stopwatch.Elapsed.TotalSeconds
        }.ToString());
        UnityEngine.Debug.Log("QWEN35_Q4_BUILD_OK output=" + output + " stored_bytes=" + storedBytes);
        AssetDatabase.Refresh();
    }

    private static void CopySharedEmbeddingQ8(
        string source,
        string output,
        string weightsDirectory,
        int shardBytes,
        JObject logicalFiles,
        JObject sourceFiles)
    {
        const string logicalName = "qwen3.5_embed_token.ncnn.bin";
        var temporaryPath = Path.Combine(weightsDirectory, logicalName + ".q4.building");
        using (var input = Qwen35ModelAssetResolver.OpenBin(source, logicalName))
        using (var archive = new FileStream(temporaryPath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024, FileOptions.SequentialScan))
        {
            input.CopyTo(archive, 1024 * 1024);
        }
        logicalFiles[logicalName] = FinalizeArchive(output, weightsDirectory, logicalName, temporaryPath, shardBytes, "row-int8", Qwen35DecoderSession.HiddenSize);
        sourceFiles[logicalName] = DescribeSource(source, logicalName);
    }

    private static void ConvertNetwork(
        string source,
        string output,
        string weightsDirectory,
        string paramName,
        string logicalName,
        int shardBytes,
        Aexis.ModelManifest runtimeManifest,
        JObject logicalFiles,
        JObject sourceFiles)
    {
        var temporaryPath = Path.Combine(weightsDirectory, logicalName + ".q4.building");
        using (var input = Qwen35ModelAssetResolver.OpenBin(source, logicalName))
        using (var archive = new FileStream(temporaryPath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024, FileOptions.SequentialScan))
        using (var writer = new NcnnQ8ArchiveWriter(archive, input.Length, defaultBlockSize: Q4GroupSize, fp32Threshold: 4096, quantizationBits: 4))
        using (var reader = new NcnnBinReader(input, writer))
        using (var ops = new AexisOps())
        using (var repro = new AexisGraphSession(ops))
        {
            repro.ApplyModelManifest(runtimeManifest);
            repro.LoadModel(File.ReadAllText(Path.Combine(source, paramName)), reader);
        }
        logicalFiles[logicalName] = FinalizeArchive(output, weightsDirectory, logicalName, temporaryPath, shardBytes, "groupwise-int4-or-fp32-small-constant", Q4GroupSize);
        sourceFiles[logicalName] = DescribeSource(source, logicalName);
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
                var fileName = logicalName + ".q4.part" + partIndex.ToString("D3");
                var partPath = Path.Combine(weightsDirectory, fileName);
                var remaining = Math.Min((long)shardBytes, source.Length - source.Position);
                using (var part = new FileStream(partPath, FileMode.Create, FileAccess.Write, FileShare.None, buffer.Length, FileOptions.SequentialScan))
                {
                    while (remaining > 0)
                    {
                        var read = source.Read(buffer, 0, (int)Math.Min(buffer.Length, remaining));
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

    private static JObject BuildPrecisionManifest()
    {
        return new JObject
        {
            ["schemaVersion"] = "aiimage.model-manifest/v1",
            ["modelId"] = "qwen3.5_0.8b",
            ["precision"] = new JObject
            {
                // Token ids such as Qwen's 248068 vocabulary control token
                // exceed FP16's largest finite value (65504).  Keeping runtime
                // activations in FP32 preserves the texture argmax label path.
                ["activationDtype"] = "FP32",
                ["weightDtype"] = "INT4",
                ["sensitiveOutputDtype"] = "FP32",
                ["requireStrictTexturePlan"] = true
            },
            ["quantization"] = new JObject
            {
                ["quantizationVersion"] = "aiimage.qwen35-mobile-q4/v4",
                ["calibrationVersion"] = "qwen35-q8-to-q4-groupwise-absmax-v4",
                ["calibrationMethod"] = "symmetric-weight-absmax-per-output-channel-group",
                // Keep the manifest's established INT4 capability enum; groupSize
                // carries the finer archive/runtime scale granularity.
                ["weightScheme"] = "INT4_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC",
                ["outputChannelAxis"] = 0,
                ["groupSize"] = Q4GroupSize,
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

    private static JObject BuildProjectionPrecisionManifest()
    {
        return new JObject
        {
            ["schemaVersion"] = "aiimage.model-manifest/v1",
            ["modelId"] = "qwen3.5_0.8b_projection_q8",
            ["precision"] = new JObject
            {
                ["activationDtype"] = "FP32",
                ["weightDtype"] = "INT4",
                ["sensitiveOutputDtype"] = "FP32",
                ["requireStrictTexturePlan"] = true
            },
            ["quantization"] = new JObject
            {
                ["quantizationVersion"] = "aiimage.qwen35-mobile-q4-q8-projection/v1",
                ["calibrationVersion"] = "qwen35-mobile-q4-q8-projection-v1",
                ["calibrationMethod"] = "q8-tied-embedding-output-projection",
                ["weightScheme"] = "INT4_WEIGHT_ONLY_PER_OUTPUT_CHANNEL_SYMMETRIC",
                ["outputChannelAxis"] = 0,
                ["groupSize"] = Q4GroupSize,
                ["symmetric"] = true,
                ["zeroPoint"] = 0,
                ["accumulationDtype"] = "FP32",
                ["activationQuantized"] = false,
                ["quantizedOperators"] = new JArray(),
                ["nodePlans"] = new JArray
                {
                    new JObject
                    {
                        ["layerName"] = "gemm_0",
                        ["operatorName"] = "Gemm",
                        ["mode"] = "W8",
                        ["activationScale"] = 1f,
                        ["activationZeroPoint"] = 0
                    }
                },
                ["unquantizedWeightDtype"] = "FP32"
            }
        };
    }

    private static string ResolveSourceDirectory()
    {
        var configured = Environment.GetEnvironmentVariable(SourceModelVariable);
        if (!string.IsNullOrWhiteSpace(configured)) return Path.GetFullPath(configured);
        configured = Environment.GetEnvironmentVariable(PlayerQ8ModelVariable);
        if (!string.IsNullOrWhiteSpace(configured)) return Path.GetFullPath(configured);
        var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        return Path.Combine(projectRoot, "Assets", "StreamingAssets", "QWEN35", "qwen3.5_0.8b_mobile_q8");
    }

    private static string ResolveOutputDirectory()
    {
        var configured = Environment.GetEnvironmentVariable(OutputModelVariable);
        if (!string.IsNullOrWhiteSpace(configured)) return Path.GetFullPath(configured);
        var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        return Path.Combine(projectRoot, "Tools", "Qwen35NcnnBaseline", "_models", "qwen3.5_0.8b_mobile_q4");
    }

    private static int ResolveShardBytes()
    {
        var value = Environment.GetEnvironmentVariable(ShardBytesVariable);
        return int.TryParse(value, out var parsed) && parsed > 0 ? parsed : DefaultShardBytes;
    }

    private static JObject DescribeSource(string modelDirectory, string logicalName)
    {
        var bytes = Qwen35ModelAssetResolver.GetStoredBytes(modelDirectory, logicalName);
        if (bytes <= 0) throw new InvalidDataException("Qwen3.5 Q8 source weight is unavailable: " + logicalName);
        using (var sha = SHA256.Create())
        using (var stream = Qwen35ModelAssetResolver.OpenBin(modelDirectory, logicalName))
            return new JObject
            {
                ["bytes"] = bytes,
                ["sha256"] = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty).ToLowerInvariant()
            };
    }

    private static string ComputeSha256(string path)
    {
        using (var sha = SHA256.Create())
        using (var stream = File.OpenRead(path))
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty).ToLowerInvariant();
    }

    private static string NormalizeRelativePath(string path) => path.Replace(Path.DirectorySeparatorChar, '/');
}
#endif
