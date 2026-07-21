using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Aexis.Ncnn;
using UnityEditor;
using UnityEngine;
using Aexis.Execution;

public static class NcnnOperatorCapabilityReportTool
{
    private const string DefaultOutputDirectory = "output/operator-capabilities";

    [MenuItem("Tools/AIImage/Inference/Export Operator Capabilities")]
    public static void ExportOperatorCapabilities()
    {
        var path = Path.Combine(Directory.GetCurrentDirectory(), DefaultOutputDirectory, "operator-capabilities.json");
        AexisOperatorCapabilities.WriteStableJson(path, AexisOperatorCapabilities.CreateDocument());
        Debug.Log("[OperatorCapabilities] wrote " + path);
    }

    [MenuItem("Tools/AIImage/Inference/Preflight YOLO and FrozenCLIP")]
    public static void PreflightReferenceModels()
    {
        var root = Directory.GetCurrentDirectory();
        var outputDirectory = Path.Combine(root, DefaultOutputDirectory);
        AexisOperatorCapabilities.WriteStableJson(
            Path.Combine(outputDirectory, "operator-capabilities.json"),
            AexisOperatorCapabilities.CreateDocument());
        ExportPreflight(
            Path.Combine(root, "Assets", "StreamingAssets", "Yolo", "yolov8n_seg.ncnn.param"),
            Path.Combine(outputDirectory, "yolov8n-seg-preflight.json"),
            new[] { CreateInput("in0", new[] { 4, 640, 640, 1, 3 }, new[] { 4, 640, 640, 1, 3 }, "NCHW", "FP32") });
        ExportPreflight(
            Path.Combine(root, "Assets", "StreamingAssets", "StableDiffusion", "FrozenCLIPEmbedder-fp16.param"),
            Path.Combine(outputDirectory, "frozen-clip-preflight.json"),
            new[]
            {
                CreateInput("token", new[] { 2, 77, 1, 1, 1 }, new[] { 2, 77, 1, 1, 1 }, "Linear", "FP32"),
                CreateInput("multiplier", new[] { 1, 1, 1, 1, 1 }, new[] { 1, 1, 1, 1, 1 }, "Scalar", "FP32"),
                CreateInput("cond", new[] { 1, 1, 1, 1, 1 }, new[] { 1, 1, 1, 1, 1 }, "Scalar", "FP32")
            });
    }

    // Unity batch entry point. It parses graph metadata only and never instantiates a runner or AexisGraphSession.
    public static void RunFromCommandLine()
    {
        var arguments = Environment.GetCommandLineArgs();
        var capabilityOutput = GetArgument(arguments, "-operatorCapabilitiesOutput")
            ?? Path.Combine(Directory.GetCurrentDirectory(), DefaultOutputDirectory, "operator-capabilities.json");
        AexisOperatorCapabilities.WriteStableJson(capabilityOutput, AexisOperatorCapabilities.CreateDocument());

        var modelPath = GetArgument(arguments, "-ncnnPreflightModel");
        if (!string.IsNullOrWhiteSpace(modelPath))
        {
            var output = GetArgument(arguments, "-ncnnPreflightOutput")
                ?? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(capabilityOutput)) ?? Directory.GetCurrentDirectory(), Path.GetFileNameWithoutExtension(modelPath) + "-preflight.json");
            var inputs = ParseInputs(GetArguments(arguments, "-ncnnPreflightInput"));
            ExportPreflight(modelPath, output, inputs.ToArray());
        }

        var strictPlanModelPath = GetArgument(arguments, "-ncnnStrictTexturePlanModel");
        if (!string.IsNullOrWhiteSpace(strictPlanModelPath))
        {
            var output = GetArgument(arguments, "-ncnnStrictTexturePlanOutput")
                ?? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(capabilityOutput)) ?? Directory.GetCurrentDirectory(), Path.GetFileNameWithoutExtension(strictPlanModelPath) + "-strict-texture-plan.json");
            var inputs = ParseInputs(GetArguments(arguments, "-ncnnStrictTexturePlanInput"));
            var debugOracleRelaxed = ResolveBool(GetArgument(arguments, "-ncnnStrictTexturePlanDebugOracle"));
            ExportStrictTexturePlan(strictPlanModelPath, output, inputs.ToArray(), debugOracleRelaxed);
        }

        Debug.Log("[OperatorCapabilities] batch report complete | capabilities=" + capabilityOutput + " | model=" + (modelPath ?? "none"));
    }

    private static void ExportPreflight(string modelPath, string outputPath, AexisPreflightTensorDescriptor[] inputs)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("ncnn param model not found", modelPath);

        var model = NcnnParamParser.Parse(File.ReadAllText(modelPath));
        var report = AexisModelPreflight.Analyze(model, new AexisModelPreflightRequest
        {
            modelName = Path.GetFileName(modelPath),
            targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
            targetDtype = "FP32",
            strict = true,
            inputs = inputs ?? Array.Empty<AexisPreflightTensorDescriptor>()
        });
        AexisModelPreflight.WriteStableJson(outputPath, report);
        Debug.Log("[OperatorCapabilities] preflight | model=" + report.modelName + " | " + report.summary + " | output=" + outputPath);
    }

    private static void ExportStrictTexturePlan(
        string modelPath,
        string outputPath,
        AexisPreflightTensorDescriptor[] inputs,
        bool debugOracleRelaxed)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("ncnn strict texture plan model not found", modelPath);

        var model = NcnnParamParser.Parse(File.ReadAllText(modelPath));
        var plan = AexisTextureExecutionPlanner.Analyze(model, new AexisTextureExecutionPlanRequest
        {
            modelName = Path.GetFileName(modelPath),
            targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
            targetDtype = "FP16",
            targetLayout = AexisTexturePlanLayout.Packed4,
            strict = !debugOracleRelaxed,
            debugOracleRelaxed = debugOracleRelaxed,
            inputs = (inputs ?? Array.Empty<AexisPreflightTensorDescriptor>()).Where(input => input != null).Select(input => new AexisTexturePlanTensorDescriptor
            {
                blob = input.blob,
                logicalShape = input.logicalShape,
                storageShape = input.storageShape,
                layout = input.layout,
                dtype = input.dtype,
                aliasGroup = "cli:" + input.blob,
                textureBacked = true
            }).ToArray()
        });
        AexisTextureExecutionPlanner.WriteStableJson(outputPath, plan);
        AexisTextureExecutionPlanner.ThrowIfDispatchRejected(plan);
        Debug.Log("[OperatorCapabilities] strict texture plan | model=" + plan.modelName + " | " + plan.summary + " | output=" + outputPath);
    }

    private static AexisPreflightTensorDescriptor CreateInput(string blob, int[] logicalShape, int[] storageShape, string layout, string dtype)
    {
        return new AexisPreflightTensorDescriptor
        {
            blob = blob,
            logicalShape = logicalShape,
            storageShape = storageShape,
            layout = layout,
            dtype = dtype
        };
    }

    // Format: blob|logical-dims|storage-dims|layout|dtype; dims use comma-separated integers.
    private static List<AexisPreflightTensorDescriptor> ParseInputs(string[] values)
    {
        var inputs = new List<AexisPreflightTensorDescriptor>();
        if (values == null)
            return inputs;

        for (var index = 0; index < values.Length; index++)
        {
            var parts = (values[index] ?? string.Empty).Split('|');
            if (parts.Length != 5 || string.IsNullOrWhiteSpace(parts[0]))
                throw new ArgumentException("-ncnnPreflightInput must be blob|logical-dims|storage-dims|layout|dtype");
            inputs.Add(CreateInput(parts[0], ParseShape(parts[1]), ParseShape(parts[2]), parts[3], parts[4]));
        }
        return inputs;
    }

    private static int[] ParseShape(string value)
    {
        var segments = (value ?? string.Empty).Split(',');
        if (segments.Length == 0)
            throw new ArgumentException("Tensor shape must contain at least one dimension.");
        var result = new int[segments.Length];
        for (var i = 0; i < segments.Length; i++)
        {
            if (!int.TryParse(segments[i], out result[i]) || result[i] <= 0)
                throw new ArgumentException("Tensor shape contains an invalid dimension: " + value);
        }
        return result;
    }

    private static string GetArgument(string[] arguments, string name)
    {
        var values = GetArguments(arguments, name);
        return values.Length == 0 ? null : values[0];
    }

    private static string[] GetArguments(string[] arguments, string name)
    {
        var values = new List<string>();
        for (var i = 0; arguments != null && i < arguments.Length - 1; i++)
        {
            if (string.Equals(arguments[i], name, StringComparison.Ordinal))
                values.Add(arguments[++i]);
        }
        return values.ToArray();
    }

    private static bool ResolveBool(string value)
    {
        return string.Equals(value, "1", StringComparison.Ordinal)
            || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "yes", StringComparison.OrdinalIgnoreCase);
    }
}
