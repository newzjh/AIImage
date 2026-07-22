using System;
using System.IO;
using Aexis.Execution;
using Aexis.Ncnn;
using Aexis.Onnx;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace Aexis.Editor
{
    [Serializable]
    internal sealed class AexisOnnxImportDiagnostics
    {
        public AexisOnnxLoweringDiagnostic[] diagnostics = Array.Empty<AexisOnnxLoweringDiagnostic>();
        public bool eligible;
    }

    public static class AexisModelPackager
    {
        public static AexisCompiledModel CompileOnnx(string assetPath)
        {
            if (string.IsNullOrWhiteSpace(assetPath))
                throw new ArgumentException("An ONNX path is required.", nameof(assetPath));

            var source = File.ReadAllBytes(assetPath);
            var result = AexisOnnxGraphLowering.Import(source);
            return new AexisCompiledModel
            {
                modelId = Path.GetFileNameWithoutExtension(assetPath),
                sourceFormat = "onnx",
                eligible = result.IsEligible,
                binaryParam = AexisNcnnBinaryParam.Serialize(result.graph),
                source = source,
                diagnosticJson = JsonUtility.ToJson(new AexisOnnxImportDiagnostics
                {
                    diagnostics = result.diagnostics ?? Array.Empty<AexisOnnxLoweringDiagnostic>(),
                    eligible = result.IsEligible
                }, true)
            };
        }

        public static AexisCompiledModel CompileNcnnParam(string assetPath)
        {
            if (string.IsNullOrWhiteSpace(assetPath))
                throw new ArgumentException("An NCNN .param path is required.", nameof(assetPath));

            var source = File.ReadAllBytes(assetPath);
            var graph = NcnnParamParser.Parse(File.ReadAllText(assetPath));
            AexisCustomLayerRegistry.ValidateDeclarations(graph.extensionDeclarations);
            var report = AexisModelPreflight.Analyze(graph, new AexisModelPreflightRequest
            {
                modelName = Path.GetFileName(assetPath),
                strict = false
            });
            var binPath = Path.ChangeExtension(assetPath, ".bin");
            return new AexisCompiledModel
            {
                modelId = Path.GetFileNameWithoutExtension(assetPath),
                sourceFormat = "ncnn-param",
                eligible = report.strictEligible,
                binaryParam = AexisNcnnBinaryParam.Serialize(graph),
                weights = File.Exists(binPath) ? File.ReadAllBytes(binPath) : Array.Empty<byte>(),
                source = source,
                diagnosticJson = AexisModelPreflight.ToStableJson(report)
            };
        }

        public static void WriteArchive(string outputPath, AexisCompiledModel model)
        {
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException("An output path is required.", nameof(outputPath));
            var fullPath = Path.GetFullPath(outputPath);
            var directory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            using (var stream = File.Create(fullPath))
                AexisModelArchive.Write(stream, model);
        }
    }

    [ScriptedImporter(1, "onnx")]
    public sealed class AexisOnnxModelImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext context)
        {
            try
            {
                AexisImportedModelAssets.AddModelAsset(context, AexisModelPackager.CompileOnnx(context.assetPath));
            }
            catch (Exception exception)
            {
                context.LogImportError(exception.Message);
                AexisImportedModelAssets.AddFailureAsset(context, "onnx", exception);
            }
        }
    }

    [ScriptedImporter(1, "param")]
    public sealed class AexisNcnnParamImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext context)
        {
            try
            {
                AexisImportedModelAssets.AddModelAsset(context, AexisModelPackager.CompileNcnnParam(context.assetPath));
            }
            catch (Exception exception)
            {
                context.LogImportError(exception.Message);
                AexisImportedModelAssets.AddFailureAsset(context, "ncnn-param", exception);
            }
        }
    }

    [ScriptedImporter(1, "aexis")]
    public sealed class AexisCompiledModelImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext context)
        {
            try
            {
                using (var stream = File.OpenRead(context.assetPath))
                    AexisImportedModelAssets.AddModelAsset(context, AexisModelArchive.Read(stream));
            }
            catch (Exception exception)
            {
                context.LogImportError(exception.Message);
                AexisImportedModelAssets.AddFailureAsset(context, "aexis", exception);
            }
        }
    }

    internal static class AexisImportedModelAssets
    {
        public static void AddModelAsset(AssetImportContext context, AexisCompiledModel model)
        {
            var asset = ScriptableObject.CreateInstance<AexisModelAsset>();
            asset.name = string.IsNullOrWhiteSpace(model.modelId) ? "AexisModel" : model.modelId;
            asset.formatVersion = AexisCompiledModel.FormatVersion;
            asset.modelId = model.modelId ?? string.Empty;
            asset.sourceFormat = model.sourceFormat ?? string.Empty;
            asset.compilerVersion = model.compilerVersion ?? string.Empty;
            asset.eligible = model.eligible;
            asset.binaryParam = model.binaryParam ?? Array.Empty<byte>();
            asset.weights = model.weights ?? Array.Empty<byte>();
            asset.source = model.source ?? Array.Empty<byte>();
            asset.manifestJson = model.manifestJson ?? string.Empty;
            asset.diagnosticJson = model.diagnosticJson ?? string.Empty;
            context.AddObjectToAsset("model", asset);
            context.SetMainObject(asset);
        }

        public static void AddFailureAsset(AssetImportContext context, string sourceFormat, Exception exception)
        {
            AddModelAsset(context, new AexisCompiledModel
            {
                modelId = Path.GetFileNameWithoutExtension(context.assetPath),
                sourceFormat = sourceFormat,
                eligible = false,
                binaryParam = Array.Empty<byte>(),
                diagnosticJson = exception.ToString()
            });
        }
    }
}
