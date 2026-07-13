using System;
using System.Collections.Generic;
using System.IO;
using NcnnCompute;
using NUnit.Framework;
using UnityEngine;

public sealed class NcnnProductionPathAuditTests
{
    private static readonly string[] ForbiddenPublicReadbackApis =
    {
        "GetBufferData(",
        "public ComputeBuffer GetBuffer("
    };

    private static readonly string[] LegacyBufferMaterializationApis =
    {
        "GetOrConvertToBuffer(",
        "Pack4ToBufferCHW(",
        "Pack4ToBufferCDHW(",
        "TextureToBuffer3(",
        "BufferToTexture3("
    };

    // These are the remaining explicitly audited legacy runner diagnostics. Any new use,
    // including an additional use in one of these files, must update the A2 audit deliberately.
    private static readonly Dictionary<string, int[]> AuditedLegacyRunnerApiCounts = new Dictionary<string, int[]>
    {
        { "Assets/Scripts/ClipNcnnReproRunner.cs", new[] { 0, 2, 0, 0, 0 } },
        { "Assets/Scripts/CodeFormerNcnnReproRunner2.cs", new[] { 0, 2, 0, 0, 0 } },
        { "Assets/Scripts/MatterNcnnReproRunner.cs", new[] { 0, 2, 0, 0, 0 } },
        { "Assets/Scripts/MONAINcnnReproRunner.cs", new[] { 0, 2, 3, 0, 0 } },
        { "Assets/Scripts/SDInpaintingNcnnReproRunner.cs", new[] { 0, 6, 2, 0, 0 } },
        { "Assets/Scripts/SDNcnnReproRunner.cs", new[] { 0, 4, 0, 0, 0 } },
        { "Assets/Scripts/YoloSegNcnnReproRunner.cs", new[] { 0, 1, 0, 0, 0 } }
    };

    [Test]
    public void InferResult_DoesNotExposeProductionBufferReadback()
    {
        var source = ReadPackageSource("NcnnCompute", "NcnnRepro.cs");
        foreach (var api in ForbiddenPublicReadbackApis)
            Assert.That(source, Does.Not.Contain(api), "Production InferResult must not expose " + api);

        Assert.That(source, Does.Contain("ReadTextureDataForOutput"));
        Assert.That(source, Does.Contain("#if UNITY_EDITOR || AIIMAGE_INFERENCE_DEBUG_ORACLE"));
        Assert.That(source, Does.Contain("public NcnnInferenceExecutionMode ExecutionMode { get; set; } = NcnnInferenceExecutionMode.ProductionTextureOnly;"));
        Assert.That(source, Does.Contain("IInferenceSession"));
    }

    [Test]
    public void ProductionBoundary_AllowsFixedBufferInputsAndRejectsIntermediateMaterializationWithTensorContract()
    {
        var repro = ReadPackageSource("NcnnCompute", "NcnnRepro.cs");
        var factory = ReadPackageSource("NcnnLayers", "NcnnLayerFactoryRepro.cs");
        var reshape = ReadPackageSource("NcnnLayers", "NcnnReshapeLayerRepro.cs");
        var embed = ReadPackageSource("NcnnLayers", "NcnnEmbedLayerRepro.cs");
        var innerProduct = ReadPackageSource("NcnnLayers", "NcnnInnerProductLayerRepro.cs");
        var permute = ReadPackageSource("NcnnLayers", "NcnnPermuteLayerRepro.cs");
        var inpainting = ReadAssetSource("Scripts/SDInpaintingNcnnReproRunner.cs");

        Assert.That(repro, Does.Contain("if (!IsDebugOracleExecution)\n                return true;"));
        Assert.That(repro, Does.Contain("logical_shape="));
        Assert.That(repro, Does.Contain("storage_shape="));
        Assert.That(repro, Does.Contain("layout="));
        Assert.That(repro, Does.Contain("dtype="));
        Assert.That(repro, Does.Contain("rejected_fallback="));
        Assert.That(factory, Does.Not.Contain("rejects ComputeBuffer model input"));
        Assert.That(factory, Does.Contain("if (IsDebugOracleExecution && !textureBlobs.ContainsKey(kv.Key))"));
        Assert.That(factory, Does.Contain("fixed Buffer input is a valid graph boundary"));
        Assert.That(factory, Does.Contain("SetCurrentBufferExecutionContext(context)"));
        var input = ReadPackageSource("NcnnLayers", "NcnnInputLayerRepro.cs");
        Assert.That(input, Does.Contain("Input is a graph-boundary alias"));
        Assert.That(input, Does.Contain("context.bufferBlobs[topName] = inputBuffer"));
        Assert.That(input, Does.Not.Contain("Materialize"));
        Assert.That(reshape, Does.Contain("owner.ExecutionMode == NcnnInferenceExecutionMode.ProductionTextureOnly"));
        Assert.That(reshape, Does.Contain("ReshapePack4ToLinearMat"));
        Assert.That(embed, Does.Contain("public override void ExecuteBuffer"));
        Assert.That(embed, Does.Contain("owner.Ops.EmbedTexture(indexBuffer"));
        Assert.That(innerProduct, Does.Contain("Gemm2DAttentionPack4ToLinearTextureA"));
        Assert.That(innerProduct, Does.Contain("TryResolveAttentionPack4ToLinearInput"));
        Assert.That(permute, Does.Contain("HasSingleContextFlattenConsumer"));
        var binaryOp = ReadPackageSource("NcnnLayers", "NcnnBinaryOpLayerRepro.cs");
        var slice = ReadPackageSource("NcnnLayers", "NcnnSliceLayerRepro.cs");
        var ops = ReadPackageSource("NcnnCompute", "NcnnOps.cs");
        var computeShader = ReadKernelSource("NcnnCompute.compute");
        var matmulKernels = ReadKernelSource("NcnnComputeIncludes", "KernelGroups", "NcnnKernels.Pack4Matmul.hlsl");
        Assert.That(binaryOp, Does.Contain("broadcastMode = 4"));
        Assert.That(binaryOp, Does.Contain("bShape.c == 1"));
        Assert.That(binaryOp, Does.Contain("IsStrictLinearMatTexture(scalarTexture)"));
        Assert.That(binaryOp, Does.Contain("BinaryOpLinearMatFixedInputScalar"));
        Assert.That(slice, Does.Contain("srcShape.dims == 1 || srcShape.dims == 2"));
        Assert.That(slice, Does.Contain("srcShape.dims == 1 && spec.axis != 0"));
        Assert.That(ops, Does.Contain("BinaryOpLinearMatFixedInputScalar(RenderTexture texture, ComputeBuffer scalar"));
        Assert.That(computeShader, Does.Contain("#pragma kernel NcnnBinaryOpLinearMatFixedInputScalar"));
        Assert.That(matmulKernels, Does.Contain("void NcnnBinaryOpLinearMatFixedInputScalar_Impl"));
        Assert.That(inpainting, Does.Contain("FrozenCLIPEmbedder-fp16.param"));
        Assert.That(inpainting, Does.Contain("FrozenCLIPEmbedder-fp16.bin"));
        Assert.That(inpainting, Does.Contain("stopAfterTopName: TextEncoderOutputBlobName"));
    }

    [Test]
    public void LegacyBufferCalls_AreConfinedToTheAuditedEngineDirectories()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var allowedRoots = new[]
        {
            Path.Combine(root, "Packages", "com.aiimage.inference.unitygpu", "Runtime", "NcnnCompute"),
            Path.Combine(root, "Packages", "com.aiimage.inference.unitygpu", "Runtime", "NcnnLayers")
        };
        var violations = new List<string>();

        foreach (var file in Directory.EnumerateFiles(Path.Combine(root, "Assets", "Scripts"), "*.cs", SearchOption.AllDirectories))
        {
            if (Array.Exists(allowedRoots, allowedRoot => file.StartsWith(allowedRoot, StringComparison.OrdinalIgnoreCase)))
                continue;

            var source = File.ReadAllText(file);
            var relativePath = MakeAssetRelative(root, file);
            var observed = new int[LegacyBufferMaterializationApis.Length];
            var total = 0;
            for (var i = 0; i < LegacyBufferMaterializationApis.Length; i++)
            {
                observed[i] = CountOccurrences(source, LegacyBufferMaterializationApis[i]);
                total += observed[i];
            }

            if (total == 0)
                continue;

            if (!AuditedLegacyRunnerApiCounts.TryGetValue(relativePath, out var expected))
            {
                violations.Add(relativePath + " => unapproved legacy Buffer API");
                continue;
            }

            for (var i = 0; i < observed.Length; i++)
            {
                if (observed[i] != expected[i])
                    violations.Add(relativePath + " => " + LegacyBufferMaterializationApis[i] + " expected=" + expected[i] + " actual=" + observed[i]);
            }
        }

        Assert.That(violations, Is.Empty, "Forbidden Buffer materialization escaped the audited engine/debug boundary:\n" + string.Join("\n", violations));
    }

    [Test]
    public void ProductionAuditLog_StatesThatIntermediateBufferMaterializationIsZero()
    {
        var source = ReadPackageSource("NcnnCompute", "NcnnRepro.cs");
        Assert.That(source, Does.Contain("[InferencePathAudit] mode=ProductionTextureOnly"));
        Assert.That(source, Does.Contain("intermediate_buffer_materializations=0"));
    }

    [Test]
    public void TensorDescriptor_CompatibleViewsKeepRenderTextureOwnerAndRemainImmutable()
    {
        var texture = new RenderTexture(4, 4, 0, RenderTextureFormat.ARGBHalf);
        try
        {
            var storage = new NcnnRepro.BufferShape(3, 4, 4, 1, 8);
            var source = NcnnRepro.CreateTextureRef(texture, storage, storage, owned: false, blobName: "source");
            var descriptor = source.Descriptor;

            var reshape = NcnnRepro.CreateTextureAlias(
                source,
                new NcnnRepro.BufferShape(2, 16, 8, 1, 1),
                storage);
            var squeeze = NcnnRepro.CreateTextureAlias(
                source,
                new NcnnRepro.BufferShape(2, 16, 8, 1, 1),
                storage);
            var flatten = NcnnRepro.CreateTextureAlias(
                source,
                new NcnnRepro.BufferShape(1, 128, 1, 1, 1),
                storage);

            Assert.That(reshape.texture, Is.SameAs(texture));
            Assert.That(squeeze.texture, Is.SameAs(texture));
            Assert.That(flatten.texture, Is.SameAs(texture));
            Assert.That(reshape.sharedTextureOwner, Is.SameAs(source));
            Assert.That(flatten.Descriptor.Owner, Is.SameAs(source));
            Assert.That(source.refs, Is.EqualTo(4));
            Assert.That(flatten.Descriptor.AliasGroup, Is.EqualTo(descriptor.AliasGroup));

            source.logicalShape = new NcnnRepro.BufferShape(3, 1, 1, 1, 1);
            source.storageShape = new NcnnRepro.BufferShape(3, 1, 1, 1, 1);
            Assert.That(source.Descriptor, Is.SameAs(descriptor));
            Assert.That(source.Descriptor.LogicalShape, Is.EqualTo(storage));
            Assert.That(source.Descriptor.StorageShape, Is.EqualTo(storage));
            Assert.That(typeof(TensorDescriptor).GetProperty(nameof(TensorDescriptor.LogicalShape)).CanWrite, Is.False);
            Assert.That(typeof(TensorDescriptor).GetProperty(nameof(TensorDescriptor.StorageShape)).CanWrite, Is.False);

        }
        finally
        {
            texture.Release();
            UnityEngine.Object.DestroyImmediate(texture);
        }
    }

    [Test]
    public void TensorDescriptor_IncompatibleAliasRequiresTextureTransformWithoutBufferFallback()
    {
        var texture = new RenderTexture(4, 4, 0, RenderTextureFormat.ARGBHalf);
        try
        {
            var sourceStorage = new NcnnRepro.BufferShape(3, 4, 4, 1, 8);
            var source = NcnnRepro.CreateTextureRef(texture, sourceStorage, sourceStorage, owned: false, blobName: "source");
            var targetLogical = new NcnnRepro.BufferShape(3, 8, 2, 1, 8);
            var targetStorage = new NcnnRepro.BufferShape(3, 8, 2, 1, 8);

            var error = Assert.Throws<TensorAliasTransformRequiredException>(
                () => NcnnRepro.CreateTextureAlias(source, targetLogical, targetStorage));

            Assert.That(error.Message, Does.Contain("source_logical=dims=3 w=4 h=4 d=1 c=8"));
            Assert.That(error.Message, Does.Contain("source_storage=dims=3 w=4 h=4 d=1 c=8"));
            Assert.That(error.Message, Does.Contain("target_logical=dims=3 w=8 h=2 d=1 c=8"));
            Assert.That(error.Message, Does.Contain("target_storage=dims=3 w=8 h=2 d=1 c=8"));
            Assert.That(error.Message, Does.Contain("requires_texture_transform=true"));
            Assert.That(error.Message, Does.Contain("buffer fallback is prohibited"));
        }
        finally
        {
            texture.Release();
            UnityEngine.Object.DestroyImmediate(texture);
        }
    }

    [Test]
    public void TensorDescriptor_CommandTextureAliasRetainsOwnerAndDescriptor()
    {
        var texture = new ComputeTexture
        {
            nameID = 17,
            width = 4,
            height = 4,
            depth = 2,
            dimension = UnityEngine.Rendering.TextureDimension.Tex2DArray,
            format = RenderTextureFormat.ARGBHalf,
            trackerLabel = "cmd-source"
        };
        var storage = new NcnnRepro.BufferShape(3, 4, 4, 1, 8);
        var source = NcnnRepro.CreateCmdTensorRef(texture, storage, storage, owned: false, blobName: "cmd-source");
        var alias = NcnnRepro.CreateCmdTensorAlias(
            source,
            new NcnnRepro.BufferShape(1, 128, 1, 1, 1),
            storage);

        Assert.That(alias.texture, Is.SameAs(texture));
        Assert.That(alias.sharedTextureOwner, Is.SameAs(source));
        Assert.That(alias.Descriptor.Owner, Is.SameAs(source));
        Assert.That(alias.Descriptor.AliasGroup, Is.EqualTo(source.Descriptor.AliasGroup));
        Assert.That(alias.Descriptor.Lifetime, Is.EqualTo(InferenceTensorLifetime.SharedAlias));
    }

    public static void RunBatchValidation()
    {
        var tests = new NcnnProductionPathAuditTests();
        tests.InferResult_DoesNotExposeProductionBufferReadback();
        tests.ProductionBoundary_AllowsFixedBufferInputsAndRejectsIntermediateMaterializationWithTensorContract();
        tests.LegacyBufferCalls_AreConfinedToTheAuditedEngineDirectories();
        tests.ProductionAuditLog_StatesThatIntermediateBufferMaterializationIsZero();
        tests.TensorDescriptor_CompatibleViewsKeepRenderTextureOwnerAndRemainImmutable();
        tests.TensorDescriptor_IncompatibleAliasRequiresTextureTransformWithoutBufferFallback();
        tests.TensorDescriptor_CommandTextureAliasRetainsOwnerAndDescriptor();
        Debug.Log("[NcnnProductionPathAudit] passed");
    }

    private static string ReadAssetSource(string relativePath)
    {
        return File.ReadAllText(Path.Combine(Application.dataPath, relativePath));
    }

    private static string ReadPackageSource(params string[] relativePath)
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var path = Path.Combine(root, "Packages", "com.aiimage.inference.unitygpu", "Runtime");
        foreach (var segment in relativePath)
            path = Path.Combine(path, segment);
        return File.ReadAllText(path);
    }

    private static string ReadKernelSource(params string[] relativePath)
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var path = Path.Combine(root, "Packages", "com.aiimage.inference.kernels", "Runtime", "Resources");
        foreach (var segment in relativePath)
            path = Path.Combine(path, segment);
        return File.ReadAllText(path);
    }

    private static string MakeAssetRelative(string root, string path)
    {
        return path.Substring(root.Length + 1).Replace('\\', '/');
    }

    private static int CountOccurrences(string source, string value)
    {
        var count = 0;
        var index = 0;
        while ((index = source.IndexOf(value, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += value.Length;
        }
        return count;
    }
}
