#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.IO;
using System.Linq;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;
using Aexis.Execution;

public sealed class NcnnC4Pack4LayoutTests
{
    [Test]
    public void AliasMatrix_UsesDescriptorsForViewsAndRejectsTransforms()
    {
        var texture = new ComputeTexture
        {
            nameID = 4104,
            width = 3,
            height = 2,
            depth = 2,
            dimension = TextureDimension.Tex2DArray,
            format = RenderTextureFormat.ARGBFloat,
            trackerLabel = "c4-alias-source"
        };
        var packed = new AexisGraphSession.BufferShape(3, 3, 2, 1, 5);
        var source = AexisGraphSession.CreateCmdTensorRef(texture, packed, packed, owned: false, blobName: "source");

        var reshapeIdentity = AexisGraphSession.CreateCmdTensorAlias(source, packed, packed);
        var permuteIdentity = AexisGraphSession.CreateCmdTensorAlias(source, packed, packed);
        var tileIdentity = AexisGraphSession.CreateCmdTensorAlias(source, packed, packed);
        Assert.That(reshapeIdentity.texture, Is.SameAs(texture));
        Assert.That(permuteIdentity.Descriptor.AliasGroup, Is.EqualTo(source.Descriptor.AliasGroup));
        Assert.That(tileIdentity.Descriptor.Lifetime, Is.EqualTo(InferenceTensorLifetime.SharedAlias));

        var cdhw = new AexisGraphSession.BufferShape(4, 3, 2, 1, 5);
        var squeeze = AexisGraphSession.CreateCmdTensorAlias(source, cdhw, packed);
        var expand = AexisGraphSession.CreateCmdTensorAlias(squeeze, packed, packed);
        Assert.That(squeeze.Descriptor.StorageShape, Is.EqualTo(source.Descriptor.StorageShape));
        Assert.That(expand.sharedTextureOwner, Is.SameAs(source));

        Assert.Throws<TensorAliasTransformRequiredException>(() => AexisGraphSession.CreateCmdTensorAlias(
            source,
            new AexisGraphSession.BufferShape(3, 2, 3, 1, 5),
            new AexisGraphSession.BufferShape(3, 2, 3, 1, 5)));
        Assert.Throws<TensorAliasTransformRequiredException>(() => AexisGraphSession.CreateCmdTensorAlias(
            source,
            new AexisGraphSession.BufferShape(1, 30, 1, 1, 1),
            new AexisGraphSession.BufferShape(3, 30, 1, 1, 1)));
    }

    [Test]
    public void CapabilityMetadata_RecordsC4ProfilesAndRejectsRotaryProductionPlaceholder()
    {
        var capabilities = AexisOperatorCapabilities.CreateDocument().operators;
        foreach (var operatorName in new[] { "Reshape", "Flatten", "Squeeze", "ExpandDims", "Permute", "Slice", "Tile", "Packing", "Cast" })
        {
            var capability = capabilities.Single(item => item.operatorName == operatorName);
            Assert.That(capability.status, Is.EqualTo(AexisOperatorCapabilityStatus.Partial), operatorName);
            Assert.That(capability.profiles.Single(profile => profile.backend == AexisOperatorCapabilityBackend.CommandBuffer).shapeProfile, Does.Contain("descriptor alias"), operatorName);
            Assert.That(capability.limitations, Does.Contain("Placeholder publication is not a production path"), operatorName);
        }

        var rotary = capabilities.Single(item => item.operatorName == "RotaryEmbed");
        Assert.That(rotary.status, Is.Not.EqualTo(AexisOperatorCapabilityStatus.Supported));
        Assert.That(rotary.commandBuffer, Is.False);
    }

    [Test]
    public void StrictPlanner_AliasesProvenIdentityAndRejectsUnprovenFlattenTransform()
    {
        var input = new AexisTexturePlanTensorDescriptor
        {
            blob = "in",
            logicalShape = new[] { 3, 3, 2, 1, 5 },
            storageShape = new[] { 3, 3, 2, 1, 5 },
            layout = AexisTexturePlanLayout.Packed4,
            dtype = "FP32",
            aliasGroup = "c4:input",
            textureBacked = true
        };
        var identityPlan = AexisTextureExecutionPlanner.Compile(
            Parse("Input input 0 1 in\nPermute permute_identity 1 1 in permuted 0=0\nTile tile_identity 1 1 permuted tiled 0=0 1=1\nPacking packing_identity 1 1 tiled packed 0=4 2=1 3=1\nCast cast_identity 1 1 packed casted 0=1 1=1\n"),
            new AexisTextureExecutionPlanRequest
            {
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                targetLayout = AexisTexturePlanLayout.Packed4,
                strict = true,
                inputs = new[] { input }
            });

        Assert.That(identityPlan.nodes.Single(node => node.operatorName == "Permute").usesDescriptorAlias, Is.True);
        Assert.That(identityPlan.nodes.Single(node => node.operatorName == "Tile").usesDescriptorAlias, Is.True);
        Assert.That(identityPlan.nodes.Single(node => node.operatorName == "Packing").usesDescriptorAlias, Is.True);
        Assert.That(identityPlan.nodes.Single(node => node.operatorName == "Cast").usesDescriptorAlias, Is.True);

        var error = Assert.Throws<StrictTextureInferencePlanException>(() => AexisTextureExecutionPlanner.Compile(
            Parse("Input input 0 1 in\nFlatten flatten 1 1 in out\n"),
            new AexisTextureExecutionPlanRequest
            {
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                targetLayout = AexisTexturePlanLayout.Packed4,
                strict = true,
                inputs = new[] { input }
            }));
        var diagnostic = error.Diagnostics.Single(item => item.operatorName == "Flatten");
        Assert.That(diagnostic.code, Is.EqualTo("command-buffer-pack4-profile-rejected"));
        Assert.That(diagnostic.rejectedPaths, Does.Contain("placeholder"));
        Assert.That(diagnostic.rejectedPaths, Does.Contain("materialize-from-buffer"));
    }

    [Test]
    public void CommandBufferPack4_RandomPermuteSliceTileMatchOracle()
    {
        const int width = 3;
        const int height = 2;
        const int channels = 5;
        var repro = new AexisGraphSession(new AexisOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        try
        {
            for (var seed = 1; seed <= 4; seed++)
            {
                var random = new System.Random(seed);
                var input = new float[width * height * channels];
                for (var index = 0; index < input.Length; index++)
                    input[index] = (float)(random.NextDouble() * 20.0 - 10.0);

                using var commandBuffer = new CommandBuffer { name = "NcnnC4Pack4Layout_" + seed };
                using var inputBuffer = new ComputeBuffer(input.Length, sizeof(float), ComputeBufferType.Structured);
                inputBuffer.SetData(input);
                var source = repro.RentTempArray(commandBuffer, width, height, 2, RenderTextureFormat.ARGBFloat);
                var permute = repro.RentTempArray(commandBuffer, height, width, 2, RenderTextureFormat.ARGBFloat);
                var slice = repro.RentTempArray(commandBuffer, width, height, 1, RenderTextureFormat.ARGBFloat);
                var tile = repro.RentTempArray(commandBuffer, width * 2, height * 2, 3, RenderTextureFormat.ARGBFloat);
                var targets = new[]
                {
                    CreateReadbackTarget(height, width), CreateReadbackTarget(height, width),
                    CreateReadbackTarget(width, height),
                    CreateReadbackTarget(width * 2, height * 2), CreateReadbackTarget(width * 2, height * 2), CreateReadbackTarget(width * 2, height * 2)
                };

                try
                {
                    repro.Ops.FillPack4FromBufferCHW(commandBuffer, inputBuffer, width, height, channels, source);
                    repro.Ops.PermutePack4(commandBuffer, source, width, height, channels, new Vector4Int(1, 0, 2, 0), height, width, channels, permute);
                    repro.Ops.SlicePack4(commandBuffer, source, width, height, channels, axis: 2, begin: 1, outW: width, outH: height, outC: 3, output: slice);
                    repro.Ops.TilePack4(
                        commandBuffer,
                        source,
                        new AexisGraphSession.BufferShape(3, width, height, 1, channels),
                        new AexisGraphSession.BufferShape(3, width * 2, height * 2, 1, channels * 2),
                        new Vector4Int(2, 2, 1, 2),
                        tile);

                    CopySlices(commandBuffer, permute, targets, 0, 2);
                    CopySlices(commandBuffer, slice, targets, 2, 1);
                    CopySlices(commandBuffer, tile, targets, 3, 3);
                    repro.ReturnTempArray(commandBuffer, tile);
                    tile = null;
                    repro.ReturnTempArray(commandBuffer, slice);
                    slice = null;
                    repro.ReturnTempArray(commandBuffer, permute);
                    permute = null;
                    repro.ReturnTempArray(commandBuffer, source);
                    source = null;
                    Graphics.ExecuteCommandBuffer(commandBuffer);

                    for (var channel = 0; channel < channels; channel++)
                    {
                        var data = Readback(targets[channel / 4]);
                        for (var y = 0; y < width; y++)
                        {
                            for (var x = 0; x < height; x++)
                            {
                                var expected = ReadInput(input, width, height, channel, y, x);
                                Assert.That(ReadLane(data[y * height + x], channel & 3), Is.EqualTo(expected).Within(1e-5f), "permute seed=" + seed + " c=" + channel + " y=" + y + " x=" + x);
                            }
                        }
                    }

                    var sliceData = Readback(targets[2]);
                    for (var channel = 0; channel < 3; channel++)
                    {
                        for (var y = 0; y < height; y++)
                        {
                            for (var x = 0; x < width; x++)
                            {
                                var expected = ReadInput(input, width, height, channel + 1, x, y);
                                Assert.That(ReadLane(sliceData[y * width + x], channel), Is.EqualTo(expected).Within(1e-5f), "slice seed=" + seed + " c=" + channel + " y=" + y + " x=" + x);
                            }
                        }
                    }

                    for (var channel = 0; channel < channels * 2; channel++)
                    {
                        var data = Readback(targets[3 + channel / 4]);
                        for (var y = 0; y < height * 2; y++)
                        {
                            for (var x = 0; x < width * 2; x++)
                            {
                                var expected = ReadInput(input, width, height, channel % channels, x % width, y % height);
                                Assert.That(ReadLane(data[y * width * 2 + x], channel & 3), Is.EqualTo(expected).Within(1e-5f), "tile seed=" + seed + " c=" + channel + " y=" + y + " x=" + x);
                            }
                        }
                    }
                }
                finally
                {
                    if (tile != null) repro.ReturnTempArray(commandBuffer, tile);
                    if (slice != null) repro.ReturnTempArray(commandBuffer, slice);
                    if (permute != null) repro.ReturnTempArray(commandBuffer, permute);
                    if (source != null) repro.ReturnTempArray(commandBuffer, source);
                    foreach (var target in targets)
                        DestroyRenderTexture(target);
                }
            }
        }
        finally
        {
            repro.Dispose();
        }
    }

    [Test]
    public void C4LayersPublishDescriptorsAndDoNotPublishCmdPlaceholders()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var layers = Path.Combine(root, "Packages", "com.aexis", "Runtime", "Ncnn", "Layers");
        foreach (var file in new[]
        {
            "AexisReshapeLayer.cs", "AexisFlattenLayer.cs", "AexisSqueezeLayer.cs", "AexisExpandDimsLayer.cs",
            "AexisPermuteLayer.cs", "AexisSliceLayer.cs", "AexisTileLayer.cs", "AexisPackingLayer.cs", "AexisCastLayer.cs"
        })
        {
            var source = File.ReadAllText(Path.Combine(layers, file));
            Assert.That(source, Does.Not.Contain("[CmdPlaceholder]"), file);
        }

        Assert.That(File.ReadAllText(Path.Combine(layers, "AexisFlattenLayer.cs")), Does.Contain("ReshapePack4ToPack4"));
        Assert.That(File.ReadAllText(Path.Combine(layers, "AexisPermuteLayer.cs")), Does.Contain("placeholder publication is prohibited"));
        Assert.That(File.ReadAllText(Path.Combine(layers, "AexisSliceLayer.cs")), Does.Contain("placeholder publication is prohibited"));
        Assert.That(File.ReadAllText(Path.Combine(layers, "AexisPackingLayer.cs")), Does.Contain("RequiresPackingTransform"));
    }

    public static void RunBatchValidation()
    {
        var tests = new NcnnC4Pack4LayoutTests();
        tests.AliasMatrix_UsesDescriptorsForViewsAndRejectsTransforms();
        tests.CapabilityMetadata_RecordsC4ProfilesAndRejectsRotaryProductionPlaceholder();
        tests.StrictPlanner_AliasesProvenIdentityAndRejectsUnprovenFlattenTransform();
        tests.CommandBufferPack4_RandomPermuteSliceTileMatchOracle();
        tests.C4LayersPublishDescriptorsAndDoNotPublishCmdPlaceholders();
        Debug.Log("[NcnnC4Pack4LayoutTests] passed");
    }

    private static RenderTexture CreateReadbackTarget(int width, int height)
    {
        var target = new RenderTexture(new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGBFloat, 0)
        {
            dimension = TextureDimension.Tex2D,
            volumeDepth = 1,
            enableRandomWrite = true,
            msaaSamples = 1
        });
        target.Create();
        return target;
    }

    private static void CopySlices(CommandBuffer commandBuffer, ComputeTexture source, RenderTexture[] targets, int targetOffset, int sliceCount)
    {
        for (var slice = 0; slice < sliceCount; slice++)
            commandBuffer.CopyTexture(source.nameID, slice, 0, targets[targetOffset + slice], 0, 0);
    }

    private static Vector4[] Readback(RenderTexture target)
    {
        var request = AsyncGPUReadback.Request(target, 0);
        request.WaitForCompletion();
        Assert.That(request.hasError, Is.False);
        return request.GetData<Vector4>().ToArray();
    }

    private static float ReadInput(float[] input, int width, int height, int channel, int x, int y)
    {
        return input[(channel * height + y) * width + x];
    }

    private static float ReadLane(Vector4 value, int lane)
    {
        return lane == 0 ? value.x : lane == 1 ? value.y : lane == 2 ? value.z : value.w;
    }

    private static void DestroyRenderTexture(RenderTexture texture)
    {
        if (texture == null)
            return;
        if (texture.IsCreated())
            texture.Release();
        UnityEngine.Object.DestroyImmediate(texture);
    }

    private static AexisGraphModel Parse(string body)
    {
        var layerCount = body.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries).Length;
        return NcnnParamParser.Parse("7767517\n" + layerCount + " 2\n" + body);
    }
}
#endif
