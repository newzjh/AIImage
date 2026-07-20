#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using Aexis.Ncnn;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class NcnnC2CdhwCmdPack4Tests
{
    [Test]
    public void CapabilityMetadata_DeclaresCdhwProfilesAndInterpAlignCorners()
    {
        var capabilities = NcnnOperatorCapabilities.CreateDocument().operators;
        foreach (var operatorName in new[] { "Convolution3D", "Deconvolution3D", "Pooling3D", "Interp" })
        {
            var capability = capabilities.Single(item => item.operatorName == operatorName);
            Assert.That(capability.status, Is.EqualTo(NcnnOperatorCapabilityStatus.Partial));
            Assert.That(capability.profiles, Is.Not.Empty);
            var profile = capability.profiles.Single(item => item.backend == NcnnOperatorCapabilityBackend.CommandBuffer);
            Assert.That(profile.layouts, Does.Contain("CDHW"));
            Assert.That(profile.layouts, Does.Contain("Packed4"));
            Assert.That(profile.shapeProfile, Does.Contain("Texture2DArray"));
        }

        var interp = capabilities.Single(item => item.operatorName == "Interp");
        Assert.That(interp.profiles.Single().supportedParameters, Does.Contain("align_corners=0|1"));
    }

    [Test]
    public void StrictPlanner_RejectsCdhwInterpWithoutRuntimeProof()
    {
        var model = Parse("Input in0 0 1 in\nInterp resize3d 1 1 in out 0=2 1=1.5 2=1.5 7=1.5\n");
        var error = Assert.Throws<StrictTextureInferencePlanException>(() => NcnnTextureExecutionPlanner.Compile(
            model,
            new NcnnTextureExecutionPlanRequest
            {
                targetBackend = NcnnOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                targetLayout = NcnnTexturePlanLayout.Packed4,
                strict = true,
                inputs = new[]
                {
                    new NcnnTexturePlanTensorDescriptor
                    {
                        blob = "in",
                        logicalShape = new[] { 4, 2, 2, 2, 3 },
                        storageShape = new[] { 4, 2, 2, 2, 3 },
                        layout = NcnnTexturePlanLayout.Packed4,
                        dtype = "FP32",
                        aliasGroup = "test:in",
                        textureBacked = true
                    }
                }
            }));

        var diagnostic = error.Diagnostics.Single(item => item.operatorName == "Interp");
        Assert.That(diagnostic.code, Is.EqualTo("command-buffer-pack4-profile-rejected"));
        Assert.That(diagnostic.rejectedPaths, Does.Contain("materialize-from-buffer"));
        Assert.That(diagnostic.rejectedPaths, Does.Contain("placeholder"));
    }

    [Test]
    public void CommandBufferPack4_CdhwInterpAlignCornersMatchesReference()
    {
        var input = new[]
        {
            0f, 1f,
            10f, 11f,
            100f, 101f,
            110f, 111f
        };
        var repro = new NcnnGraphSession(new NcnnOps()) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
        var persistentOutputs = new RenderTexture[3];
        for (var slice = 0; slice < persistentOutputs.Length; slice++)
        {
            persistentOutputs[slice] = new RenderTexture(new RenderTextureDescriptor(3, 3, RenderTextureFormat.ARGBFloat, 0)
            {
                dimension = TextureDimension.Tex2D,
                volumeDepth = 1,
                enableRandomWrite = true,
                msaaSamples = 1
            });
            persistentOutputs[slice].Create();
        }

        using var commandBuffer = new CommandBuffer { name = "NcnnC2CdhwInterpAlignCorners" };
        using var inputBuffer = new ComputeBuffer(input.Length, sizeof(float), ComputeBufferType.Structured);
        inputBuffer.SetData(input);
        var source = repro.RentTempArray(commandBuffer, 2, 2, 2, RenderTextureFormat.ARGBFloat);
        var destination = repro.RentTempArray(commandBuffer, 3, 3, 3, RenderTextureFormat.ARGBFloat);
        try
        {
            repro.Ops.FillPack4FromBufferCDHW(commandBuffer, inputBuffer, 2, 2, 2, 1, source);
            repro.Ops.InterpPack4CDHW(
                commandBuffer,
                source,
                2,
                2,
                2,
                1,
                3,
                3,
                3,
                1,
                1.5f,
                1.5f,
                1.5f,
                resizeType: 2,
                alignCorners: true,
                output: destination);
            for (var slice = 0; slice < 3; slice++)
                commandBuffer.CopyTexture(destination.nameID, slice, 0, persistentOutputs[slice], 0, 0);
            repro.ReturnTempArray(commandBuffer, destination);
            destination = null;
            repro.ReturnTempArray(commandBuffer, source);
            source = null;
            Graphics.ExecuteCommandBuffer(commandBuffer);

            for (var z = 0; z < 3; z++)
            {
                var readback = AsyncGPUReadback.Request(persistentOutputs[z], 0);
                readback.WaitForCompletion();
                Assert.That(readback.hasError, Is.False, "slice=" + z);
                var values = readback.GetData<Vector4>();
                for (var y = 0; y < 3; y++)
                {
                    for (var x = 0; x < 3; x++)
                    {
                        var expected = x * 0.5f + y * 5f + z * 50f;
                        Assert.That(values[y * 3 + x].x, Is.EqualTo(expected).Within(1e-4f), "x=" + x + " y=" + y + " z=" + z);
                    }
                }
            }
        }
        finally
        {
            if (destination != null)
                repro.ReturnTempArray(commandBuffer, destination);
            if (source != null)
                repro.ReturnTempArray(commandBuffer, source);
            foreach (var persistentOutput in persistentOutputs)
            {
                if (persistentOutput == null)
                    continue;
                if (persistentOutput.IsCreated())
                    persistentOutput.Release();
                UnityEngine.Object.DestroyImmediate(persistentOutput);
            }
            repro.Dispose();
        }
    }

    [Test]
    public void CdhwCommandLayersPublishDescriptorsAndAvoidCmdPlaceholders()
    {
        var root = Path.GetDirectoryName(Application.dataPath);
        var layersDirectory = Path.Combine(root, "Packages", "com.aexis", "Runtime", "Ncnn", "Layers");
        foreach (var file in new[]
        {
            "NcnnConvolution3DLayerRepro.cs",
            "NcnnDeconvolution3DLayerRepro.cs",
            "NcnnPooling3DLayerRepro.cs"
        })
        {
            var source = File.ReadAllText(Path.Combine(layersDirectory, file));
            Assert.That(source, Does.Contain("GetCmdTensorContract"), file);
            Assert.That(source, Does.Contain("CreateCmdTensorRef"), file);
        }

        var interp = File.ReadAllText(Path.Combine(layersDirectory, "NcnnInterpLayerRepro.cs"));
        Assert.That(interp, Does.Contain("InterpPack4CDHW("));
        Assert.That(interp, Does.Contain("alignCorners"));
        Assert.That(interp, Does.Contain("dims == 4"));
    }

    public static void RunBatchValidation()
    {
        var tests = new NcnnC2CdhwCmdPack4Tests();
        tests.CapabilityMetadata_DeclaresCdhwProfilesAndInterpAlignCorners();
        tests.StrictPlanner_RejectsCdhwInterpWithoutRuntimeProof();
        tests.CommandBufferPack4_CdhwInterpAlignCornersMatchesReference();
        tests.CdhwCommandLayersPublishDescriptorsAndAvoidCmdPlaceholders();
        Debug.Log("[NcnnC2CdhwCmdPack4Tests] passed");
    }

    private static NcnnParamModel Parse(string body)
    {
        var layerCount = body.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries).Length;
        return NcnnParamParser.Parse("7767517\n" + layerCount + " 2\n" + body);
    }
}
#endif
