using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Aexis;
using Aexis.Execution;
using Aexis.Ncnn;
using Aexis.Onnx;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Tests.Editor
{
    public sealed class AexisP0ProductionBaselineTests
    {
        [Test]
        public void CommandBufferMissingImplementation_FailsInsteadOfPublishingPlaceholder()
        {
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops);
            using var commandBuffer = new CommandBuffer();
            var layer = new AexisGraphModel.Layer
            {
                name = "missing_cmd",
                typeName = "MissingCmd",
                bottomNames = new[] { "input" },
                topNames = new[] { "output" }
            };
            var context = new AexisLayerCommandBufferContext
            {
                commandBuffer = commandBuffer,
                blobs = new Dictionary<string, AexisGraphSession.CmdTensorRef>(StringComparer.Ordinal),
                shapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal),
                remaining = new Dictionary<string, int>(StringComparer.Ordinal)
            };

            var exception = Assert.Throws<NotSupportedException>(() => new MissingCommandBufferLayer().ExecuteCommandBuffer(session, layer, context));

            Assert.That(exception.Message, Does.Contain("rejected_fallback=placeholder"));
            Assert.That(context.blobs, Is.Empty);
        }

        [Test]
        public void CommandBufferExecutionLayers_DoNotPublishPlaceholderOutputs()
        {
            var package = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(AexisGraphSession).Assembly);
            Assert.That(package, Is.Not.Null, "Unable to resolve the com.aexis package path.");
            var layersPath = Path.Combine(package.resolvedPath, "Runtime", "Execution", "Layers");
            var offenders = Directory.GetFiles(layersPath, "*.cs", SearchOption.AllDirectories)
                .Where(path => File.ReadAllText(path).IndexOf("PublishCmdPlaceholder(", StringComparison.Ordinal) >= 0)
                .Select(path => path.Substring(package.resolvedPath.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Replace('\\', '/'))
                .ToArray();

            Assert.That(offenders, Is.Empty,
                "Production execution layers must reject unsupported CommandBuffer profiles instead of publishing blank tensors: "
                + string.Join(", ", offenders));
        }

        [Test]
        public void PackageBatchSmoke_CoversP0OnnxCompilationAndNcnnAliasCompatibility()
        {
            Aexis.Editor.AexisPackageValidation.RunBatchSmoke();
        }

        [Test]
        public void OnnxLowering_MapsCoreGraphAndInfersBroadcastShape()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 3, 4, 4));
            model.graph.inputs.Add(Value("bias", TensorDataType.Float32, 1, 3, 1, 1));
            model.graph.nodes.Add(Node("add", "Add", new[] { "x", "bias" }, new[] { "sum" }));
            model.graph.nodes.Add(Node("relu", "Relu", new[] { "sum" }, new[] { "output" }));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(result.graph.layers, Has.Count.EqualTo(4));
            Assert.That(result.graph.layers[2].typeName, Is.EqualTo("BinaryOp"));
            Assert.That(result.graph.layers[2].GetInt(0), Is.EqualTo(0));
            Assert.That(result.graph.layers[3].typeName, Is.EqualTo("ReLU"));
            Assert.That(Find(result, "output").shape, Is.EqualTo(new long[] { 1, 3, 4, 4 }));
        }

        [Test]
        public void OnnxLowering_NcnnAndSentisUnaryExtensionsUseStrictPack4CommandBufferAbi()
        {
            RequireArgbFloatCompute();
            var model = new OnnxModel { opset = 13 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 5, 1, 1));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 1, 5, 1, 1));
            model.graph.nodes.Add(Node("relu6", "Relu6", new[] { "x" }, new[] { "clamped" }));
            model.graph.nodes.Add(Node("square", "Square", new[] { "clamped" }, new[] { "squared" }));
            model.graph.nodes.Add(Node("rsqrt", "Rsqrt", new[] { "squared" }, new[] { "y" }));

            var lowered = AexisOnnxGraphLowering.Lower(model);
            Assert.That(lowered.IsEligible, Is.True, DescribeLowering(lowered));
            var clipLayer = lowered.graph.layers.Single(layer => string.Equals(layer.typeName, "Clip", StringComparison.Ordinal));
            Assert.That(clipLayer.GetFloat(0), Is.EqualTo(0f));
            Assert.That(clipLayer.GetFloat(1), Is.EqualTo(6f));
            var unaryLayers = lowered.graph.layers.Where(layer => string.Equals(layer.typeName, "UnaryOp", StringComparison.Ordinal)).ToArray();
            Assert.That(unaryLayers.Select(layer => layer.typeName), Is.EqualTo(new[] { "UnaryOp", "UnaryOp" }));
            Assert.That(unaryLayers.Select(layer => layer.GetInt(0)), Is.EqualTo(new[] { 4, 6 }),
                "The lowering must retain ncnn UnaryOp::Operation_SQUARE and Operation_RSQRT ABI values.");

            var compiled = AexisOnnxGraphCompiler.Compile(lowered);
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            compiled.LoadInto(session);
            var report = session.AnalyzeLoadedModelPreflight(StrictTextureRequest(new[] { Texture("x", 3, 1, 1, 1, 5) }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.nodes.Where(node => !string.Equals(node.executionPath, "external-pack4-input", StringComparison.Ordinal)).Select(node => node.executionPath),
                Is.EqualTo(new[] { "command-buffer-pack4:pointwise-clip", "command-buffer-pack4:unary", "command-buffer-pack4:unary" }), DescribeReport(report));

            using var commandBuffer = new CommandBuffer { name = "AexisNcnnUnaryExtensionsPack4Golden" };
            var inputUpload = CreatePack4Array(new[]
            {
                1f, 2f, 7f, 8f,
                3f, -7f, 99f, -13f
            }, 1, 1, 2);
            var outputReadback = CreatePack4Target(1, 1, 2);
            ComputeTexture input = null;
            try
            {
                input = session.RentTempArray(commandBuffer, 1, 1, 2, RenderTextureFormat.ARGBFloat);
                commandBuffer.CopyTexture(inputUpload, 0, 0, input.nameID, 0, 0);
                commandBuffer.CopyTexture(inputUpload, 1, 0, input.nameID, 1, 0);
                using (var execution = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["x"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["x"] = new AexisGraphSession.BufferShape(3, 1, 1, 1, 5)
                    },
                    new[] { "y" }))
                {
                    commandBuffer.CopyTexture(execution.GetTexture("y").nameID, 0, 0, outputReadback, 0, 0);
                    commandBuffer.CopyTexture(execution.GetTexture("y").nameID, 1, 0, outputReadback, 1, 0);
                }
                session.ReturnTempArray(commandBuffer, input);
                input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.Zero);
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                Assert.That(ReadPack4Slice(outputReadback, 0), Is.EqualTo(new[] { 1f, 0.5f, 1f / 6f, 1f / 6f }).Within(2e-6f));
                Assert.That(ReadPack4Slice(outputReadback, 1), Is.EqualTo(new[] { 1f / 3f, 0f, 0f, 0f }).Within(2e-6f));
            }
            finally
            {
                if (input != null)
                    session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [TestCase("none", false)]
        [TestCase("tanh", true)]
        public void OnnxGelu_UsesDeclaredApproximationOnStrictPack4CommandBuffer(string approximate, bool fast)
        {
            RequireArgbFloatCompute();
            var model = new OnnxModel { opset = 20 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 5, 1, 1));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 1, 5, 1, 1));
            var gelu = Node("gelu", "Gelu", new[] { "x" }, new[] { "y" });
            gelu.attributes["approximate"] = StringAttribute(approximate);
            model.graph.nodes.Add(gelu);

            var lowered = AexisOnnxGraphLowering.Lower(model);
            Assert.That(lowered.IsEligible, Is.True, DescribeLowering(lowered));
            var layer = lowered.graph.layers.Single(candidate => string.Equals(candidate.typeName, "GELU", StringComparison.Ordinal));
            Assert.That(layer.GetInt(0), Is.EqualTo(fast ? 1 : 0));
            Assert.That(layer.GetString("onnx.approximate"), Is.EqualTo(approximate));

            var compiled = AexisOnnxGraphCompiler.Compile(lowered);
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            compiled.LoadInto(session);
            var report = session.AnalyzeLoadedModelPreflight(StrictTextureRequest(new[] { Texture("x", 3, 1, 1, 1, 5) }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.nodes.Single(candidate => candidate.layer == "gelu").executionPath,
                Is.EqualTo("command-buffer-pack4:gelu"));

            using var commandBuffer = new CommandBuffer { name = "AexisOnnxGeluPack4Golden" };
            var inputUpload = CreatePack4Array(new[]
            {
                -2f, -1f, 0f, 1f,
                2f, 0f, 0f, 0f
            }, 1, 1, 2);
            var outputReadback = CreatePack4Target(1, 1, 2);
            ComputeTexture input = null;
            try
            {
                input = session.RentTempArray(commandBuffer, 1, 1, 2, RenderTextureFormat.ARGBFloat);
                commandBuffer.CopyTexture(inputUpload, 0, 0, input.nameID, 0, 0);
                commandBuffer.CopyTexture(inputUpload, 1, 0, input.nameID, 1, 0);
                using (var execution = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["x"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["x"] = new AexisGraphSession.BufferShape(3, 1, 1, 1, 5)
                    },
                    new[] { "y" }))
                {
                    commandBuffer.CopyTexture(execution.GetTexture("y").nameID, 0, 0, outputReadback, 0, 0);
                    commandBuffer.CopyTexture(execution.GetTexture("y").nameID, 1, 0, outputReadback, 1, 0);
                }
                session.ReturnTempArray(commandBuffer, input);
                input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                var expected = fast
                    ? new[] { -0.0454023f, -0.1588080f, 0f, 0.8411920f, 1.9545977f }
                    : new[] { -0.0455003f, -0.1586553f, 0f, 0.8413447f, 1.9544997f };
                Assert.That(ReadPack4Slice(outputReadback, 0), Is.EqualTo(expected.Take(4).ToArray()).Within(4e-5f));
                Assert.That(ReadPack4Slice(outputReadback, 1), Is.EqualTo(new[] { expected[4], 0f, 0f, 0f }).Within(4e-5f));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.Zero);
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
            }
            finally
            {
                if (input != null)
                    session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void OnnxGelu_RejectsUnknownApproximationBeforeStrictExecution()
        {
            var model = new OnnxModel { opset = 20 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 4, 1, 1));
            var gelu = Node("gelu", "Gelu", new[] { "x" }, new[] { "y" });
            gelu.attributes["approximate"] = StringAttribute("erf_fast");
            model.graph.nodes.Add(gelu);

            var lowered = AexisOnnxGraphLowering.Lower(model);

            Assert.That(lowered.IsEligible, Is.False, DescribeLowering(lowered));
            Assert.That(lowered.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-gelu-approximation" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_BoundedHighRankArenaKeepsLogicalRankAndPlansPack4Relu()
        {
            RequireArgbFloatCompute();
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 2, 2, 2, 2, 2));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 1, 2, 2, 2, 2, 2));
            model.graph.nodes.Add(Node("relu", "Relu", new[] { "x" }, new[] { "y" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                enableBoundedHighRankArenaLowering = true,
                maximumBoundedArenaRank = 8
            });

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(Find(result, "x").shape, Is.EqualTo(new long[] { 1, 2, 2, 2, 2, 2 }));
            Assert.That(Find(result, "x").runtimeShape, Is.EqualTo(new long[] { 4, 2, 2, 2 }));
            Assert.That(Find(result, "y").shape, Is.EqualTo(new long[] { 1, 2, 2, 2, 2, 2 }));
            Assert.That(Find(result, "y").runtimeShape, Is.EqualTo(new long[] { 4, 2, 2, 2 }));

            var compiled = AexisOnnxGraphCompiler.Compile(result);
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            compiled.LoadInto(session);
            var report = session.AnalyzeLoadedModelPreflight(StrictTextureRequest(new[] { Texture("x", 4, 2, 2, 2, 4) }));

            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.nodes.Single(candidate => candidate.layer == "relu").executionPath,
                Does.Contain("command-buffer-pack4:relu"), DescribeReport(report));

            using var commandBuffer = new CommandBuffer { name = "AexisBoundedHighRankReluGolden" };
            var inputUpload = CreatePack4Array(new[]
            {
                -1f, 2f, -3f, 4f, 5f, -6f, 7f, -8f, -9f, 10f, 11f, -12f, 13f, 14f, -15f, 16f,
                17f, -18f, 19f, -20f, -21f, 22f, -23f, 24f, 25f, -26f, 27f, 28f, -29f, 30f, 31f, -32f
            }, 2, 2, 2);
            var outputReadback = CreatePack4Target(2, 2, 2);
            ComputeTexture input = null;
            try
            {
                input = session.RentTempArray(commandBuffer, 2, 2, 2, RenderTextureFormat.ARGBFloat);
                commandBuffer.CopyTexture(inputUpload, 0, 0, input.nameID, 0, 0);
                commandBuffer.CopyTexture(inputUpload, 1, 0, input.nameID, 1, 0);
                using (var execution = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["x"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["x"] = new AexisGraphSession.BufferShape(4, 2, 2, 2, 4)
                    },
                    new[] { "y" }))
                {
                    commandBuffer.CopyTexture(execution.GetTexture("y").nameID, 0, 0, outputReadback, 0, 0);
                    commandBuffer.CopyTexture(execution.GetTexture("y").nameID, 1, 0, outputReadback, 1, 0);
                }
                session.ReturnTempArray(commandBuffer, input); input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Single(candidate => candidate.layer == "relu").executionPath,
                    Does.Contain("command-buffer-pack4:relu"));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.Zero);
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                Assert.That(ReadPack4Slice(outputReadback, 0), Is.EqualTo(new[]
                {
                    0f, 2f, 0f, 4f, 5f, 0f, 7f, 0f, 0f, 10f, 11f, 0f, 13f, 14f, 0f, 16f
                }).Within(1e-6f));
                Assert.That(ReadPack4Slice(outputReadback, 1), Is.EqualTo(new[]
                {
                    17f, 0f, 19f, 0f, 0f, 22f, 0f, 24f, 25f, 0f, 27f, 28f, 0f, 30f, 31f, 0f
                }).Within(1e-6f));
            }
            finally
            {
                if (input != null) session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void OnnxLowering_BoundedHighRankArenaRejectsAxisAwareConvolution()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 2, 2, 2, 2, 2));
            model.graph.initializers["weight"] = Tensor("weight", TensorDataType.Float32, 2, 2, 1, 1);
            model.graph.nodes.Add(Node("conv", "Conv", new[] { "x", "weight" }, new[] { "y" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                enableBoundedHighRankArenaLowering = true
            });

            Assert.That(result.IsEligible, Is.False, DescribeLowering(result));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-bounded-high-rank-profile" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_NormalizesStaticConvIntoNcnnTextureContract()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 3, 8, 8));
            model.graph.initializers["weight"] = Tensor("weight", TensorDataType.Float32, 4, 3, 3, 3);
            model.graph.initializers["bias"] = Tensor("bias", TensorDataType.Float32, 4);
            var conv = Node("conv", "Conv", new[] { "x", "weight", "bias" }, new[] { "y" });
            conv.attributes["strides"] = Ints(1, 1);
            conv.attributes["pads"] = Ints(1, 1, 1, 1);
            model.graph.nodes.Add(conv);

            var result = AexisOnnxGraphLowering.Lower(model);
            var layer = result.graph.layers[1];

            Assert.That(result.IsEligible, Is.True);
            Assert.That(layer.bottomNames, Is.EqualTo(new[] { "x" }));
            Assert.That(layer.GetInt(0), Is.EqualTo(4));
            Assert.That(layer.GetInt(1), Is.EqualTo(3));
            Assert.That(layer.GetInt(11), Is.EqualTo(3));
            Assert.That(layer.GetInt(5), Is.EqualTo(1));
            Assert.That(layer.GetInt(6), Is.EqualTo(108));
            Assert.That(layer.stringParams["onnx.weight"], Is.EqualTo("weight"));
        }

        [Test]
        public void NcnnNormalizeAndPriorBox_LoadedProfilesRequireAndPlanTextureNativeContracts()
        {
            var normalize = SentisLayer("Normalize", new[] { "data" }, new[] { "normalized" });
            normalize.intParams[0] = "1";
            normalize.intParams[1] = "0";
            normalize.intParams[2] = "0.0001";
            normalize.intParams[3] = "3";
            normalize.intParams[4] = "1";
            normalize.intParams[9] = "0";
            var normalizeReport = AnalyzeLoadedLayer(
                normalize,
                StrictTextureRequest(new[] { Texture("data", 3, 2, 2, 1, 3) }),
                1f, 0.5f, 2f);

            Assert.That(normalizeReport.strictEligible, Is.True, DescribeReport(normalizeReport));
            Assert.That(normalizeReport.texturePlan.nodes.Single().executionPath,
                Is.EqualTo("command-buffer-pack4:normalize"));

            var priorBox = SentisLayer("PriorBox", new[] { "feature", "image" }, new[] { "priors" });
            priorBox.intParams[-23300] = "2";
            priorBox.intParams[-23302] = "2";
            priorBox.intParams[3] = "0.1";
            priorBox.intParams[4] = "0.1";
            priorBox.intParams[5] = "0.2";
            priorBox.intParams[6] = "0.2";
            priorBox.intParams[7] = "0";
            priorBox.intParams[8] = "0";
            priorBox.intParams[9] = "-233";
            priorBox.intParams[10] = "-233";
            priorBox.intParams[11] = "-233";
            priorBox.intParams[12] = "-233";
            var priorReport = AnalyzeLoadedLayer(
                priorBox,
                StrictTextureRequest(new[]
                {
                    Texture("feature", 3, 2, 2, 1, 4),
                    Texture("image", 3, 8, 8, 1, 3)
                }));

            Assert.That(priorReport.strictEligible, Is.True, DescribeReport(priorReport));
            Assert.That(priorReport.texturePlan.nodes.Single().executionPath,
                Is.EqualTo("command-buffer-linearmat:priorbox"));
            Assert.That(priorReport.texturePlan.nodes.Single().outputs[0].logicalShape,
                Is.EqualTo(new[] { 2, 32, 2, 1, 1 }));
            Assert.That(priorReport.texturePlan.nodes.Single().outputs[0].dtype, Is.EqualTo("FP32"));
        }

        [Test]
        public void NcnnCoreRegistry_ExposesOnlyCommandBufferProfilesOrExplicitDescriptorAliases()
        {
            // Snapshot of ref/ncnn-master/src/CMakeLists.txt ncnn_add_layer entries.
            // The test keeps package capability metadata from claiming registry
            // completeness while silently dropping a production operator.
            var layers = new[]
            {
                "AbsVal", "ArgMax", "BatchNorm", "Bias", "BNLL", "Concat", "Convolution", "Crop", "Deconvolution", "Dropout", "Eltwise", "ELU", "Embed", "Exp", "Flatten", "InnerProduct", "Input", "Log", "LRN", "MemoryData", "MVN", "Pooling", "Power", "PReLU", "Proposal", "Reduction", "ReLU", "Reshape", "ROIPooling", "Scale", "Sigmoid", "Slice", "Softmax", "Split", "SPP", "TanH", "Threshold", "Tile", "RNN", "LSTM", "BinaryOp", "UnaryOp", "ConvolutionDepthWise", "Padding", "Squeeze", "ExpandDims", "Normalize", "Permute", "PriorBox", "DetectionOutput", "Interp", "DeconvolutionDepthWise", "ShuffleChannel", "InstanceNorm", "Clip", "Reorg", "YoloDetectionOutput", "Quantize", "Dequantize", "Yolov3DetectionOutput", "PSROIPooling", "ROIAlign", "Packing", "Requantize", "Cast", "HardSigmoid", "SELU", "HardSwish", "Noop", "PixelShuffle", "DeepCopy", "Mish", "StatisticsPooling", "Swish", "Gemm", "GroupNorm", "LayerNorm", "Softplus", "GRU", "MultiHeadAttention", "GELU", "Convolution1D", "Pooling1D", "ConvolutionDepthWise1D", "Convolution3D", "ConvolutionDepthWise3D", "Pooling3D", "MatMul", "Deconvolution1D", "DeconvolutionDepthWise1D", "Deconvolution3D", "DeconvolutionDepthWise3D", "Einsum", "DeformableConv2D", "GLU", "Fold", "Unfold", "GridSample", "CumulativeSum", "CopyTo", "Erf", "Diag", "CELU", "Shrink", "RMSNorm", "Spectrogram", "InverseSpectrogram", "Flip", "SDPA", "RotaryEmbed"
            };
            var aliases = new HashSet<string>(StringComparer.Ordinal)
            {
                "Input", "Split", "Dropout", "Noop", "DeepCopy"
            };

            foreach (var layer in layers)
            {
                Assert.That(AexisLayerFactory.IsRegistered(layer), Is.True, layer + " must remain registered.");
                Assert.That(AexisOperatorCapabilities.TryGet(layer, out var capability), Is.True, layer + " must expose a capability document.");
                if (aliases.Contains(layer))
                {
                    Assert.That(capability.status, Is.EqualTo(AexisOperatorCapabilityStatus.AliasOnly), layer);
                    continue;
                }

                Assert.That(capability.profiles.Any(profile =>
                    string.Equals(profile.backend, AexisOperatorCapabilityBackend.CommandBuffer, StringComparison.Ordinal)
                    && profile.minInputs >= 0 && profile.maxOutputs != 0), Is.True,
                    layer + " must retain a CommandBuffer production profile.");
            }
        }

        [Test]
        public void OnnxLowering_ReportsUnsupportedOperatorBeforeRuntime()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1));
            model.graph.nodes.Add(Node("control_flow", "Loop", new[] { "x" }, new[] { "out" }));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.False, DescribeLowering(result));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "unsupported-operator" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_BoundedIfInlinesTheProvenBranchIntoThePack4Graph()
        {
            var model = new OnnxModel { opset = 13 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 1));
            model.graph.initializers["condition"] = IntTensor("condition", new[] { 1 });

            var thenBranch = new OnnxGraph { name = "then" };
            thenBranch.nodes.Add(Node("then_identity", "Identity", new[] { "x" }, new[] { "then_y" }));
            thenBranch.outputs.Add(Value("then_y", TensorDataType.Float32, 1));
            var elseBranch = new OnnxGraph { name = "else" };
            elseBranch.nodes.Add(Node("else_identity", "Identity", new[] { "x" }, new[] { "else_y" }));
            elseBranch.outputs.Add(Value("else_y", TensorDataType.Float32, 1));

            var control = Node("if", "If", new[] { "condition" }, new[] { "y" });
            control.attributes["then_branch"] = new OnnxAttribute { type = 5, graph = thenBranch };
            control.attributes["else_branch"] = new OnnxAttribute { type = 5, graph = elseBranch };
            model.graph.nodes.Add(control);

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions { enableBoundedControlFlowLowering = true });

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(result.graph.layers.Any(layer => layer.typeName == "If"), Is.False);
            Assert.That(result.graph.layers.Any(layer => layer.topNames.Contains("y")), Is.True);
        }

        [Test]
        public void OnnxLowering_BoundedLoopUnrollsFixedTripCountWithoutRuntimeControlFlow()
        {
            var model = new OnnxModel { opset = 13 };
            // Exercise the actual LinearMat binary-exact CommandBuffer kernel.
            // Scalar x scalar is deliberately not claimed as a GPU profile;
            // the scalar kernel is a broadcast kernel rather than an exact
            // two-input operator.
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 1));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 1, 1));
            model.graph.initializers["trip"] = Int64Tensor("trip", new long[] { 2 });
            model.graph.initializers["condition"] = IntTensor("condition", new[] { 1 });

            var body = new OnnxGraph { name = "body" };
            body.inputs.Add(Value("iteration", TensorDataType.Int32));
            body.inputs.Add(Value("continue_in", TensorDataType.Int32));
            body.inputs.Add(Value("carry", TensorDataType.Float32, 1, 1));
            body.initializers["body_continue"] = IntTensor("body_continue", new[] { 1 });
            body.initializers["one"] = FloatTensor("one", new[] { 1f }, 1, 1);
            body.nodes.Add(Node("increment", "Add", new[] { "carry", "one" }, new[] { "carry_out" }));
            body.outputs.Add(Value("body_continue", TensorDataType.Int32));
            body.outputs.Add(Value("carry_out", TensorDataType.Float32, 1, 1));

            var loop = Node("loop", "Loop", new[] { "trip", "condition", "x" }, new[] { "y" });
            loop.attributes["body"] = new OnnxAttribute { type = 5, graph = body };
            model.graph.nodes.Add(loop);

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                enableBoundedControlFlowLowering = true,
                maximumStaticLoopIterations = 2
            });

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(result.graph.layers.Any(layer => layer.typeName == "Loop"), Is.False);
            Assert.That(result.graph.layers.Count(layer => layer.typeName == "BinaryOp"), Is.EqualTo(2));
            Assert.That(Find(result, "y").shape, Is.EqualTo(new long[] { 1, 1 }));

            var compiled = AexisOnnxGraphCompiler.Compile(result);
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
            compiled.LoadInto(session);
            var report = session.AnalyzeLoadedModelPreflight(StrictTextureRequest(new[] { LinearTexture("x", 2, 1, 1, 1, 1) }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.nodes.All(candidate => candidate.strictEligible), Is.True, DescribeReport(report));
        }

        [Test]
        public void OnnxLowering_BoundedScanUnrollsStaticAxisZeroIntoSliceAndConcat()
        {
            var model = new OnnxModel { opset = 13 };
            model.graph.inputs.Add(Value("state", TensorDataType.Float32, 1, 1));
            model.graph.inputs.Add(Value("values", TensorDataType.Float32, 2, 1));
            model.graph.outputs.Add(Value("final_state", TensorDataType.Float32, 1, 1));
            model.graph.outputs.Add(Value("scanned", TensorDataType.Float32, 2, 1));

            var body = new OnnxGraph { name = "scan_body" };
            body.inputs.Add(Value("state_in", TensorDataType.Float32, 1, 1));
            body.inputs.Add(Value("value_in", TensorDataType.Float32, 1, 1));
            body.nodes.Add(Node("add", "Add", new[] { "state_in", "value_in" }, new[] { "state_out" }));
            body.nodes.Add(Node("expose", "Identity", new[] { "state_out" }, new[] { "scan_out" }));
            body.outputs.Add(Value("state_out", TensorDataType.Float32, 1, 1));
            body.outputs.Add(Value("scan_out", TensorDataType.Float32, 1, 1));

            var scan = Node("scan", "Scan", new[] { "state", "values" }, new[] { "final_state", "scanned" });
            scan.attributes["body"] = new OnnxAttribute { type = 5, graph = body };
            scan.attributes["num_scan_inputs"] = Int(1);
            model.graph.nodes.Add(scan);

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                enableBoundedControlFlowLowering = true,
                maximumStaticScanSteps = 2
            });

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(result.graph.layers.Any(layer => layer.typeName == "Scan"), Is.False);
            Assert.That(result.graph.layers.Count(layer => layer.typeName == "Crop"), Is.EqualTo(2));
            Assert.That(result.graph.layers.Any(layer => layer.typeName == "Concat" && layer.topNames.Contains("scanned")), Is.True);
        }

        [Test]
        public void OnnxLowering_BoundedSequenceAndOptionalBecomeTextureNativeViews()
        {
            var model = new OnnxModel { opset = 13 };
            model.graph.inputs.Add(Value("a", TensorDataType.Float32, 1, 2));
            model.graph.inputs.Add(Value("b", TensorDataType.Float32, 1, 2));
            model.graph.outputs.Add(Value("element", TensorDataType.Float32, 1, 2));
            model.graph.initializers["index"] = IntTensor("index", new[] { 1 });
            model.graph.nodes.Add(Node("construct", "SequenceConstruct", new[] { "a", "b" }, new[] { "sequence" }));
            model.graph.nodes.Add(Node("at", "SequenceAt", new[] { "sequence", "index" }, new[] { "selected" }));
            model.graph.nodes.Add(Node("optional", "Optional", new[] { "selected" }, new[] { "wrapped" }));
            model.graph.nodes.Add(Node("get", "OptionalGetElement", new[] { "wrapped" }, new[] { "element" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions { enableBoundedControlFlowLowering = true });

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(result.graph.layers.Any(layer => layer.typeName == "SequenceConstruct" || layer.typeName == "SequenceAt" || layer.typeName == "Optional"), Is.False);
            Assert.That(result.graph.layers.Any(layer => layer.typeName == "Concat" && layer.topNames.Contains("sequence")), Is.True);
            Assert.That(Find(result, "element").shape, Is.EqualTo(new long[] { 1, 2 }));
        }

        [Test]
        public void OnnxLowering_RejectsUnsupportedOpsetAndOperatorBeforeIntroduction()
        {
            var unsupported = new OnnxModel { opset = 21 };
            unsupported.graph.inputs.Add(Value("x", TensorDataType.Float32, 1));
            unsupported.graph.nodes.Add(Node("identity", "Identity", new[] { "x" }, new[] { "y" }));
            unsupported.graph.outputs.Add(Value("y", TensorDataType.Float32, 1));

            var early = new OnnxModel { opset = 13 };
            early.graph.inputs.Add(Value("x", TensorDataType.Float32, 2, 2));
            early.graph.nodes.Add(Node("trilu", "Trilu", new[] { "x" }, new[] { "y" }));
            early.graph.outputs.Add(Value("y", TensorDataType.Float32, 2, 2));

            var unsupportedResult = AexisOnnxGraphLowering.Lower(unsupported);
            var earlyResult = AexisOnnxGraphLowering.Lower(early);

            Assert.That(unsupportedResult.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "unsupported-opset" && diagnostic.blocking));
            Assert.That(earlyResult.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "operator-before-introduction-opset" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_RejectsInvalidArityAndMissingRequiredInput()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1));
            model.graph.nodes.Add(Node("add", "Add", new[] { "x", string.Empty }, new[] { "y" }));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 1));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.False, DescribeLowering(result));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "missing-required-input" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_RejectsDuplicateNamesAndTensorProducers()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1));
            model.graph.nodes.Add(Node("duplicate", "Identity", new[] { "x" }, new[] { "value" }));
            model.graph.nodes.Add(Node("duplicate", "Identity", new[] { "x" }, new[] { "value" }));
            model.graph.outputs.Add(Value("value", TensorDataType.Float32, 1));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "duplicate-node-name" && diagnostic.blocking));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "duplicate-tensor-producer" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_RejectsForwardReferenceAndUnproducedGraphOutput()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1));
            model.graph.valueInfos.Add(Value("later", TensorDataType.Float32, 1));
            model.graph.nodes.Add(Node("consumer", "Identity", new[] { "later" }, new[] { "used" }));
            model.graph.nodes.Add(Node("producer", "Identity", new[] { "x" }, new[] { "later" }));
            model.graph.outputs.Add(Value("missing", TensorDataType.Float32, 1));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "non-topological-input" && diagnostic.blocking));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "unproduced-graph-output" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_StrictImportContractRequiresDeclaredGraphOutput()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1));
            model.graph.nodes.Add(Node("identity", "Identity", new[] { "x" }, new[] { "y" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions { requireDeclaredGraphOutputs = true });

            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "missing-graph-output" && diagnostic.blocking));
        }

        [TestCase(11, 1)]
        [TestCase(13, 1)]
        public void OnnxLowering_UsesOpsetAwareSoftmaxDefaultAxis(int opset, int expectedRuntimeAxis)
        {
            var model = new OnnxModel { opset = opset };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 2, 3));
            model.graph.nodes.Add(Node("softmax", "Softmax", new[] { "x" }, new[] { "y" }));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.True);
            var softmax = result.graph.layers.Single(layer => layer.typeName == "Softmax");
            Assert.That(softmax.GetInt(0), Is.EqualTo(expectedRuntimeAxis));
            if (opset < 13)
            {
                Assert.That(result.graph.layers.Count(layer => layer.typeName == "Reshape"), Is.EqualTo(2));
                Assert.That(softmax.GetString("onnx.legacy_flatten_shape"), Is.EqualTo("1,6"));
            }
            else
            {
                Assert.That(result.graph.layers.Count(layer => layer.typeName == "Reshape"), Is.Zero);
            }
        }

        [TestCase("Softmax", 0)]
        [TestCase("LogSoftmax", 1)]
        [TestCase("Hardmax", 2)]
        public void OnnxLowering_EncodesAxisActivationMode(string opType, int expectedMode)
        {
            var model = new OnnxModel { opset = 13 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 4));
            model.graph.nodes.Add(Node(opType.ToLowerInvariant(), opType, new[] { "x" }, new[] { "y" }));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.True);
            Assert.That(result.graph.layers.Last().typeName, Is.EqualTo("Softmax"));
            Assert.That(result.graph.layers.Last().GetInt(10), Is.EqualTo(expectedMode));
        }

        [Test]
        public void OnnxLowering_SqueezeWithoutAxesRemovesAllStaticSingletons()
        {
            var model = new OnnxModel { opset = 13 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 2, 1, 3));
            model.graph.nodes.Add(Node("squeeze", "Squeeze", new[] { "x" }, new[] { "y" }));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.True);
            Assert.That(Find(result, "y").shape, Is.EqualTo(new long[] { 2, 3 }));
            Assert.That(result.graph.layers.Last().GetString("axes"), Is.EqualTo("1"));
        }

        [Test]
        public void OnnxLowering_ReduceWithEmptyAxesNoopPreservesShape()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 2, 3));
            var reduce = Node("reduce", "ReduceSum", new[] { "x" }, new[] { "y" });
            reduce.attributes["noop_with_empty_axes"] = Int(1);
            model.graph.nodes.Add(reduce);

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.True);
            Assert.That(Find(result, "y").shape, Is.EqualTo(new long[] { 1, 2, 3 }));
            Assert.That(result.graph.layers.Last().typeName, Is.EqualTo("Noop"));
        }

        [Test]
        public void OnnxLowering_RejectsDynamicClipBoundAndDropoutTrainingMode()
        {
            var clipModel = new OnnxModel { opset = 13 };
            clipModel.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 4));
            clipModel.graph.inputs.Add(Value("minimum", TensorDataType.Float32));
            clipModel.graph.nodes.Add(Node("clip", "Clip", new[] { "x", "minimum" }, new[] { "y" }));

            var dropoutModel = new OnnxModel { opset = 13 };
            dropoutModel.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 4));
            dropoutModel.graph.inputs.Add(Value("ratio", TensorDataType.Float32));
            dropoutModel.graph.inputs.Add(Value("training", TensorDataType.Int32));
            dropoutModel.graph.nodes.Add(Node("dropout", "Dropout", new[] { "x", "ratio", "training" }, new[] { "y" }));

            var clip = AexisOnnxGraphLowering.Lower(clipModel);
            var dropout = AexisOnnxGraphLowering.Lower(dropoutModel);

            Assert.That(clip.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "non-static-clip-bound" && diagnostic.blocking));
            Assert.That(dropout.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "dynamic-dropout-training-mode" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_RejectsLayerNormOutsideFinalAxis()
        {
            var model = new OnnxModel { opset = 17 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 2, 3));
            model.graph.initializers["scale"] = FloatTensor("scale", new[] { 1f, 1f }, 2);
            model.graph.initializers["bias"] = FloatTensor("bias", new[] { 0f, 0f }, 2);
            var layerNorm = Node("layernorm", "LayerNormalization", new[] { "x", "scale", "bias" }, new[] { "y" });
            layerNorm.attributes["axis"] = Int(1);
            model.graph.nodes.Add(layerNorm);

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.False);
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "unsupported-layernorm-axis" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_PReluAcceptsOnlyScalarOrNchwChannelBroadcastSlope()
        {
            var validModel = new OnnxModel();
            validModel.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 3, 2, 2));
            validModel.graph.initializers["slope"] = FloatTensor("slope", new[] { 0.1f, 0.2f, 0.3f }, 3, 1, 1);
            validModel.graph.nodes.Add(Node("prelu", "PRelu", new[] { "x", "slope" }, new[] { "y" }));

            var invalidModel = new OnnxModel();
            invalidModel.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 3, 2, 2));
            invalidModel.graph.initializers["slope"] = FloatTensor("slope", new[] { 0.1f, 0.2f, 0.3f }, 3);
            invalidModel.graph.nodes.Add(Node("prelu", "PRelu", new[] { "x", "slope" }, new[] { "y" }));

            var valid = AexisOnnxGraphLowering.Lower(validModel);
            var invalid = AexisOnnxGraphLowering.Lower(invalidModel);

            Assert.That(valid.IsEligible, Is.True);
            Assert.That(valid.graph.layers.Last().GetInt(0, 0), Is.EqualTo(3));
            Assert.That(invalid.IsEligible, Is.False);
            Assert.That(invalid.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "missing-static-prelu-slope" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_RejectsLayerNormAuxiliaryOutputsAndAffineMismatch()
        {
            var model = new OnnxModel { opset = 17 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 2, 5));
            model.graph.initializers["scale"] = FloatTensor("scale", new[] { 1f, 1f, 1f, 1f }, 4);
            model.graph.initializers["bias"] = FloatTensor("bias", new[] { 0f, 0f, 0f, 0f }, 4);
            model.graph.nodes.Add(Node("layernorm", "LayerNormalization", new[] { "x", "scale", "bias" }, new[] { "y", "mean", "invstd" }));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.False);
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-layernorm-auxiliary-outputs" && diagnostic.blocking));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "normalization-affine-shape-mismatch" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_PreservesLogicalBoolCastSemantics()
        {
            var toBool = new OnnxModel { opset = 19 };
            var floatInput = Value("x", TensorDataType.Float32, 1, 4);
            floatInput.onnxDataType = 1;
            toBool.graph.inputs.Add(floatInput);
            var cast = Node("cast_bool", "Cast", new[] { "x" }, new[] { "mask" });
            cast.attributes["to"] = Int(9);
            toBool.graph.nodes.Add(cast);

            var fromBool = new OnnxModel { opset = 19 };
            var boolInput = Value("mask", TensorDataType.Int32, 1, 4);
            boolInput.onnxDataType = 9;
            fromBool.graph.inputs.Add(boolInput);
            var castFloat = Node("cast_float", "Cast", new[] { "mask" }, new[] { "y" });
            castFloat.attributes["to"] = Int(1);
            fromBool.graph.nodes.Add(castFloat);

            var boolResult = AexisOnnxGraphLowering.Lower(toBool);
            var floatResult = AexisOnnxGraphLowering.Lower(fromBool);

            Assert.That(boolResult.IsEligible, Is.True);
            Assert.That(boolResult.graph.layers.Last().GetInt(1), Is.EqualTo(7));
            Assert.That(Find(boolResult, "mask").dataType, Is.EqualTo(TensorDataType.Int32));
            Assert.That(Find(boolResult, "mask").onnxDataType, Is.EqualTo(9));
            Assert.That(floatResult.IsEligible, Is.True);
            Assert.That(floatResult.graph.layers.Last().GetInt(0), Is.EqualTo(7));
        }

        [Test]
        public void OnnxLowering_InfersOneHotAndConstantOfShapeOutputDtypes()
        {
            var oneHotModel = new OnnxModel { opset = 13 };
            oneHotModel.graph.inputs.Add(Value("indices", TensorDataType.Int32, 1, 2));
            oneHotModel.graph.initializers["depth"] = IntTensor("depth", new[] { 3 }, 1);
            oneHotModel.graph.initializers["values"] = IntTensor("values", new[] { 0, 1 }, 2);
            oneHotModel.graph.nodes.Add(Node("onehot", "OneHot", new[] { "indices", "depth", "values" }, new[] { "encoded" }));

            var constantModel = new OnnxModel { opset = 13 };
            constantModel.graph.initializers["shape"] = IntTensor("shape", new[] { 2, 3 }, 2);
            var constant = Node("fill", "ConstantOfShape", new[] { "shape" }, new[] { "filled" });
            constant.attributes["value"] = TensorAttribute(IntTensor("fill_value", new[] { 7 }));
            constantModel.graph.nodes.Add(constant);

            var oneHot = AexisOnnxGraphLowering.Lower(oneHotModel);
            var constantOfShape = AexisOnnxGraphLowering.Lower(constantModel);

            Assert.That(oneHot.IsEligible, Is.True);
            Assert.That(Find(oneHot, "encoded").dataType, Is.EqualTo(TensorDataType.Int32));
            Assert.That(Find(oneHot, "encoded").shape, Is.EqualTo(new long[] { 1, 2, 3 }));
            Assert.That(constantOfShape.IsEligible, Is.True);
            Assert.That(Find(constantOfShape, "filled").dataType, Is.EqualTo(TensorDataType.Int32));
            Assert.That(Find(constantOfShape, "filled").shape, Is.EqualTo(new long[] { 2, 3 }));
        }

        [Test]
        public void OnnxCompiler_ConvertsBoolRawInitializerToImmutableLogicalInt32Weights()
        {
            var model = new OnnxModel { opset = 13 };
            model.graph.initializers["mask"] = new OnnxTensor
            {
                name = "mask",
                dataType = TensorDataType.Int32,
                onnxDataType = 9,
                dims = new long[] { 2 },
                rawData = new byte[] { 0, 1 }
            };
            model.graph.nodes.Add(Node("identity", "Identity", new[] { "mask" }, new[] { "result" }));

            var lowering = AexisOnnxGraphLowering.Lower(model);
            var compiled = AexisOnnxGraphCompiler.Compile(lowering);

            Assert.That(lowering.IsEligible, Is.True);
            Assert.That(compiled.immutableWeights, Is.EqualTo(new[] { 0f, 1f }));
        }

        [Test]
        public void OnnxLowering_RejectsDynamicPadValueAndNonCanonicalResizeAxes()
        {
            var padModel = new OnnxModel { opset = 13 };
            padModel.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 1, 2, 2));
            padModel.graph.inputs.Add(Value("pad_value", TensorDataType.Float32));
            padModel.graph.initializers["pads"] = IntTensor("pads", new[] { 0, 0, 1, 1, 0, 0, 1, 1 }, 8);
            padModel.graph.nodes.Add(Node("pad", "Pad", new[] { "x", "pads", "pad_value" }, new[] { "y" }));

            var resizeModel = new OnnxModel { opset = 18 };
            resizeModel.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 1, 2, 2));
            resizeModel.graph.initializers["sizes"] = IntTensor("sizes", new[] { 4, 4 }, 2);
            var resize = Node("resize", "Resize", new[] { "x", string.Empty, string.Empty, "sizes" }, new[] { "y" });
            resize.attributes["axes"] = Ints(1, 3);
            resizeModel.graph.nodes.Add(resize);

            var pad = AexisOnnxGraphLowering.Lower(padModel);
            var resized = AexisOnnxGraphLowering.Lower(resizeModel);

            Assert.That(pad.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "dynamic-pad-value" && diagnostic.blocking));
            Assert.That(resized.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "unsupported-resize-axes" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_RejectsUnimplementedNearestResizeProfileAndAmbiguousSizeSource()
        {
            var alignCornersModel = new OnnxModel { opset = 18 };
            alignCornersModel.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 1, 2, 2));
            alignCornersModel.graph.initializers["sizes"] = IntTensor("sizes", new[] { 1, 1, 3, 3 }, 4);
            var alignCorners = Node("resize", "Resize", new[] { "x", string.Empty, string.Empty, "sizes" }, new[] { "y" });
            alignCorners.attributes["coordinate_transformation_mode"] = StringAttribute("align_corners");
            alignCornersModel.graph.nodes.Add(alignCorners);

            var ambiguousModel = new OnnxModel { opset = 18 };
            ambiguousModel.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 1, 2, 2));
            ambiguousModel.graph.initializers["scales"] = FloatTensor("scales", new[] { 1f, 1f, 1.5f, 1.5f }, 4);
            ambiguousModel.graph.initializers["sizes"] = IntTensor("sizes", new[] { 1, 1, 3, 3 }, 4);
            ambiguousModel.graph.nodes.Add(Node("resize", "Resize", new[] { "x", string.Empty, "scales", "sizes" }, new[] { "y" }));

            var alignCornersResult = AexisOnnxGraphLowering.Lower(alignCornersModel);
            var ambiguousResult = AexisOnnxGraphLowering.Lower(ambiguousModel);

            Assert.That(alignCornersResult.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-resize-coordinates" && diagnostic.blocking));
            Assert.That(ambiguousResult.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "ambiguous-resize-size-source" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_AveragePoolPreservesFloorAndIncludePadContracts()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 1, 4, 4));
            var pool = Node("pool", "AveragePool", new[] { "x" }, new[] { "y" });
            pool.attributes["kernel_shape"] = Ints(3, 3);
            pool.attributes["strides"] = Ints(2, 2);
            pool.attributes["pads"] = Ints(0, 0, 0, 0);
            pool.attributes["count_include_pad"] = Int(1);
            model.graph.nodes.Add(pool);

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.True);
            var layer = result.graph.layers.Single(candidate => candidate.name == "pool");
            Assert.That(layer.GetInt(5), Is.EqualTo(1));
            Assert.That(layer.GetInt(6), Is.EqualTo(1));
            Assert.That(Find(result, "y").shape, Is.EqualTo(new long[] { 1, 1, 1, 1 }));

            pool.attributes["count_include_pad"] = Int(2);
            var rejected = AexisOnnxGraphLowering.Lower(model);
            Assert.That(rejected.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "invalid-count-include-pad" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_BoundedNonZeroSynthesizesGpuCountOutput()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 4));
            model.graph.nodes.Add(Node("nonzero", "NonZero", new[] { "x" }, new[] { "indices" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                enableBoundedDataIndexLowering = true,
                outputCapacities = { ["nonzero"] = 4 }
            });

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            var layer = result.graph.layers[1];
            Assert.That(layer.topNames, Is.EqualTo(new[] { "indices", "indices.count" }));
            Assert.That(layer.GetInt(30), Is.EqualTo(4));
            Assert.That(Find(result, "indices").shape, Is.EqualTo(new long[] { 1, 4 }));
            Assert.That(Find(result, "indices.count").shape, Is.EqualTo(new long[] { 1 }));
        }

        [Test]
        public void OnnxLowering_SeedsRandomLikeForTheCanonicalPack4CommandBufferProfile()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 5, 2, 3));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 1, 5, 2, 3));
            var random = Node("seeded_random", "RandomUniformLike", new[] { "x" }, new[] { "y" });
            random.attributes["seed"] = FloatAttribute(37f);
            random.attributes["low"] = FloatAttribute(-2f);
            random.attributes["high"] = FloatAttribute(3f);
            model.graph.nodes.Add(random);

            var result = AexisOnnxGraphLowering.Lower(model);
            var layer = result.graph.layers.Single(candidate => candidate.name == "seeded_random");
            var report = AnalyzeLoadedLayer(layer, StrictTextureRequest(new[] { Texture("x", 3, 3, 2, 1, 5) }));

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(layer.typeName, Is.EqualTo("RandomLike"));
            Assert.That(layer.GetString("aexis.random.operator"), Is.EqualTo("RandomUniformLike"));
            Assert.That(layer.GetInt(0), Is.EqualTo(37));
            Assert.That(layer.GetFloat(1), Is.EqualTo(-2f));
            Assert.That(layer.GetFloat(2), Is.EqualTo(3f));
            Assert.That(Find(result, "y").runtimeShape, Is.EqualTo(new long[] { 5, 2, 3 }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.nodes.Single(node => node.layer == "seeded_random").executionPath,
                Is.EqualTo("command-buffer-pack4:deterministic-rng"));
        }

        [Test]
        public void OnnxLowering_RejectsRandomLikeWithoutAnExactExplicitSeed()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 1, 4, 2, 2));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 1, 4, 2, 2));
            var random = Node("unseeded_random", "RandomNormalLike", new[] { "x" }, new[] { "y" });
            random.attributes["seed"] = FloatAttribute(1.5f);
            model.graph.nodes.Add(random);

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.False, DescribeLowering(result));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "missing-or-invalid-deterministic-random-seed" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_MapsStaticRandomUniformToTheZeroInputPack4ArenaProfile()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 1, 3, 2, 2));
            var random = Node("static_random", "RandomUniform", Array.Empty<string>(), new[] { "y" });
            random.attributes["seed"] = FloatAttribute(73f);
            random.attributes["low"] = FloatAttribute(-2f);
            random.attributes["high"] = FloatAttribute(3f);
            random.attributes["shape"] = Ints(1, 3, 2, 2);
            model.graph.nodes.Add(random);

            var result = AexisOnnxGraphLowering.Lower(model);
            var layer = result.graph.layers.Single(candidate => candidate.name == "static_random");
            var report = AnalyzeLoadedLayer(layer, StrictTextureRequest(Array.Empty<AexisTexturePlanTensorDescriptor>()));

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(layer.typeName, Is.EqualTo("RandomLike"));
            Assert.That(layer.bottomNames, Is.Empty);
            Assert.That(layer.GetString("aexis.random.operator"), Is.EqualTo("RandomUniform"));
            Assert.That(layer.GetInt(0), Is.EqualTo(73));
            Assert.That(layer.GetInt(10), Is.EqualTo(3));
            Assert.That(layer.GetInt(11), Is.EqualTo(2));
            Assert.That(layer.GetInt(12), Is.EqualTo(2));
            Assert.That(layer.GetInt(13), Is.EqualTo(1));
            Assert.That(layer.GetInt(14), Is.EqualTo(3));
            Assert.That(Find(result, "y").runtimeShape, Is.EqualTo(new long[] { 3, 2, 2 }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.nodes.Single(node => node.layer == "static_random").executionPath,
                Is.EqualTo("command-buffer-pack4:deterministic-static-rng"));
        }

        [Test]
        public void OnnxLowering_MapsSeededMultinomialToTheBoundedPack4Int32Profile()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("logits", TensorDataType.Float32, 2, 5));
            model.graph.outputs.Add(Value("samples", TensorDataType.Int32, 2, 5));
            var multinomial = Node("sample", "Multinomial", new[] { "logits" }, new[] { "samples" });
            multinomial.attributes["sample_size"] = Int(5);
            multinomial.attributes["seed"] = FloatAttribute(1234567f);
            multinomial.attributes["dtype"] = Int(6);
            model.graph.nodes.Add(multinomial);

            var result = AexisOnnxGraphLowering.Lower(model);
            var layer = result.graph.layers.Single(candidate => candidate.name == "sample");
            var report = AnalyzeLoadedLayer(layer, StrictTextureRequest(new[] { Texture("logits", 3, 1, 2, 1, 5) }));

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(layer.typeName, Is.EqualTo("Multinomial"));
            Assert.That(layer.GetInt(0), Is.EqualTo(5));
            Assert.That(layer.GetInt(1), Is.EqualTo(1234567));
            Assert.That(Find(result, "samples").dataType, Is.EqualTo(TensorDataType.Int32));
            Assert.That(Find(result, "samples").shape, Is.EqualTo(new long[] { 2, 5 }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            var node = report.texturePlan.nodes.Single(candidate => candidate.layer == "sample");
            Assert.That(node.executionPath, Is.EqualTo("command-buffer-pack4:bounded-multinomial"));
            Assert.That(node.outputs[0].logicalDtype, Is.EqualTo("Int32"));
        }

        [Test]
        public void OnnxLowering_CompilesForwardRnnIntoTheBoundedPack4ImmutableWeightProfile()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 3, 1, 1));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 3, 1, 1));
            model.graph.initializers["w"] = FloatTensor("w", new[] { 0.5f }, 1, 1, 1);
            model.graph.initializers["r"] = FloatTensor("r", new[] { 0.25f }, 1, 1, 1);
            model.graph.initializers["b"] = FloatTensor("b", new[] { 0.1f, 0.2f }, 1, 2);
            var recurrent = Node("forward_rnn", "RNN", new[] { "x", "w", "r", "b" }, new[] { "y" });
            recurrent.attributes["hidden_size"] = Int(1);
            model.graph.nodes.Add(recurrent);

            var lowering = AexisOnnxGraphLowering.Lower(model);
            Assert.That(lowering.IsEligible, Is.True, DescribeLowering(lowering));
            var layer = lowering.graph.layers.Single(candidate => candidate.name == "forward_rnn");
            var compiled = AexisOnnxGraphCompiler.Compile(lowering);
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
            compiled.LoadInto(session);
            var report = session.AnalyzeLoadedModelPreflight(StrictTextureRequest(new[] { Texture("x", 3, 3, 1, 1, 1) }));

            Assert.That(layer.typeName, Is.EqualTo("RNN"));
            Assert.That(layer.bottomNames, Is.EqualTo(new[] { "x" }));
            Assert.That(layer.GetInt(0), Is.EqualTo(1));
            Assert.That(layer.GetInt(1), Is.EqualTo(1));
            Assert.That(layer.GetString("onnx.w"), Is.EqualTo("w"));
            Assert.That(layer.GetString("onnx.r"), Is.EqualTo("r"));
            Assert.That(compiled.immutableWeights, Is.EqualTo(new[] { 0.5f, 0.25f, 0.3f }).Within(1e-7f));
            Assert.That(Find(lowering, "y").shape, Is.EqualTo(new long[] { 3, 1, 1 }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.nodes.Single(node => node.layer == "forward_rnn").executionPath,
                Is.EqualTo("command-buffer-pack4:bounded-rnn-fp32"));
        }

        [TestCase("GRU", 3)]
        [TestCase("LSTM", 4)]
        public void OnnxLowering_CompilesForwardGatedRecurrentProfilesIntoImmutablePack4Weights(string operatorName, int gates)
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 2, 1, 1));
            model.graph.outputs.Add(Value("y", TensorDataType.Float32, 2, 1, 1));
            model.graph.initializers["w"] = FloatTensor("w", Enumerable.Repeat(0.25f, gates).ToArray(), 1, gates, 1);
            model.graph.initializers["r"] = FloatTensor("r", Enumerable.Repeat(0.125f, gates).ToArray(), 1, gates, 1);
            var bias = new float[2 * gates];
            for (var index = 0; index < bias.Length; index++) bias[index] = index + 1;
            model.graph.initializers["b"] = FloatTensor("b", bias, 1, 2 * gates);
            var recurrent = Node("forward_" + operatorName.ToLowerInvariant(), operatorName, new[] { "x", "w", "r", "b" }, new[] { "y" });
            recurrent.attributes["hidden_size"] = Int(1);
            model.graph.nodes.Add(recurrent);

            var lowering = AexisOnnxGraphLowering.Lower(model);
            Assert.That(lowering.IsEligible, Is.True, DescribeLowering(lowering));
            var compiled = AexisOnnxGraphCompiler.Compile(lowering);
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
            compiled.LoadInto(session);
            var report = session.AnalyzeLoadedModelPreflight(StrictTextureRequest(new[] { Texture("x", 3, 2, 1, 1, 1) }));

            Assert.That(compiled.immutableWeights.Length, Is.EqualTo(3 * gates));
            Assert.That(compiled.immutableWeights.Skip(2 * gates).ToArray(),
                Is.EqualTo(Enumerable.Range(0, gates).Select(index => (float)(2 * index + gates + 2)).ToArray()).Within(1e-7f));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.nodes.Single(node => node.layer == "forward_" + operatorName.ToLowerInvariant()).executionPath,
                Is.EqualTo("command-buffer-pack4:bounded-" + operatorName.ToLowerInvariant() + "-fp32"));
        }

        [Test]
        public void OnnxLowering_BoundedNmsSynthesizesPaddedIndicesAndGpuCountOutput()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("boxes", TensorDataType.Float32, 1, 3, 4));
            model.graph.inputs.Add(Value("scores", TensorDataType.Float32, 1, 2, 3));
            model.graph.initializers.Add("max_output", Int64Tensor("max_output", new long[] { 2 }, 1));
            model.graph.initializers.Add("iou", FloatTensor("iou", new[] { 0.5f }, 1));
            model.graph.initializers.Add("score", FloatTensor("score", new[] { 0.2f }, 1));
            model.graph.nodes.Add(Node("nms", "NonMaxSuppression",
                new[] { "boxes", "scores", "max_output", "iou", "score" }, new[] { "selected_indices" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                enableBoundedNonMaxSuppressionLowering = true,
                outputCapacities = { ["nms"] = 4 }
            });

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            var layer = result.graph.layers.Single(candidate => candidate.name == "nms");
            Assert.That(layer.typeName, Is.EqualTo("Nms"));
            Assert.That(layer.bottomNames, Is.EqualTo(new[] { "boxes", "scores" }));
            Assert.That(layer.topNames, Is.EqualTo(new[] { "selected_indices", "selected_indices.aexis_count" }));
            Assert.That(layer.GetInt(0), Is.EqualTo(4));
            Assert.That(layer.GetInt(1), Is.EqualTo(2));
            Assert.That(Find(result, "selected_indices").shape, Is.EqualTo(new long[] { 4, 3 }));
            Assert.That(Find(result, "selected_indices.aexis_count").shape, Is.EqualTo(new long[] { 1 }));
        }

        [Test]
        public void OnnxLowering_BoundedNmsRejectsConsumerWithoutGpuCountContract()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("boxes", TensorDataType.Float32, 1, 3, 4));
            model.graph.inputs.Add(Value("scores", TensorDataType.Float32, 1, 2, 3));
            model.graph.initializers.Add("max_output", Int64Tensor("max_output", new long[] { 2 }, 1));
            model.graph.nodes.Add(Node("nms", "NonMaxSuppression",
                new[] { "boxes", "scores", "max_output" }, new[] { "selected_indices" }));
            model.graph.nodes.Add(Node("consumer", "Identity", new[] { "selected_indices" }, new[] { "output" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                enableBoundedNonMaxSuppressionLowering = true,
                outputCapacities = { ["nms"] = 4 }
            });

            Assert.That(result.IsEligible, Is.False, DescribeLowering(result));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "bounded-nms-consumer-requires-count" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_BoundedNonZeroRejectsCapacityBelowStaticMaximum()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 4));
            model.graph.nodes.Add(Node("nonzero", "NonZero", new[] { "x" }, new[] { "indices" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                enableBoundedDataIndexLowering = true,
                outputCapacities = { ["nonzero"] = 3 }
            });

            Assert.That(result.IsEligible, Is.False, DescribeLowering(result));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "insufficient-output-capacity" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_RejectsStandardNonZeroWithoutBoundedContract()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 4));
            model.graph.nodes.Add(Node("nonzero", "NonZero", new[] { "x" }, new[] { "indices" }));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.False);
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic => diagnostic.code == "bounded-data-index-profile-required" && diagnostic.blocking));
        }

        [TestCase("NonZero")]
        [TestCase("Compress")]
        public void OnnxLowering_RejectsBoundedCompactionConsumedWithoutGpuCount(string opType)
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("x", TensorDataType.Float32, 4));
            if (opType == "Compress")
                model.graph.inputs.Add(Value("condition", TensorDataType.Int32, 4));
            model.graph.nodes.Add(Node("compact", opType,
                opType == "Compress" ? new[] { "x", "condition" } : new[] { "x" },
                new[] { "compacted" }));
            model.graph.nodes.Add(Node("consumer", "Identity", new[] { "compacted" }, new[] { "output" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                enableBoundedDataIndexLowering = true,
                outputCapacities = { ["compact"] = 4 }
            });

            Assert.That(result.IsEligible, Is.False, DescribeLowering(result));
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "bounded-data-index-consumer-requires-count" && diagnostic.blocking));
        }

        [Test]
        public void LongNcnnAliases_AreCanonicalizedBeforeFixedKeyLookup()
        {
            var depthwise1D = new AexisGraphModel.Layer
            {
                typeName = "ConvolutionDepthWise1D",
                type = AexisLayerTypeKey.FromString("ConvolutionDepthWise1D")
            };
            var cumulative = new AexisGraphModel.Layer { typeName = "CumulativeSum", type = AexisLayerTypeKey.FromString("CumulativeSum") };

            Assert.That(AexisLayerFactory.Create(depthwise1D).TypeKey, Is.EqualTo(AexisLayerTypes.Convolution1D));
            Assert.That(AexisLayerFactory.Create(cumulative).TypeKey, Is.EqualTo(AexisLayerTypes.CumSum));
            Assert.That(AexisLayerFactory.IsRegistered("Deconvolution1D"), Is.True);
            Assert.That(AexisLayerFactory.Create(new AexisGraphModel.Layer { typeName = "BNLL" }).TypeKey, Is.EqualTo(AexisLayerTypes.BNLL));
            Assert.That(AexisOperatorCapabilities.TryGet("CopyTo", out var copyTo), Is.True);
            Assert.That(copyTo.ranks, Is.EqualTo(new[] { 3, 4 }));
            Assert.That(AexisOperatorCapabilities.TryGet("MVN", out var mvn), Is.True);
            Assert.That(mvn.ranks, Is.EqualTo(new[] { 3, 4 }));
        }

        [Test]
        public void NcnnReferenceP2Profiles_AreExplicitCapabilities()
        {
            var recurrent = new[]
            {
                "GRU", "LSTM", "RNN"
            };
            var document = AexisOperatorCapabilities.CreateDocument();
            foreach (var operatorName in recurrent)
            {
                Assert.That(AexisOperatorCapabilities.TryGet(operatorName, out var capability), Is.True, operatorName);
                Assert.That(capability.status, Is.EqualTo(AexisOperatorCapabilityStatus.SupportedByProfile), operatorName);
                Assert.That(capability.importSupported, Is.True, operatorName);
                Assert.That(capability.commandBuffer, Is.True, operatorName);
                Assert.That(capability.ranks, Is.EqualTo(new[] { 3 }), operatorName);
                Assert.That(document.operators.Any(entry => entry.operatorName == operatorName), Is.True, operatorName);
            }

            Assert.That(AexisOperatorCapabilities.TryGet("StatisticsPooling", out var statisticsPooling), Is.True);
            Assert.That(statisticsPooling.status, Is.EqualTo(AexisOperatorCapabilityStatus.SupportedByProfile));
            Assert.That(statisticsPooling.importSupported, Is.True);
            Assert.That(statisticsPooling.commandBuffer, Is.True);
            Assert.That(document.operators.Any(entry => entry.operatorName == "StatisticsPooling"), Is.True);

            Assert.That(AexisOperatorCapabilities.TryGet("ConvolutionDepthWise3D", out var depthWise3D), Is.True);
            Assert.That(depthWise3D.status, Is.EqualTo(AexisOperatorCapabilityStatus.SupportedByProfile));
            Assert.That(depthWise3D.importSupported, Is.True);
            Assert.That(depthWise3D.commandBuffer, Is.True);
            Assert.That(AexisOperatorCapabilities.TryGet("DeconvolutionDepthWise3D", out var deconvDepthWise3D), Is.True);
            Assert.That(deconvDepthWise3D.status, Is.EqualTo(AexisOperatorCapabilityStatus.SupportedByProfile));
            Assert.That(deconvDepthWise3D.importSupported, Is.True);
            Assert.That(deconvDepthWise3D.commandBuffer, Is.True);
            Assert.That(AexisOperatorCapabilities.TryGet("DeconvolutionDepthWise1D", out var deconvDepthWise1D), Is.True);
            Assert.That(deconvDepthWise1D.status, Is.EqualTo(AexisOperatorCapabilityStatus.SupportedByProfile));
            Assert.That(deconvDepthWise1D.importSupported, Is.True);
            Assert.That(deconvDepthWise1D.commandBuffer, Is.True);

            foreach (var operatorName in new[]
            {
                "DeformableConv2D", "DetectionOutput", "Diag", "Einsum", "Flip", "Fold", "GLU", "GridSample",
                "Proposal", "PSROIPooling", "ROIAlign", "ROIPooling", "SPP", "YoloDetectionOutput", "Yolov3DetectionOutput"
            })
            {
                Assert.That(AexisOperatorCapabilities.TryGet(operatorName, out var capability), Is.True, operatorName);
                Assert.That(capability.status, Is.EqualTo(AexisOperatorCapabilityStatus.SupportedByProfile), operatorName);
                Assert.That(capability.commandBuffer, Is.True, operatorName);
            }

            var model = NcnnParamParser.Parse(
                "7767517\n"
                + "2 2\n"
                + "Input input 0 1 data\n"
                + "GridSample unsupported_grid 1 1 data output\n");
            var report = AexisModelPreflight.Analyze(model, new AexisModelPreflightRequest
            {
                strict = true,
                inputs = new[]
                {
                    new AexisPreflightTensorDescriptor
                    {
                        blob = "data",
                        logicalShape = new[] { 3, 8, 8, 1, 4 },
                        storageShape = new[] { 3, 8, 8, 1, 4 },
                        layout = AexisTexturePlanLayout.Packed4,
                        dtype = "FP32",
                        logicalDtype = "Float32"
                    }
                }
            });

            var node = report.nodes.Single(entry => entry.operatorName == "GridSample");
            Assert.That(report.missingNodes, Is.Empty);
            Assert.That(node.strictEligible, Is.False);
            Assert.That(node.status, Is.EqualTo(AexisOperatorCapabilityStatus.SupportedByProfile));
            Assert.That(node.issues, Has.Some.Contains("sample_type"));

            var plannerModel = new AexisGraphModel();
            plannerModel.layers.Add(new AexisGraphModel.Layer
            {
                name = "unsupported_grid",
                typeName = "GridSample",
                bottoms = 2,
                tops = 1,
                bottomNames = new[] { "data", "grid" },
                topNames = new[] { "output" },
                // The node now reaches the loaded P1 verifier. Its invalid sample
                // mode must be rejected there, rather than failing I/O discovery.
                intParams = { [0] = "4", [1] = "1", [2] = "0" }
            });
            var relaxedPlan = AexisTextureExecutionPlanner.Analyze(plannerModel, new AexisTextureExecutionPlanRequest
            {
                strict = true,
                debugOracleRelaxed = false,
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                targetLayout = AexisTexturePlanLayout.Packed4,
                inputs = new[]
                {
                    Texture("data", 3, 8, 8, 1, 4),
                    Texture("grid", 3, 4, 4, 1, 4)
                }
            });

            Assert.That(relaxedPlan.dispatchAllowed, Is.False);
            Assert.That(relaxedPlan.nodes.Single().acceptedByDebugOracle, Is.False);
            Assert.That(relaxedPlan.diagnostics, Has.Some.Matches<AexisTextureExecutionPlanDiagnostic>(diagnostic =>
                diagnostic.code == "command-buffer-pack4-profile-rejected" && diagnostic.blocking));
        }

        [Test]
        public void NcnnBias_RequiresImmutablePack4ConstantsAndHasNoBufferExecutionPath()
        {
            var layer = AexisLayerFactory.Create(new AexisGraphModel.Layer { typeName = "Bias", type = AexisLayerTypes.Bias });

            Assert.That(layer, Is.TypeOf<AexisBiasLayer>());
            Assert.That(layer.SupportsBufferPath, Is.False);
            Assert.That(layer.SupportsCommandBufferPath, Is.True);
            Assert.That(AexisOperatorCapabilities.TryGet("Bias", out var capability), Is.True);
            Assert.That(capability.status, Is.EqualTo(AexisOperatorCapabilityStatus.SupportedByProfile));
            Assert.That(capability.requiredParameters, Does.Contain("0:bias_data_size"));
            Assert.That(capability.profiles[0].rejectedParameters, Does.Contain("transient ComputeBuffer input or output"));
        }

        [Test]
        public void NcnnBias_LoadedImmutableWeightsPassStrictModelPreflight()
        {
            var layer = new AexisGraphModel.Layer
            {
                name = "bias",
                typeName = "Bias",
                type = AexisLayerTypes.Bias,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams = { [0] = "4" }
            };

            var report = AnalyzeLoadedLayer(
                layer,
                StrictTextureRequest(new[] { Texture("data", 3, 4, 2, 1, 4) }),
                1f, 2f, 3f, 4f);

            Assert.That(report.strictEligible, Is.True, report.summary);
            Assert.That(report.texturePlan.nodes[0].executionPath, Does.Contain("bias"));
        }

        [Test]
        public void InstanceNorm_LoadedAffineWeightsUseStrictTextureNativeProfile()
        {
            var layer = new AexisGraphModel.Layer
            {
                name = "instance_norm",
                typeName = "InstanceNorm",
                type = AexisLayerTypes.InstanceNorm,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams = { [0] = "4", [1] = "0.00001", [2] = "1" }
            };

            var implementation = AexisLayerFactory.Create(layer);
            Assert.That(implementation.SupportsBufferPath, Is.False);
            var report = AnalyzeLoadedLayer(
                layer,
                StrictTextureRequest(new[] { Texture("data", 3, 4, 2, 1, 4) }),
                1f, 1f, 1f, 1f,
                0f, 0f, 0f, 0f);

            Assert.That(report.strictEligible, Is.True, report.summary);
            Assert.That(report.texturePlan.nodes[0].executionPath, Does.Contain("instance-norm"));
        }

        [Test]
        public void Bnll_UsesTextureNativePointwiseProfile()
        {
            var layer = new AexisGraphModel.Layer
            {
                name = "bnll",
                typeName = "BNLL",
                type = AexisLayerTypeKey.FromString("BNLL"),
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "out" }
            };

            var report = AnalyzeLoadedLayer(layer, StrictTextureRequest(new[] { Texture("data", 3, 4, 2, 1, 8) }));

            Assert.That(report.strictEligible, Is.True);
            Assert.That(report.texturePlan.nodes[0].executionPath, Does.Contain("pointwise"));
        }

        [Test]
        public void NcnnCumulativeSum_DefaultsAxisZeroButOnnxCumSumRequiresLoweredAxis()
        {
            var ncnn = SentisLayer("CumulativeSum", new[] { "data" }, new[] { "output" });
            var onnx = SentisLayer("CumSum", new[] { "data" }, new[] { "output" });
            var request = StrictTextureRequest(new[] { LinearTexture("data", 1, 4, 1, 1, 1) });

            var ncnnReport = AnalyzeLoadedLayer(ncnn, request);
            var onnxReport = AnalyzeLoadedLayer(onnx, request);

            Assert.That(ncnnReport.strictEligible, Is.True, DescribeReport(ncnnReport));
            Assert.That(ncnnReport.texturePlan.nodes[0].executionPath, Does.Contain("cumsum"));
            Assert.That(onnxReport.strictEligible, Is.False, DescribeReport(onnxReport));
            Assert.That(onnxReport.texturePlan.diagnostics, Has.Some.Matches<AexisTextureExecutionPlanDiagnostic>(diagnostic =>
                diagnostic.reason.Contains("statically lowered axis")));
        }

        [TestCase("Power")]
        [TestCase("Threshold")]
        public void NcnnPointwiseAliases_UseTextureNativeProfiles(string op)
        {
            var layer = new AexisGraphModel.Layer
            {
                name = op.ToLowerInvariant(),
                typeName = op,
                type = AexisLayerTypeKey.FromString(op),
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "out" },
                intParams = { [0] = "2", [1] = "0.5", [2] = "1" }
            };

            var report = AnalyzeLoadedLayer(layer, StrictTextureRequest(new[] { Texture("data", 3, 4, 2, 1, 8) }));

            Assert.That(report.strictEligible, Is.True);
            Assert.That(report.texturePlan.nodes[0].executionPath, Does.Contain("pointwise"));
        }

        [Test]
        public void PReluAndLrn_LoadedConstantsPassOnlyExactStrictProfiles()
        {
            var prelu = new AexisGraphModel.Layer
            {
                name = "prelu",
                typeName = "PReLU",
                type = AexisLayerTypes.PReLU,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams = { [0] = "4" }
            };
            var lrn = new AexisGraphModel.Layer
            {
                name = "lrn",
                typeName = "LRN",
                type = AexisLayerTypes.LRN,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams = { [0] = "0", [1] = "3", [2] = "0.3", [3] = "1", [4] = "1" }
            };
            var input = Texture("data", 3, 2, 2, 1, 4);

            var preluReport = AnalyzeLoadedLayer(prelu, StrictTextureRequest(new[] { input }), 0.1f, 0.2f, 0.3f, 0.4f);
            var lrnReport = AnalyzeLoadedLayer(lrn, StrictTextureRequest(new[] { input }));

            Assert.That(preluReport.strictEligible, Is.True);
            Assert.That(preluReport.texturePlan.nodes[0].executionPath, Does.Contain("prelu-channel"));
            Assert.That(lrnReport.strictEligible, Is.True);
            Assert.That(lrnReport.texturePlan.nodes[0].executionPath, Does.Contain("lrn-across-channels"));

            prelu.intParams[0] = "3";
            var mismatched = AnalyzeLoadedLayer(prelu, StrictTextureRequest(new[] { input }), 0.1f, 0.2f, 0.3f);
            Assert.That(mismatched.strictEligible, Is.False);
        }

        [TestCase("NonZero")]
        [TestCase("Compress")]
        [TestCase("GatherND")]
        [TestCase("Scatter")]
        [TestCase("ScatterElements")]
        [TestCase("ScatterND")]
        public void DataDependentIndexOps_RequireExactTextureNativeProfile(string op)
        {
            var model = new AexisGraphModel();
            model.layers.Add(new AexisGraphModel.Layer
            {
                name = op + "_node",
                typeName = op,
                type = AexisLayerTypeKey.FromString(op),
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "out" }
            });
            var report = AexisModelPreflight.Analyze(model, new AexisModelPreflightRequest
            {
                strict = true,
                inputs = new[] { new AexisPreflightTensorDescriptor { blob = "data", logicalShape = new[] { 1, 4, 1, 1, 1 }, storageShape = new[] { 2, 4, 1, 1, 1 }, layout = "Packed4", dtype = "FP32", logicalDtype = "Float32" } },
                textureInputs = new[]
                {
                    new AexisTexturePlanTensorDescriptor
                    {
                        blob = "data", logicalShape = new[] { 1, 4, 1, 1, 1 }, storageShape = new[] { 2, 4, 1, 1, 1 },
                        layout = AexisTexturePlanLayout.Packed4, dtype = "FP32", logicalDtype = "Float32", textureBacked = true
                    }
                }
            });

            Assert.That(report.strictEligible, Is.False);
            Assert.That(report.nodes[0].status, Is.EqualTo(AexisOperatorCapabilityStatus.SupportedByProfile));
            Assert.That(report.nodes[0].recommendedAction, Is.Not.Empty);
            Assert.That(report.texturePlan, Is.Not.Null);
            Assert.That(report.texturePlan.dispatchAllowed, Is.False);
        }

        [Test]
        public void BoundedNonZero_ExactTextureProfilePassesStrictPreflight()
        {
            var model = new AexisGraphModel();
            var layer = new AexisGraphModel.Layer
            {
                name = "nonzero",
                typeName = "NonZero",
                type = AexisLayerTypeKey.FromString("NonZero"),
                bottoms = 1,
                tops = 2,
                bottomNames = new[] { "data" },
                topNames = new[] { "indices", "indices.count" },
                intParams = { [30] = "4" },
                stringParams = { ["capacity"] = "4" }
            };
            model.layers.Add(layer);

            var request = StrictTextureRequest(new[]
            {
                LinearTexture("data", 1, 4, 1, 1, 1)
            });

            var staticReport = AexisModelPreflight.Analyze(model, request);
            Assert.That(staticReport.strictEligible, Is.False, "Conditional profiles require the loaded runtime verifier.");

            var report = AnalyzeLoadedLayer(layer, request);

            Assert.That(report.nodes[0].strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.dispatchAllowed, Is.True, DescribeReport(report));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.nodes[0].outputs[0].logicalShape, Is.EqualTo(new[] { 2, 4, 1, 1, 1 }));
            Assert.That(report.nodes[0].outputs[0].logicalDtype, Is.EqualTo("Int32"));
            Assert.That(report.nodes[0].outputs[0].dtype, Is.EqualTo("FP32"));
        }

        [Test]
        public void BoundedNonZero_GraphSessionCommandBufferReturnsValueAndCountTogether()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat compute profile required by this integration golden.");

            var layer = SentisLayer("NonZero", new[] { "data" }, new[] { "values", "count" });
            layer.intParams[30] = "4";
            layer.stringParams["capacity"] = "4";
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(SerializeSingleLayer(layer), reader);
            using var inputBuffer = new ComputeBuffer(4, sizeof(float));
            inputBuffer.SetData(new[] { 0f, -2f, 0f, 5f });
            using var commandBuffer = new CommandBuffer();
            var input = session.RentTempMat(commandBuffer, 4, 1, RenderTextureFormat.RFloat);
            var valueReadback = CreateRFloatTarget(4, 1);
            var countReadback = CreateRFloatTarget(1, 1);
            try
            {
                ops.FillLinearMatFromBuffer(commandBuffer, inputBuffer, 4, 1, input);
                var inputShape = new AexisGraphSession.BufferShape(1, 4, 1, 1, 1);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal) { ["data"] = inputShape },
                    new[] { "values", "count" }))
                {
                    Assert.That(result.OutputNames, Is.EquivalentTo(new[] { "values", "count" }));
                    Assert.That(result.GetLogicalShape("values"), Is.EqualTo(new AexisGraphSession.BufferShape(2, 4, 1, 1, 1)));
                    Assert.That(result.GetLogicalShape("count"), Is.EqualTo(new AexisGraphSession.BufferShape(1, 1, 1, 1, 1)));
                    commandBuffer.CopyTexture(new RenderTargetIdentifier(result.GetTexture("values").nameID), new RenderTargetIdentifier(valueReadback));
                    commandBuffer.CopyTexture(new RenderTargetIdentifier(result.GetTexture("count").nameID), new RenderTargetIdentifier(countReadback));
                }
                session.ReturnTempArray(commandBuffer, input);
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(ReadRFloat(valueReadback, 4), Is.EqualTo(new[] { 1f, 3f, -1f, -1f }).Within(1e-6f));
                Assert.That(ReadRFloat(countReadback, 1), Is.EqualTo(new[] { 2f }).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(valueReadback);
                UnityEngine.Object.DestroyImmediate(countReadback);
            }
        }

        [Test]
        public void BoundedDataIndex_StrictPlannerRequiresTerminalValueAndCountOutputs()
        {
            var model = new AexisGraphModel();
            var nonZero = SentisLayer("NonZero", new[] { "data" }, new[] { "values", "count" });
            nonZero.intParams[30] = "4";
            nonZero.stringParams["capacity"] = "4";
            model.layers.Add(nonZero);
            model.layers.Add(SentisLayer("UnaryOp", new[] { "values" }, new[] { "output" }));

            var plan = AexisTextureExecutionPlanner.Analyze(model, new AexisTextureExecutionPlanRequest
            {
                strict = true,
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                targetLayout = AexisTexturePlanLayout.Packed4,
                inputs = new[] { LinearTexture("data", 1, 4, 1, 1, 1) }
            });

            Assert.That(plan.strictEligible, Is.False);
            Assert.That(plan.diagnostics, Has.Some.Matches<AexisTextureExecutionPlanDiagnostic>(diagnostic =>
                diagnostic.code == "bounded-data-index-output-must-be-terminal" && diagnostic.reason.Contains("values")));
        }

        [Test]
        public void DataIndex_StrictProfilesRejectAxisDtypeDepthAndCapacityMismatches()
        {
            var compress = SentisLayer("Compress", new[] { "data", "condition" }, new[] { "values", "count" });
            compress.intParams[30] = "4";
            compress.intParams[0] = "1";
            compress.stringParams["capacity"] = "4";
            Assert.That(AnalyzeLoadedLayer(compress, StrictTextureRequest(new[]
            {
                LinearTexture("data", 1, 4, 1, 1, 1),
                LinearTexture("condition", 1, 4, 1, 1, 1, "Float32")
            })).strictEligible, Is.False);

            var scatter = ScatterNdLayer();
            scatter.intParams[1] = "2";
            scatter.stringParams["index_depth"] = "2";
            Assert.That(AnalyzeLoadedLayer(scatter, StrictTextureRequest(new[]
            {
                LinearTexture("data", 1, 8, 1, 1, 1),
                LinearTexture("indices", 2, 1, 3, 1, 1, "Int32"),
                LinearTexture("updates", 1, 3, 1, 1, 1)
            })).strictEligible, Is.False);

            scatter = ScatterNdLayer();
            Assert.That(AnalyzeLoadedLayer(scatter, StrictTextureRequest(new[]
            {
                LinearTexture("data", 1, 8, 1, 1, 1),
                LinearTexture("indices", 2, 1, 3, 1, 1, "Int32"),
                LinearTexture("updates", 1, 3, 1, 1, 1, "Int32")
            })).strictEligible, Is.False);

            var nonZero = SentisLayer("NonZero", new[] { "data" }, new[] { "values", "count" });
            var oversize = Math.Max(1, SystemInfo.maxTextureSize) + 1;
            nonZero.intParams[30] = oversize.ToString(CultureInfo.InvariantCulture);
            nonZero.stringParams["capacity"] = nonZero.intParams[30];
            Assert.That(AnalyzeLoadedLayer(nonZero, StrictTextureRequest(new[]
            {
                LinearTexture("data", 1, oversize, 1, 1, 1)
            })).strictEligible, Is.False);
        }

        [Test]
        public void SoftmaxFamily_StrictPreflightAcceptsOnlyDefinedModes()
        {
            foreach (var mode in new[] { 0, 1, 2 })
            {
                var layer = SoftmaxLayer(mode);
                var report = AnalyzeLoadedLayer(layer, StrictTextureRequest(new[] { LinearTexture("data", 2, 4, 1, 1, 1) }));
                Assert.That(report.strictEligible, Is.True, "mode=" + mode.ToString(CultureInfo.InvariantCulture));
                Assert.That(report.texturePlan.nodes[0].executionPath, Does.Contain(mode == 1 ? "log-softmax" : mode == 2 ? "hardmax" : "softmax"));
            }

            var rejected = AnalyzeLoadedLayer(SoftmaxLayer(3), StrictTextureRequest(new[] { LinearTexture("data", 2, 4, 1, 1, 1) }));
            Assert.That(rejected.strictEligible, Is.False);
            Assert.That(rejected.texturePlan.diagnostics, Has.Some.Matches<AexisTextureExecutionPlanDiagnostic>(diagnostic =>
                diagnostic.reason.Contains("mode is outside")));
        }

        [Test]
        public void SoftmaxFamily_LinearMatGpuMatchesStableReference()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat compute profile required by this golden.");

            var values = new[] { -1f, 0.5f, 1f, 2f };
            var maximum = values.Max();
            var exp = values.Select(value => Math.Exp(value - maximum)).ToArray();
            var sum = exp.Sum();
            var expectedSoftmax = exp.Select(value => (float)(value / sum)).ToArray();
            var expectedLogSoftmax = values.Select(value => (float)(value - maximum - Math.Log(sum))).ToArray();
            var expectedHardmax = new[] { 0f, 0f, 0f, 1f };

            using var ops = new AexisOps();
            var input = CreateLinearMat(values);
            try
            {
                foreach (var profile in new[]
                {
                    (mode: 0, expected: expectedSoftmax),
                    (mode: 1, expected: expectedLogSoftmax),
                    (mode: 2, expected: expectedHardmax)
                })
                {
                    var output = CreateRFloatTarget(values.Length, 1);
                    try
                    {
                        ops.SoftmaxLinearMat2D(input, values.Length, 1, output, axis: 0, mode: profile.mode);
                        var actual = ReadRFloat(output, values.Length);
                        Assert.That(actual, Is.EqualTo(profile.expected).Within(2e-5f), "mode=" + profile.mode.ToString(CultureInfo.InvariantCulture));
                    }
                    finally
                    {
                        UnityEngine.Object.DestroyImmediate(output);
                    }
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
            }
        }

        [Test]
        public void Reduction_LinearMatReduceAllConsumesEveryRow()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat compute profile required by this golden.");

            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops);
            var input = CreateLinearMat(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 3, 2);
            try
            {
                var inputShape = new AexisGraphSession.BufferShape(2, 3, 2, 1, 1);
                var context = new AexisLayerBufferContext
                {
                    textureBlobs = new Dictionary<string, AexisGraphSession.TensorRef>(StringComparer.Ordinal)
                    {
                        ["input"] = new AexisGraphSession.TensorRef
                        {
                            texture = input,
                            width = 3,
                            height = 2,
                            packs = 1,
                            refs = 1,
                            owned = false,
                            hasLogicalShape = true,
                            logicalShape = inputShape,
                            hasStorageShape = true,
                            storageShape = inputShape,
                            layoutKind = AexisTextureTensorLayoutKind.LinearMat
                        }
                    },
                    textureShapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal) { ["input"] = inputShape },
                    bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal),
                    bufferRefs = new Dictionary<string, AexisGraphSession.BufferRef>(StringComparer.Ordinal),
                    bufferViews = new Dictionary<string, AexisTensorBuffer>(StringComparer.Ordinal),
                    indexBlobs = new Dictionary<string, AexisGraphSession.IndexRef>(StringComparer.Ordinal),
                    remaining = new Dictionary<string, int>(StringComparer.Ordinal) { ["input"] = 1 },
                    tempOwned = new List<IDisposable>()
                };
                var layer = new AexisGraphModel.Layer
                {
                    name = "reduce_all",
                    typeName = "Reduction",
                    type = AexisLayerTypes.Reduction,
                    bottoms = 1,
                    tops = 1,
                    bottomNames = new[] { "input" },
                    topNames = new[] { "output" },
                    intParams = { [0] = "0", [1] = "1", [4] = "0" }
                };

                new AexisReductionLayer().ExecuteRenderTexturePath(session, layer, context);

                Assert.That(context.textureBlobs.TryGetValue("output", out var output), Is.True);
                Assert.That(ReadRFloat(output.texture, 1)[0], Is.EqualTo(21f).Within(1e-5f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
            }
        }

        [Test]
        public void Cast_LinearMatFloatToLogicalBoolCanonicalizesNonZeroValues()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat compute profile required by this golden.");

            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops);
            var input = CreateLinearMat(new[] { -2f, 0f, 0.25f, 3f });
            try
            {
                var inputShape = new AexisGraphSession.BufferShape(1, 4, 1, 1, 1);
                var storageShape = new AexisGraphSession.BufferShape(2, 4, 1, 1, 1);
                var context = new AexisLayerBufferContext
                {
                    textureBlobs = new Dictionary<string, AexisGraphSession.TensorRef>(StringComparer.Ordinal)
                    {
                        ["input"] = new AexisGraphSession.TensorRef
                        {
                            texture = input,
                            width = 4,
                            height = 1,
                            packs = 1,
                            refs = 1,
                            owned = false,
                            hasLogicalShape = true,
                            logicalShape = inputShape,
                            hasStorageShape = true,
                            storageShape = storageShape,
                            layoutKind = AexisTextureTensorLayoutKind.LinearMat
                        }
                    },
                    textureShapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal) { ["input"] = inputShape },
                    bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal),
                    bufferRefs = new Dictionary<string, AexisGraphSession.BufferRef>(StringComparer.Ordinal),
                    bufferViews = new Dictionary<string, AexisTensorBuffer>(StringComparer.Ordinal),
                    indexBlobs = new Dictionary<string, AexisGraphSession.IndexRef>(StringComparer.Ordinal),
                    remaining = new Dictionary<string, int>(StringComparer.Ordinal) { ["input"] = 1 },
                    tempOwned = new List<IDisposable>()
                };
                var layer = new AexisGraphModel.Layer
                {
                    name = "cast_bool",
                    typeName = "Cast",
                    type = AexisLayerTypes.Cast,
                    bottoms = 1,
                    tops = 1,
                    bottomNames = new[] { "input" },
                    topNames = new[] { "output" },
                    intParams = { [0] = "1", [1] = "7" }
                };

                new AexisCastLayer().ExecuteRenderTexturePath(session, layer, context);

                Assert.That(context.textureBlobs.TryGetValue("output", out var output), Is.True);
                Assert.That(ReadRFloat(output.texture, 4), Is.EqualTo(new[] { 1f, 0f, 1f, 1f }).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
            }
        }

        [Test]
        public void ResizeNearestHalfPixel_UsesRoundPreferFloorAtExactTie()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            var input = CreatePack4Array(new[] { 1f, 0f, 0f, 0f, 2f, 0f, 0f, 0f }, 2, 1, 1);
            var output = CreatePack4Target(3, 1, 1);
            try
            {
                ops.InterpPack4Nearest(input, 1, 1.5f, 1f, output, 0);
                var actual = ReadPack4Slice(output, 0);
                Assert.That(new[] { actual[0], actual[4], actual[8] }, Is.EqualTo(new[] { 1f, 1f, 2f }).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
                UnityEngine.Object.DestroyImmediate(output);
            }
        }

        [Test]
        public void LayerNormPack4Linear2D_IgnoresTailPaddingLanes()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            using var gamma = new ComputeBuffer(5, sizeof(float));
            using var beta = new ComputeBuffer(5, sizeof(float));
            gamma.SetData(new[] { 1f, 1f, 1f, 1f, 1f });
            beta.SetData(new float[5]);
            var input = CreatePack4Array(new[] { 1f, 2f, 3f, 4f, 5f, 0f, 0f, 0f }, 2, 1, 1);
            var output = CreatePack4Target(2, 1, 1);
            try
            {
                const float epsilon = 1e-5f;
                ops.LayerNormPack4Linear2D(input, 5, 1, epsilon, true, gamma, beta, output);
                var actual = ReadPack4Slice(output, 0).Take(5).ToArray();
                var source = new[] { 1f, 2f, 3f, 4f, 5f };
                var mean = source.Average();
                var variance = source.Select(value => (value - mean) * (value - mean)).Average();
                var expected = source.Select(value => (float)((value - mean) / Math.Sqrt(variance + epsilon))).ToArray();
                Assert.That(actual, Is.EqualTo(expected).Within(2e-5f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
                UnityEngine.Object.DestroyImmediate(output);
            }
        }

        [Test]
        public void PReluPack4_ReusesChannelSlopesAcrossDepthSlices()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            using var slopes = new ComputeBuffer(4, sizeof(float));
            slopes.SetData(new[] { 1f, 2f, 3f, 4f });
            var input = CreatePack4Array(new[]
            {
                -1f, -2f, -3f, -4f,
                -5f, -6f, -7f, -8f
            }, 1, 1, 2);
            var output = CreatePack4Target(1, 1, 2);
            try
            {
                ops.PReluPack4(input, slopes, 4, 1, output);
                Assert.That(ReadPack4Slice(output, 0), Is.EqualTo(new[] { -1f, -4f, -9f, -16f }).Within(1e-6f));
                Assert.That(ReadPack4Slice(output, 1), Is.EqualTo(new[] { -5f, -12f, -21f, -32f }).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
                UnityEngine.Object.DestroyImmediate(output);
            }
        }

        [Test]
        public void LrnPack4_AcrossChannelsMatchesAlphaDivSizeReference()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            var input = CreatePack4Array(new[] { 1f, 1f, 1f, 1f }, 1, 1, 1);
            var output = CreatePack4Target(1, 1, 1);
            try
            {
                ops.LrnPack4(input, 1, 1, 4, 0, 3, 0.3f, 1f, 1f, output);
                var actual = ReadPack4Slice(output, 0);
                Assert.That(actual, Is.EqualTo(new[] { 1f / 1.2f, 1f / 1.3f, 1f / 1.3f, 1f / 1.2f }).Within(2e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
                UnityEngine.Object.DestroyImmediate(output);
            }
        }

        [Test]
        public void NcnnPointwiseAliases_Pack4GpuMatchReferenceParameters()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();

            var profiles = new[]
            {
                (input: new[] { 0f, 1f, 2f, 3f }, type: AexisOps.PointwiseType.Exp, a: 2f, b: 2f, c: -1f, expected: new[] { 0.5f, 2f, 8f, 32f }),
                (input: new[] { 1f, 10f, 100f, 1000f }, type: AexisOps.PointwiseType.Log, a: 10f, b: 1f, c: 0f, expected: new[] { 0f, 1f, 2f, 3f }),
                (input: new[] { 0f, 1f, 2f, 3f }, type: AexisOps.PointwiseType.Power, a: 2f, b: 2f, c: 1f, expected: new[] { 1f, 9f, 25f, 49f }),
                (input: new[] { -1f, 0f, 1f, 2f }, type: AexisOps.PointwiseType.Threshold, a: 1f, b: 0f, c: 0f, expected: new[] { 0f, 0f, 0f, 1f })
            };

            foreach (var profile in profiles)
            {
                var input = CreatePack4Array(profile.input, 1, 1, 1);
                var output = CreatePack4Target(1, 1, 1);
                try
                {
                    ops.PointwisePack4(input, 1, profile.type, profile.a, profile.b, output, profile.c);
                    Assert.That(ReadPack4Slice(output, 0), Is.EqualTo(profile.expected).Within(2e-5f), profile.type.ToString());
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(input);
                    UnityEngine.Object.DestroyImmediate(output);
                }
            }
        }

        [Test]
        public void PointwiseAndUnaryPack4_ZeroTailLanesForEveryDepthSlice()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            var values = new[]
            {
                1f, 2f, 3f, 4f,
                0f, 0f, 0f, 0f,
                5f, 6f, 7f, 8f,
                0f, 0f, 0f, 0f
            };
            var input = CreatePack4Array(values, 1, 1, 4);
            var pointwise = CreatePack4Target(1, 1, 4);
            var unary = CreatePack4Target(1, 1, 4);
            try
            {
                ops.PointwisePack4(input, 4, AexisOps.PointwiseType.Exp, -1f, 1f, pointwise, 0f, logicalChannels: 5);
                ops.UnaryOpPack4(input, 4, 7, unary, logicalChannels: 5);

                foreach (var output in new[] { pointwise, unary })
                {
                    Assert.That(ReadPack4Slice(output, 0), Is.EqualTo(new[] { (float)Math.Exp(1), (float)Math.Exp(2), (float)Math.Exp(3), (float)Math.Exp(4) }).Within(3e-4f));
                    Assert.That(ReadPack4Slice(output, 1), Is.EqualTo(new[] { 1f, 0f, 0f, 0f }).Within(1e-6f));
                    Assert.That(ReadPack4Slice(output, 2), Is.EqualTo(new[] { (float)Math.Exp(5), (float)Math.Exp(6), (float)Math.Exp(7), (float)Math.Exp(8) }).Within(0.25f));
                    Assert.That(ReadPack4Slice(output, 3), Is.EqualTo(new[] { 1f, 0f, 0f, 0f }).Within(1e-6f));
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
                UnityEngine.Object.DestroyImmediate(pointwise);
                UnityEngine.Object.DestroyImmediate(unary);
            }
        }

        [Test]
        public void ScaleCommandBuffer_PreservesFp32FormatAndPublishesDescriptor()
        {
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops);
            using var commandBuffer = new CommandBuffer();
            using var reader = new AexisFloatArrayWeightReader(new[] { 2f });
            var layer = SentisLayer("Scale", new[] { "data" }, new[] { "output" });
            layer.intParams[0] = "1";
            layer.intParams[1] = "0";
            var implementation = AexisLayerFactory.Create(layer);
            implementation.LoadLayer(session, layer, reader);

            var shape = new AexisGraphSession.BufferShape(3, 2, 1, 1, 5);
            var inputTexture = new ComputeTexture
            {
                nameID = Shader.PropertyToID("_AexisScaleFp32Input"),
                width = 2,
                height = 1,
                depth = 2,
                dimension = TextureDimension.Tex2DArray,
                format = RenderTextureFormat.ARGBFloat,
                trackerLabel = "scale-fp32-input"
            };
            var input = AexisGraphSession.CreateCmdTensorRef(inputTexture, shape, shape, owned: false, blobName: "data");
            var context = new AexisLayerCommandBufferContext
            {
                commandBuffer = commandBuffer,
                blobs = new Dictionary<string, AexisGraphSession.CmdTensorRef>(StringComparer.Ordinal) { ["data"] = input },
                shapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal) { ["data"] = shape },
                remaining = new Dictionary<string, int>(StringComparer.Ordinal) { ["data"] = 1 }
            };

            implementation.ExecuteCommandBuffer(session, layer, context);

            Assert.That(context.blobs.TryGetValue("output", out var output), Is.True);
            Assert.That(output.texture.format, Is.EqualTo(RenderTextureFormat.ARGBFloat));
            Assert.That(output.IsDescriptorPublished, Is.True);
            Assert.That(output.Descriptor.DataType, Is.EqualTo(InferenceTensorDataType.Float32));
            Assert.That(output.Descriptor.LogicalShape.c, Is.EqualTo(5));
            Assert.That(output.Descriptor.Packing.PackCount, Is.EqualTo(2));
            session.ReturnTempArray(commandBuffer, output.texture);
        }

        [Test]
        public void StrictPointwisePreflight_RejectsTextureArrayCapacityOverflow()
        {
            var exp = SentisLayer("Exp", new[] { "data" }, new[] { "output" });
            var report = AnalyzeLoadedLayer(
                exp,
                StrictTextureRequest(new[] { Texture("data", 3, 1, 1, 1, int.MaxValue) }));

            Assert.That(report.strictEligible, Is.False, DescribeReport(report));
            Assert.That(report.texturePlan.diagnostics, Has.Some.Matches<AexisTextureExecutionPlanDiagnostic>(diagnostic =>
                diagnostic.reason.Contains("Texture2DArray slices")));
        }

        [Test]
        public void MvnPack4_GpuMatchesNcnnAcrossChannelModesAndIgnoresTailLanes()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            using var dummyAffine = new ComputeBuffer(1, sizeof(float));
            dummyAffine.SetData(new[] { 1f });
            var inputValues = new[]
            {
                1f, 2f, 5f, 6f, 3f, 4f, 7f, 8f,
                10f, 1000f, 2000f, 3000f, 14f, 4000f, 5000f, 6000f
            };
            var input = CreatePack4Array(inputValues, 2, 1, 2);
            try
            {
                foreach (var acrossChannels in new[] { false, true })
                {
                    var groups = acrossChannels ? 1 : 5;
                    var statsA = CreatePack4Target(groups, 1, 1);
                    var statsB = CreatePack4Target(groups, 1, 1);
                    var output = CreatePack4Target(2, 1, 2);
                    try
                    {
                        const float epsilon = 0.125f;
                        ops.GroupNormPack4Tex(input, 2, 1, 1, 5, 2, groups, epsilon, dummyAffine, dummyAffine, statsA, statsB, output, true, true);

                        var channelValues = new[]
                        {
                            new[] { 1f, 3f }, new[] { 2f, 4f }, new[] { 5f, 7f }, new[] { 6f, 8f }, new[] { 10f, 14f }
                        };
                        var flattened = channelValues.SelectMany(values => values).ToArray();
                        var globalMean = flattened.Average();
                        var globalVariance = flattened.Select(value => (value - globalMean) * (value - globalMean)).Average();
                        var expected = new float[10];
                        for (var channel = 0; channel < channelValues.Length; channel++)
                        {
                            var mean = acrossChannels ? globalMean : channelValues[channel].Average();
                            var variance = acrossChannels
                                ? globalVariance
                                : channelValues[channel].Select(value => (value - mean) * (value - mean)).Average();
                            for (var x = 0; x < 2; x++)
                                expected[channel * 2 + x] = (float)((channelValues[channel][x] - mean) / (Math.Sqrt(variance) + epsilon));
                        }

                        var firstPack = ReadPack4Slice(output, 0);
                        var tailPack = ReadPack4Slice(output, 1);
                        for (var channel = 0; channel < 4; channel++)
                        for (var x = 0; x < 2; x++)
                            Assert.That(firstPack[x * 4 + channel], Is.EqualTo(expected[channel * 2 + x]).Within(3e-5f), "across=" + acrossChannels + " channel=" + channel + " x=" + x);
                        Assert.That(new[] { tailPack[0], tailPack[4] }, Is.EqualTo(new[] { expected[8], expected[9] }).Within(3e-5f));
                        Assert.That(new[] { tailPack[1], tailPack[2], tailPack[3], tailPack[5], tailPack[6], tailPack[7] }, Is.EqualTo(new float[6]).Within(1e-6f));
                    }
                    finally
                    {
                        UnityEngine.Object.DestroyImmediate(statsA);
                        UnityEngine.Object.DestroyImmediate(statsB);
                        UnityEngine.Object.DestroyImmediate(output);
                    }
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
            }
        }

        [Test]
        public void Pooling1DPack4_GpuMatchesMaxAndAveragePaddingSemantics()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            var input = CreatePack4Array(new[]
            {
                1f, 10f, -1f, 2f,
                2f, 20f, -2f, 4f,
                3f, 30f, -3f, 6f,
                4f, 40f, -4f, 8f
            }, 4, 1, 1);
            try
            {
                var maxOutput = CreatePack4Target(3, 1, 1);
                var includeOutput = CreatePack4Target(3, 1, 1);
                var excludeOutput = CreatePack4Target(3, 1, 1);
                var adaptiveOutput = CreatePack4Target(2, 1, 1);
                try
                {
                    ops.Pooling1DPack4(input, 4, 1, 0, 3, 2, 1, false, false, maxOutput);
                    ops.Pooling1DPack4(input, 4, 1, 1, 3, 2, 1, true, false, includeOutput);
                    ops.Pooling1DPack4(input, 4, 1, 1, 3, 2, 1, false, false, excludeOutput);
                    ops.Pooling1DPack4(input, 4, 1, 1, 1, 1, 0, false, true, adaptiveOutput);

                    Assert.That(ReadPack4Slice(maxOutput, 0), Is.EqualTo(new[]
                    {
                        2f, 20f, -1f, 4f, 4f, 40f, -2f, 8f, 4f, 40f, -4f, 8f
                    }).Within(1e-6f));
                    Assert.That(ReadPack4Slice(includeOutput, 0), Is.EqualTo(new[]
                    {
                        1f, 10f, -1f, 2f, 3f, 30f, -3f, 6f, 4f / 3f, 40f / 3f, -4f / 3f, 8f / 3f
                    }).Within(2e-5f));
                    Assert.That(ReadPack4Slice(excludeOutput, 0), Is.EqualTo(new[]
                    {
                        1.5f, 15f, -1.5f, 3f, 3f, 30f, -3f, 6f, 4f, 40f, -4f, 8f
                    }).Within(2e-5f));
                    Assert.That(ReadPack4Slice(adaptiveOutput, 0), Is.EqualTo(new[]
                    {
                        1.5f, 15f, -1.5f, 3f, 3.5f, 35f, -3.5f, 7f
                    }).Within(2e-5f));
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(maxOutput);
                    UnityEngine.Object.DestroyImmediate(includeOutput);
                    UnityEngine.Object.DestroyImmediate(excludeOutput);
                    UnityEngine.Object.DestroyImmediate(adaptiveOutput);
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
            }
        }

        [Test]
        public void Pooling2DPack4_GpuMatchesAverageIncludeAndExcludePadSemantics()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            var input = CreatePack4Array(new[]
            {
                1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f,
                3f, 3f, 3f, 3f, 4f, 4f, 4f, 4f
            }, 2, 2, 1);
            var excludeOutput = CreatePack4Target(3, 3, 1);
            var includeOutput = CreatePack4Target(3, 3, 1);
            try
            {
                ops.PoolingPack4(input, 1, 2, 2, 1, 1, 1, 1, 1, excludeOutput, false);
                ops.PoolingPack4(input, 1, 2, 2, 1, 1, 1, 1, 1, includeOutput, true);
                var exclude = ReadPack4Slice(excludeOutput, 0).Where((_, index) => index % 4 == 0).ToArray();
                var include = ReadPack4Slice(includeOutput, 0).Where((_, index) => index % 4 == 0).ToArray();
                Assert.That(exclude, Is.EqualTo(new[] { 1f, 1.5f, 2f, 2f, 2.5f, 3f, 3f, 3.5f, 4f }).Within(1e-6f));
                Assert.That(include, Is.EqualTo(new[] { 0.25f, 0.75f, 0.5f, 1f, 2.5f, 1.5f, 0.75f, 1.75f, 1f }).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
                UnityEngine.Object.DestroyImmediate(excludeOutput);
                UnityEngine.Object.DestroyImmediate(includeOutput);
            }
        }

        [Test]
        public void Pooling2D_StrictProfileUsesNcnnPadModeGeometry()
        {
            var pooling = SentisLayer("Pooling", new[] { "data" }, new[] { "output" });
            pooling.intParams[0] = "1";
            pooling.intParams[1] = "3";
            pooling.intParams[11] = "3";
            pooling.intParams[2] = "2";
            pooling.intParams[12] = "2";
            pooling.intParams[5] = "0";
            pooling.intParams[6] = "1";
            var report = AnalyzeLoadedLayer(pooling, StrictTextureRequest(new[] { Texture("data", 3, 4, 4, 1, 4) }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.nodes.Single(node => node.layer == "pooling").outputs[0].logicalShape, Is.EqualTo(new[] { 3, 2, 2, 1, 4 }));
        }

        [Test]
        public void CopyToPack4_GpuSupportsUnalignedChannelOffsetWithoutLaneRaces()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            var destination = CreatePack4Array(new[] { 10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f }, 1, 1, 2);
            var source = CreatePack4Array(new[] { 100f, 101f, 102f, 999f }, 1, 1, 1);
            var output = CreatePack4Target(1, 1, 2);
            try
            {
                ops.CopyPack4(destination, 0, output, 0, 2);
                ops.CopyToPack4(source, 1, 1, 1, 3, 8, 0, 0, 0, 1, output);
                Assert.That(ReadPack4Slice(output, 0), Is.EqualTo(new[] { 10f, 100f, 101f, 102f }).Within(1e-6f));
                Assert.That(ReadPack4Slice(output, 1), Is.EqualTo(new[] { 14f, 15f, 16f, 17f }).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(destination);
                UnityEngine.Object.DestroyImmediate(source);
                UnityEngine.Object.DestroyImmediate(output);
            }
        }

        [Test]
        public void NcnnOneDimensionalConvolutions_LoadedTexturePathsMatchReference()
        {
            RequireArgbFloatCompute();

            var depthwise = new AexisGraphModel.Layer
            {
                name = "depthwise1d",
                typeName = "ConvolutionDepthWise1D",
                type = AexisLayerTypeKey.FromString("ConvolutionDepthWise1D"),
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "input" },
                topNames = new[] { "output" },
                intParams =
                {
                    [0] = "4", [1] = "3", [2] = "1", [3] = "1", [4] = "1", [15] = "1",
                    [5] = "1", [6] = "12", [7] = "4", [9] = "0"
                }
            };
            var depthwiseWeights = Enumerable.Range(0, 4).SelectMany(_ => new[] { 1f, 0f, -1f })
                .Concat(new[] { 0f, 1f, 2f, 3f }).ToArray();
            var depthwiseInput = CreateLinearMat(new[]
            {
                1f, 2f, 3f, 4f,
                10f, 20f, 30f, 40f,
                100f, 200f, 300f, 400f,
                1000f, 2000f, 3000f, 4000f
            }, 4, 4);
            ExecuteLoadedLinearLayerAndAssert(
                depthwise,
                depthwiseWeights,
                depthwiseInput,
                new AexisGraphSession.BufferShape(2, 4, 4, 1, 1),
                new[]
                {
                    -2f, -2f, -2f, 3f,
                    -19f, -19f, -19f, 31f,
                    -198f, -198f, -198f, 302f,
                    -1997f, -1997f, -1997f, 3003f
                });
            Assert.That(AnalyzeLoadedLayer(depthwise, StrictTextureRequest(new[] { LinearTexture("input", 2, 4, 4, 1, 1) }), depthwiseWeights).strictEligible, Is.True);
            Assert.That(AnalyzeLoadedLayer(depthwise, StrictTextureRequest(new[] { Texture("input", 3, 4, 1, 1, 4) }), depthwiseWeights).strictEligible, Is.False);

            var deconvolution = new AexisGraphModel.Layer
            {
                name = "deconvolution1d",
                typeName = "Deconvolution1D",
                type = AexisLayerTypeKey.FromString("Deconvolution1D"),
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "input" },
                topNames = new[] { "output" },
                intParams =
                {
                    [0] = "4", [1] = "3", [2] = "1", [3] = "2", [4] = "1", [15] = "1",
                    [5] = "1", [6] = "48", [9] = "0", [18] = "0", [20] = "0", [28] = "0"
                }
            };
            var deconvolutionWeights = new List<float>(48);
            for (var outputChannel = 0; outputChannel < 4; outputChannel++)
            for (var inputChannel = 0; inputChannel < 4; inputChannel++)
                deconvolutionWeights.AddRange(outputChannel == inputChannel ? new[] { 0f, 1f, 0f } : new[] { 0f, 0f, 0f });
            deconvolutionWeights.AddRange(new[] { 0f, 1f, 2f, 3f });
            var deconvolutionInput = CreateLinearMat(new[]
            {
                1f, 2f, 3f,
                10f, 20f, 30f,
                100f, 200f, 300f,
                1000f, 2000f, 3000f
            }, 3, 4);
            ExecuteLoadedLinearLayerAndAssert(
                deconvolution,
                deconvolutionWeights.ToArray(),
                deconvolutionInput,
                new AexisGraphSession.BufferShape(2, 3, 4, 1, 1),
                new[]
                {
                    1f, 0f, 2f, 0f, 3f,
                    11f, 1f, 21f, 1f, 31f,
                    102f, 2f, 202f, 2f, 302f,
                    1003f, 3f, 2003f, 3f, 3003f
                },
                enableGeneralConvolution: true);
            var deconvolutionReport = AnalyzeLoadedLayer(deconvolution, StrictTextureRequest(new[] { LinearTexture("input", 2, 3, 4, 1, 1) }), deconvolutionWeights.ToArray());
            Assert.That(deconvolutionReport.strictEligible, Is.True, DescribeReport(deconvolutionReport));
            Assert.That(AnalyzeLoadedLayer(deconvolution, StrictTextureRequest(new[] { Texture("input", 3, 3, 1, 1, 4) }), deconvolutionWeights.ToArray()).strictEligible, Is.False);
        }

        [Test]
        public void ConvolutionDepthWise3D_CommandBufferPack4CdhwMatchesTailChannelGolden()
        {
            RequireArgbFloatCompute();
            var layer = new AexisGraphModel.Layer
            {
                name = "depthwise3d_pack4_golden",
                typeName = "ConvolutionDepthWise3D",
                type = AexisLayerTypes.ConvDw3D,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams =
                {
                    [0] = "5", [1] = "1", [11] = "1", [21] = "1",
                    [2] = "1", [12] = "1", [22] = "1",
                    [3] = "1", [13] = "1", [23] = "1",
                    [4] = "0", [14] = "0", [15] = "0", [16] = "0", [17] = "0", [24] = "0",
                    [5] = "1", [6] = "5", [7] = "5", [9] = "0"
                }
            };
            var immutableWeights = new[]
            {
                2f, 3f, 4f, 5f, 6f,
                0.5f, 1.5f, 2.5f, 3.5f, 4.5f
            };
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(immutableWeights);
            session.LoadModel(SerializeSingleLayer(layer), reader);
            using var commandBuffer = new CommandBuffer { name = "AexisDepthWise3dPack4CdhwGolden" };
            var inputUpload = CreatePack4Array(new[]
            {
                1f, 2f, 3f, 4f, 10f, 20f, 30f, 40f,
                5f, 999f, 999f, 999f, 50f, 999f, 999f, 999f,
                6f, 7f, 8f, 9f, 60f, 70f, 80f, 90f,
                10f, 999f, 999f, 999f, 100f, 999f, 999f, 999f
            }, 2, 1, 4);
            var outputReadback = CreatePack4Target(2, 1, 4);
            ComputeTexture input = null;
            try
            {
                input = session.RentTempArray(commandBuffer, 2, 1, 4, RenderTextureFormat.ARGBFloat);
                for (var slice = 0; slice < 4; slice++)
                    commandBuffer.CopyTexture(inputUpload, slice, 0, input.nameID, slice, 0);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(4, 2, 1, 2, 5)
                    },
                    new[] { "output" }))
                {
                    Assert.That(result.GetLogicalShape("output"), Is.EqualTo(new AexisGraphSession.BufferShape(4, 2, 1, 2, 5)));
                    for (var slice = 0; slice < 4; slice++)
                        commandBuffer.CopyTexture(result.GetTexture("output").nameID, slice, 0, outputReadback, slice, 0);
                }
                session.ReturnTempArray(commandBuffer, input); input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().executionPath,
                    Is.EqualTo("command-buffer-pack4:depthwise-convolution3d-cdhw"));
                Assert.That(ReadPack4Slice(outputReadback, 0), Is.EqualTo(new[]
                {
                    2.5f, 7.5f, 14.5f, 23.5f, 20.5f, 61.5f, 122.5f, 203.5f
                }).Within(1e-5f));
                Assert.That(ReadPack4Slice(outputReadback, 1), Is.EqualTo(new[]
                {
                    34.5f, 0f, 0f, 0f, 304.5f, 0f, 0f, 0f
                }).Within(1e-5f));
                Assert.That(ReadPack4Slice(outputReadback, 2), Is.EqualTo(new[]
                {
                    12.5f, 22.5f, 34.5f, 48.5f, 120.5f, 211.5f, 322.5f, 453.5f
                }).Within(1e-5f));
                Assert.That(ReadPack4Slice(outputReadback, 3), Is.EqualTo(new[]
                {
                    64.5f, 0f, 0f, 0f, 604.5f, 0f, 0f, 0f
                }).Within(1e-5f));
            }
            finally
            {
                if (input != null) session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void DeconvolutionDepthWise3D_CommandBufferPack4CdhwMatchesStrideAndTailGolden()
        {
            RequireArgbFloatCompute();
            var layer = new AexisGraphModel.Layer
            {
                name = "deconvolution_depthwise3d_pack4_golden",
                typeName = "DeconvolutionDepthWise3D",
                type = AexisLayerTypes.DeconvDw3D,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams =
                {
                    [0] = "5", [1] = "1", [11] = "1", [21] = "1",
                    [2] = "1", [12] = "1", [22] = "1",
                    [3] = "2", [13] = "1", [23] = "2",
                    [4] = "0", [14] = "0", [15] = "0", [16] = "0", [17] = "0", [24] = "0",
                    [18] = "0", [19] = "0", [20] = "0",
                    [5] = "1", [6] = "5", [7] = "5", [9] = "0"
                }
            };
            var immutableWeights = new[]
            {
                2f, 3f, 4f, 5f, 6f,
                0.5f, 1.5f, 2.5f, 3.5f, 4.5f
            };
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(immutableWeights);
            session.LoadModel(SerializeSingleLayer(layer), reader);
            using var commandBuffer = new CommandBuffer { name = "AexisDeconvDepthWise3dPack4CdhwGolden" };
            var inputUpload = CreatePack4Array(new[]
            {
                1f, 2f, 3f, 4f, 10f, 20f, 30f, 40f,
                5f, 0f, 0f, 0f, 50f, 0f, 0f, 0f,
                6f, 7f, 8f, 9f, 60f, 70f, 80f, 90f,
                10f, 0f, 0f, 0f, 100f, 0f, 0f, 0f
            }, 2, 1, 4);
            var outputReadback = CreatePack4Target(3, 1, 6);
            ComputeTexture input = null;
            try
            {
                input = session.RentTempArray(commandBuffer, 2, 1, 4, RenderTextureFormat.ARGBFloat);
                for (var slice = 0; slice < 4; slice++)
                    commandBuffer.CopyTexture(inputUpload, slice, 0, input.nameID, slice, 0);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(4, 2, 1, 2, 5)
                    },
                    new[] { "output" }))
                {
                    Assert.That(result.GetLogicalShape("output"), Is.EqualTo(new AexisGraphSession.BufferShape(4, 3, 1, 3, 5)));
                    for (var slice = 0; slice < 6; slice++)
                        commandBuffer.CopyTexture(result.GetTexture("output").nameID, slice, 0, outputReadback, slice, 0);
                }
                session.ReturnTempArray(commandBuffer, input); input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().executionPath,
                    Is.EqualTo("command-buffer-pack4:depthwise-deconvolution3d-cdhw"));
                Assert.That(ReadPack4Slice(outputReadback, 0), Is.EqualTo(new[]
                {
                    2.5f, 7.5f, 14.5f, 23.5f, 0.5f, 1.5f, 2.5f, 3.5f, 20.5f, 61.5f, 122.5f, 203.5f
                }).Within(1e-5f));
                Assert.That(ReadPack4Slice(outputReadback, 1), Is.EqualTo(new[]
                {
                    34.5f, 0f, 0f, 0f, 4.5f, 0f, 0f, 0f, 304.5f, 0f, 0f, 0f
                }).Within(1e-5f));
                Assert.That(ReadPack4Slice(outputReadback, 2), Is.EqualTo(new[]
                {
                    0.5f, 1.5f, 2.5f, 3.5f, 0.5f, 1.5f, 2.5f, 3.5f, 0.5f, 1.5f, 2.5f, 3.5f
                }).Within(1e-5f));
                Assert.That(ReadPack4Slice(outputReadback, 3), Is.EqualTo(new[]
                {
                    4.5f, 0f, 0f, 0f, 4.5f, 0f, 0f, 0f, 4.5f, 0f, 0f, 0f
                }).Within(1e-5f));
                Assert.That(ReadPack4Slice(outputReadback, 4), Is.EqualTo(new[]
                {
                    12.5f, 22.5f, 34.5f, 48.5f, 0.5f, 1.5f, 2.5f, 3.5f, 120.5f, 211.5f, 322.5f, 453.5f
                }).Within(1e-5f));
                Assert.That(ReadPack4Slice(outputReadback, 5), Is.EqualTo(new[]
                {
                    64.5f, 0f, 0f, 0f, 4.5f, 0f, 0f, 0f, 604.5f, 0f, 0f, 0f
                }).Within(1e-5f));
            }
            finally
            {
                if (input != null) session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void DeconvolutionDepthWise1D_CommandBufferPack4MatchesStrideAndTailGolden()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat LinearMat profile required by this golden.");
            var layer = new AexisGraphModel.Layer
            {
                name = "deconvolution_depthwise1d_pack4_golden",
                typeName = "DeconvolutionDepthWise1D",
                type = AexisLayerTypes.DeconvDw1D,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams =
                {
                    [0] = "5", [1] = "1", [2] = "1", [3] = "2", [4] = "0", [15] = "0",
                    [5] = "1", [6] = "5", [7] = "5", [9] = "0", [18] = "0", [20] = "0", [28] = "0"
                }
            };
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(new[]
            {
                2f, 3f, 4f, 5f, 6f,
                0.5f, 1.5f, 2.5f, 3.5f, 4.5f
            });
            session.LoadModel(SerializeSingleLayer(layer), reader);
            using var upload = new ComputeBuffer(10, sizeof(float));
            upload.SetData(new[] { 1f, 10f, 2f, 20f, 3f, 30f, 4f, 40f, 5f, 50f });
            using var commandBuffer = new CommandBuffer { name = "AexisDeconvDepthWise1dPack4Golden" };
            var input = session.RentTempMat(commandBuffer, 2, 5, RenderTextureFormat.RFloat);
            var outputReadback = CreateRFloatTarget(3, 5);
            try
            {
                ops.FillLinearMatFromBuffer(commandBuffer, upload, 2, 5, input);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(2, 2, 5, 1, 1)
                    },
                    new[] { "output" }))
                {
                    Assert.That(result.GetLogicalShape("output"), Is.EqualTo(new AexisGraphSession.BufferShape(2, 3, 5, 1, 1)));
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, new RenderTargetIdentifier(outputReadback));
                }
                session.ReturnTempArray(commandBuffer, input);
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().executionPath,
                    Is.EqualTo("command-buffer-pack4:depthwise-deconvolution1d"));
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().scratch, Has.Length.EqualTo(2));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.EqualTo(0));
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                Assert.That(ReadRFloat(outputReadback, 15), Is.EqualTo(new[]
                {
                    2.5f, 0.5f, 20.5f,
                    7.5f, 1.5f, 61.5f,
                    14.5f, 2.5f, 122.5f,
                    23.5f, 3.5f, 203.5f,
                    34.5f, 4.5f, 304.5f
                }).Within(1e-5f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void StatisticsPooling_CommandBufferLinearMatMatchesMeanAndPopulationStdGolden()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat LinearMat profile required by this golden.");

            var layer = new AexisGraphModel.Layer
            {
                name = "statistics_pooling_pack4_golden",
                typeName = "StatisticsPooling",
                type = AexisLayerTypes.StatsPooling,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams = { [0] = "1", [1] = "0" }
            };
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(SerializeSingleLayer(layer), reader);
            using var upload = new ComputeBuffer(8, sizeof(float));
            upload.SetData(new[] { 1f, 3f, 5f, 7f, 2f, 4f, 6f, 8f });
            using var commandBuffer = new CommandBuffer { name = "AexisStatisticsPoolingPack4Golden" };
            var input = session.RentTempMat(commandBuffer, 4, 2, RenderTextureFormat.RFloat);
            var outputReadback = CreateRFloatTarget(1, 4);
            try
            {
                ops.FillLinearMatFromBuffer(commandBuffer, upload, 4, 2, input);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(2, 4, 2, 1, 1)
                    },
                    new[] { "output" }))
                {
                    Assert.That(result.GetLogicalShape("output"), Is.EqualTo(new AexisGraphSession.BufferShape(2, 1, 4, 1, 1)));
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, new RenderTargetIdentifier(outputReadback));
                }
                session.ReturnTempArray(commandBuffer, input);
                input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().executionPath,
                    Is.EqualTo("command-buffer-pack4:statistics-pooling-linearmat"));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.EqualTo(0));
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                Assert.That(ReadRFloat(outputReadback, 4), Is.EqualTo(new[]
                {
                    4f, 5f, Mathf.Sqrt(5f), Mathf.Sqrt(5f)
                }).Within(1e-5f));
            }
            finally
            {
                if (input != null)
                    session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void SpectrogramAndInverseSpectrogram_CommandBufferLinearMatRoundTripsGolden()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat LinearMat profile required by this golden.");

            var spectrogram = new AexisGraphModel.Layer
            {
                name = "spectrogram_pack4_golden",
                typeName = "Spectrogram",
                type = AexisLayerTypes.Spectrogram,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "spectrum" },
                intParams = { [0] = "4", [1] = "2", [2] = "1" }
            };
            var inverse = new AexisGraphModel.Layer
            {
                name = "inverse_spectrogram_pack4_golden",
                typeName = "InverseSpectrogram",
                type = AexisLayerTypes.InvSpectrogram,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "spectrum" },
                topNames = new[] { "output" },
                intParams = { [0] = "4", [1] = "2", [2] = "1" }
            };
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(SerializeLayers(spectrogram, inverse), reader);
            using var upload = new ComputeBuffer(6, sizeof(float));
            upload.SetData(new[] { 1f, 2f, 3f, 4f, 5f, 6f });
            using var commandBuffer = new CommandBuffer { name = "AexisSpectrogramPack4Golden" };
            var input = session.RentTempMat(commandBuffer, 6, 1, RenderTextureFormat.RFloat);
            var outputReadback = CreateRFloatTarget(6, 1);
            try
            {
                ops.FillLinearMatFromBuffer(commandBuffer, upload, 6, 1, input);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(2, 6, 1, 1, 1)
                    },
                    new[] { "output" }))
                {
                    Assert.That(result.GetLogicalShape("output"), Is.EqualTo(new AexisGraphSession.BufferShape(2, 6, 1, 1, 1)));
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, new RenderTargetIdentifier(outputReadback));
                }
                session.ReturnTempArray(commandBuffer, input);
                input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().executionPath,
                    Is.EqualTo("command-buffer-pack4:inverse-spectrogram-linearmat"));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.EqualTo(0));
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                Assert.That(ReadRFloat(outputReadback, 6), Is.EqualTo(new[] { 1f, 2f, 3f, 4f, 5f, 6f }).Within(2e-4f));
            }
            finally
            {
                if (input != null)
                    session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void Rnn_CommandBufferPack4MatchesBoundedForwardGolden()
        {
            ExecuteBoundedRecurrentGolden(
                "RNN", AexisLayerTypes.RNN, AexisRecurrentKind.Rnn, inputSize: 2, hiddenSize: 2,
                immutableWeights: new[]
                {
                    0.5f, -0.25f, 0.1f, 0.4f,
                    0.2f, 0f, 0f, 0.3f,
                    0.1f, -0.2f
                },
                inputByChannel: new[] { 1f, 2f, 3f, 4f, 5f, 6f });
        }

        [Test]
        public void Gru_CommandBufferPack4MatchesBoundedForwardGolden()
        {
            ExecuteBoundedRecurrentGolden(
                "GRU", AexisLayerTypes.GRU, AexisRecurrentKind.Gru, inputSize: 1, hiddenSize: 1,
                immutableWeights: new[]
                {
                    0f, 0f, 1f,
                    0f, 0f, 0f,
                    0f, 0f, 0f
                },
                inputByChannel: new[] { 1f, 2f });
        }

        [Test]
        public void Lstm_CommandBufferPack4MatchesBoundedForwardGolden()
        {
            ExecuteBoundedRecurrentGolden(
                "LSTM", AexisLayerTypes.LSTM, AexisRecurrentKind.Lstm, inputSize: 1, hiddenSize: 1,
                immutableWeights: new[]
                {
                    0f, 0f, 0f, 1f,
                    0f, 0f, 0f, 0f,
                    0f, 0f, 0f, 0f
                },
                inputByChannel: new[] { 1f, 2f });
        }

        [Test]
        public void DeterministicRandomUniformLike_CommandBufferPack4IsSeedReproducibleAndZerosTailLanes()
        {
            RequireArgbFloatCompute();
            var layer = new AexisGraphModel.Layer
            {
                name = "seeded_uniform",
                typeName = "RandomUniformLike",
                type = AexisLayerTypes.RandomLike,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams = { [0] = "20260728", [1] = "-2", [2] = "3" }
            };
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(SerializeSingleLayer(layer), reader);
            var inputUpload = CreatePack4Array(new[]
            {
                0f, 0f, 0f, 0f, 0f, 99f, 99f, 99f,
                0f, 0f, 0f, 0f, 0f, 99f, 99f, 99f
            }, 2, 1, 2);
            var firstReadback = CreatePack4Target(2, 1, 2);
            var secondReadback = CreatePack4Target(2, 1, 2);
            try
            {
                ExecuteSeededRandomPass(session, inputUpload, firstReadback, "AexisSeededRandomFirst");
                ExecuteSeededRandomPass(session, inputUpload, secondReadback, "AexisSeededRandomSecond");

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().executionPath, Is.EqualTo("command-buffer-pack4:deterministic-rng"));
                for (var slice = 0; slice < 2; slice++)
                {
                    var first = ReadPack4Slice(firstReadback, slice);
                    var second = ReadPack4Slice(secondReadback, slice);
                    Assert.That(second, Is.EqualTo(first).Within(0f));
                    for (var index = 0; index < first.Length; index += 4)
                    {
                        Assert.That(first[index], Is.GreaterThanOrEqualTo(-2f).And.LessThan(3f));
                        Assert.That(first[index + 1], Is.GreaterThanOrEqualTo(-2f).And.LessThan(3f));
                        Assert.That(first[index + 2], Is.GreaterThanOrEqualTo(-2f).And.LessThan(3f));
                        Assert.That(first[index + 3], Is.GreaterThanOrEqualTo(-2f).And.LessThan(3f));
                    }
                }
                var tail = ReadPack4Slice(firstReadback, 1);
                Assert.That(new[] { tail[1], tail[2], tail[3], tail[5], tail[6], tail[7] },
                    Is.EqualTo(new[] { 0f, 0f, 0f, 0f, 0f, 0f }).Within(0f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(firstReadback);
                UnityEngine.Object.DestroyImmediate(secondReadback);
            }
        }

        [Test]
        public void RandomUniform_CommandBufferStaticPack4ProfileUsesSeededGpuOutput()
        {
            RequireArgbFloatCompute();
            var layer = new AexisGraphModel.Layer
            {
                name = "static_random_uniform_pack4_golden",
                typeName = "RandomUniform",
                type = AexisLayerTypes.RandomLike,
                bottoms = 0,
                tops = 1,
                bottomNames = Array.Empty<string>(),
                topNames = new[] { "output" },
                intParams = { [0] = "1234", [1] = "-2", [2] = "3", [10] = "3", [11] = "2", [12] = "2", [13] = "1", [14] = "3" }
            };
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(SerializeSingleLayer(layer), reader);
            using var commandBuffer = new CommandBuffer { name = "AexisStaticRandomUniformPack4Golden" };
            var outputReadback = CreatePack4Target(2, 2, 1);
            try
            {
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal),
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal),
                    new[] { "output" }))
                {
                    Assert.That(result.GetLogicalShape("output"), Is.EqualTo(new AexisGraphSession.BufferShape(3, 2, 2, 1, 3)));
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, 0, 0, outputReadback, 0, 0);
                }
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().executionPath, Is.EqualTo("command-buffer-pack4:deterministic-static-rng"));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.EqualTo(0));
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                var values = ReadPack4Slice(outputReadback, 0);
                for (var pixel = 0; pixel < 4; pixel++)
                {
                    Assert.That(values[pixel * 4], Is.InRange(-2f, 3f));
                    Assert.That(values[pixel * 4 + 1], Is.InRange(-2f, 3f));
                    Assert.That(values[pixel * 4 + 2], Is.InRange(-2f, 3f));
                    Assert.That(values[pixel * 4 + 3], Is.EqualTo(0f).Within(1e-7f));
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void Multinomial_CommandBufferPack4UsesSeededGpuIndicesAndZerosTailLanes()
        {
            RequireArgbFloatCompute();
            var layer = new AexisGraphModel.Layer
            {
                name = "bounded_multinomial_pack4_golden",
                typeName = "Multinomial",
                type = AexisLayerTypes.Multinomial,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "logits" },
                topNames = new[] { "samples" },
                intParams = { [0] = "5", [1] = "20260728" }
            };
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(SerializeSingleLayer(layer), reader);
            var upload = CreatePack4Array(new[]
            {
                -100f, -100f, 100f, -100f,
                100f, -100f, -100f, -100f,
                -100f, 99f, 99f, 99f,
                -100f, 99f, 99f, 99f
            }, 1, 2, 2);
            var readback = CreatePack4Target(1, 2, 2);
            ComputeTexture input = null;
            try
            {
                using var commandBuffer = new CommandBuffer { name = "AexisMultinomialPack4Golden" };
                input = session.RentTempArray(commandBuffer, 1, 2, 2, RenderTextureFormat.ARGBFloat);
                commandBuffer.CopyTexture(upload, 0, 0, input.nameID, 0, 0);
                commandBuffer.CopyTexture(upload, 1, 0, input.nameID, 1, 0);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["logits"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["logits"] = new AexisGraphSession.BufferShape(3, 1, 2, 1, 5)
                    },
                    new[] { "samples" }))
                {
                    Assert.That(result.GetLogicalShape("samples"), Is.EqualTo(new AexisGraphSession.BufferShape(3, 1, 2, 1, 5)));
                    commandBuffer.CopyTexture(result.GetTexture("samples").nameID, 0, 0, readback, 0, 0);
                    commandBuffer.CopyTexture(result.GetTexture("samples").nameID, 1, 0, readback, 1, 0);
                }
                session.ReturnTempArray(commandBuffer, input);
                input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().executionPath, Is.EqualTo("command-buffer-pack4:bounded-multinomial"));
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().outputs[0].logicalDtype, Is.EqualTo("Int32"));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.EqualTo(0));
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                var firstPack = ReadPack4Slice(readback, 0);
                var tailPack = ReadPack4Slice(readback, 1);
                Assert.That(new[] { firstPack[0], firstPack[1], firstPack[2], firstPack[3], tailPack[0] }, Is.EqualTo(new[] { 2f, 2f, 2f, 2f, 2f }).Within(0f));
                Assert.That(new[] { firstPack[4], firstPack[5], firstPack[6], firstPack[7], tailPack[4] }, Is.EqualTo(new[] { 0f, 0f, 0f, 0f, 0f }).Within(0f));
                Assert.That(new[] { tailPack[1], tailPack[2], tailPack[3], tailPack[5], tailPack[6], tailPack[7] },
                    Is.EqualTo(new[] { 0f, 0f, 0f, 0f, 0f, 0f }).Within(0f));
            }
            finally
            {
                if (input != null)
                {
                    using var cleanup = new CommandBuffer { name = "AexisMultinomialPack4Cleanup" };
                    session.ReturnTempArray(cleanup, input);
                    Graphics.ExecuteCommandBuffer(cleanup);
                }
                UnityEngine.Object.DestroyImmediate(upload);
                UnityEngine.Object.DestroyImmediate(readback);
            }
        }

        private static void ExecuteSeededRandomPass(AexisGraphSession session, RenderTexture inputUpload, RenderTexture readback, string commandName)
        {
            using var commandBuffer = new CommandBuffer { name = commandName };
            var input = session.RentTempArray(commandBuffer, 2, 1, 2, RenderTextureFormat.ARGBFloat);
            try
            {
                commandBuffer.CopyTexture(inputUpload, 0, 0, input.nameID, 0, 0);
                commandBuffer.CopyTexture(inputUpload, 1, 0, input.nameID, 1, 0);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(3, 2, 1, 1, 5)
                    },
                    new[] { "output" }))
                {
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, 0, 0, readback, 0, 0);
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, 1, 0, readback, 1, 0);
                }
                session.ReturnTempArray(commandBuffer, input);
                input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);
            }
            finally
            {
                if (input != null) session.ReturnTempArray(commandBuffer, input);
            }
        }

        [Test]
        public void PaddingAndReorg_StrictProfilesRejectUnsafeDescriptorEdges()
        {
            var padding = SentisLayer("Padding", new[] { "data" }, new[] { "output" });
            padding.intParams[2] = "-1";
            Assert.That(AnalyzeLoadedLayer(padding, StrictTextureRequest(new[] { Texture("data", 3, 4, 4, 1, 4) })).strictEligible, Is.False);

            padding = SentisLayer("Padding", new[] { "data" }, new[] { "output" });
            padding.intParams[4] = "2";
            padding.intParams[2] = "1";
            Assert.That(AnalyzeLoadedLayer(padding, StrictTextureRequest(new[] { Texture("data", 3, 1, 4, 1, 4) })).strictEligible, Is.False);

            padding = SentisLayer("Padding", new[] { "data" }, new[] { "output" });
            padding.intParams[4] = "2";
            padding.intParams[2] = "4";
            Assert.That(AnalyzeLoadedLayer(padding, StrictTextureRequest(new[] { Texture("data", 3, 4, 4, 1, 4) })).strictEligible, Is.False);

            var reorg = SentisLayer("Reorg", new[] { "data" }, new[] { "output" });
            reorg.intParams[0] = "2";
            reorg.intParams[1] = "0";
            var channelOverflow = int.MaxValue / 4 + 1;
            var report = AnalyzeLoadedLayer(
                reorg,
                StrictTextureRequest(new[] { Texture("data", 3, 2, 2, 1, channelOverflow) }));
            Assert.That(report.strictEligible, Is.False);
            Assert.That(report.texturePlan.diagnostics.Any(diagnostic =>
                diagnostic != null && diagnostic.reason != null && diagnostic.reason.Contains("descriptor range")), Is.True,
                DescribeReport(report));
        }

        [Test]
        public void DataIndexLinearMat_GpuMatchesBoundedAndUniqueReferenceContracts()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat compute profile required by this golden.");

            using var ops = new AexisOps();
            var vectorShape = new AexisGraphSession.BufferShape(1, 4, 1, 1, 1);
            var vectorStorage = new AexisGraphSession.BufferShape(2, 4, 1, 1, 1);
            var data = CreateLinearMat(new[] { 10f, 20f, 30f, 40f });
            var nonZeroInput = CreateLinearMat(new[] { 0f, 2f, 0f, -1f });
            var condition = CreateLinearMat(new[] { 0f, 1f, 1f, 0f });
            var indices = CreateLinearMat(new[] { 3f, 1f }, 1, 2);
            var updates = CreateLinearMat(new[] { 7f, 8f });
            var nonZeroOutput = CreateRFloatTarget(4, 1);
            var nonZeroCount = CreateRFloatTarget(1, 1);
            var compressOutput = CreateRFloatTarget(4, 1);
            var compressCount = CreateRFloatTarget(1, 1);
            var gatherOutput = CreateRFloatTarget(2, 1);
            var scatterOutput = CreateRFloatTarget(4, 1);
            try
            {
                ops.AexisNonZeroLinearMat(nonZeroInput, vectorShape, vectorStorage, 4, nonZeroOutput, nonZeroCount);
                ops.AexisCompressLinearMat(nonZeroInput, vectorShape, vectorStorage, condition, vectorShape, vectorStorage, 4, compressOutput, compressCount);

                var indexShape = new AexisGraphSession.BufferShape(2, 1, 2, 1, 1);
                var indexStorage = new AexisGraphSession.BufferShape(2, 1, 2, 1, 1);
                var outputShape = new AexisGraphSession.BufferShape(1, 2, 1, 1, 1);
                var outputStorage = new AexisGraphSession.BufferShape(2, 2, 1, 1, 1);
                ops.AexisGatherNdLinearMat(data, vectorShape, vectorStorage, indices, indexShape, indexStorage, outputShape, outputStorage, gatherOutput);
                ops.AexisScatterLinearMat(data, vectorShape, vectorStorage, indices, indexShape, indexStorage, updates, outputShape, outputStorage, 2, scatterOutput);

                Assert.That(ReadRFloat(nonZeroCount, 1)[0], Is.EqualTo(2f));
                Assert.That(ReadRFloat(nonZeroOutput, 2), Is.EqualTo(new[] { 1f, 3f }).Within(1e-6f));
                Assert.That(ReadRFloat(compressCount, 1)[0], Is.EqualTo(2f));
                Assert.That(ReadRFloat(compressOutput, 2), Is.EqualTo(new[] { 2f, 0f }).Within(1e-6f));
                Assert.That(ReadRFloat(gatherOutput, 2), Is.EqualTo(new[] { 40f, 20f }).Within(1e-6f));
                Assert.That(ReadRFloat(scatterOutput, 4), Is.EqualTo(new[] { 10f, 8f, 30f, 7f }).Within(1e-6f));
            }
            finally
            {
                foreach (var texture in new[] { data, nonZeroInput, condition, indices, updates, nonZeroOutput, nonZeroCount, compressOutput, compressCount, gatherOutput, scatterOutput })
                    UnityEngine.Object.DestroyImmediate(texture);
            }
        }

        [Test]
        public void GatherAndScatter_RequireAndAcceptDeclaredLinearProfiles()
        {
            var gather = new AexisGraphModel.Layer
            {
                name = "gather",
                typeName = "GatherND",
                type = AexisLayerTypeKey.FromString("GatherND"),
                bottoms = 2,
                tops = 1,
                bottomNames = new[] { "data", "indices" },
                topNames = new[] { "result" },
                intParams = { [0] = "0", [1] = "1" },
                stringParams = { ["batch_dims"] = "0", ["index_depth"] = "1", ["index_dtype"] = "Int32", ["indices_in_range"] = "1" }
            };
            var scatter = new AexisGraphModel.Layer
            {
                name = "scatter",
                typeName = "ScatterND",
                type = AexisLayerTypeKey.FromString("ScatterND"),
                bottoms = 3,
                tops = 1,
                bottomNames = new[] { "data", "indices", "updates" },
                topNames = new[] { "result" },
                intParams = { [-1] = "1" },
                stringParams = { ["unique_indices"] = "1", ["indices_in_range"] = "1", ["index_dtype"] = "Int32", ["reduction"] = "none", ["index_depth"] = "1" }
            };

            var gatherModel = new AexisGraphModel();
            gatherModel.layers.Add(gather);
            var gatherRequest = StrictTextureRequest(new[]
            {
                LinearTexture("data", 1, 8, 1, 1, 1),
                LinearTexture("indices", 2, 1, 3, 1, 1, "Int32")
            });
            var gatherReport = AnalyzeLoadedLayer(gather, gatherRequest);
            Assert.That(gatherReport.strictEligible, Is.True, DescribeReport(gatherReport));
            Assert.That(gatherReport.nodes[0].outputs[0].logicalShape, Is.EqualTo(new[] { 1, 3, 1, 1, 1 }));

            var scatterModel = new AexisGraphModel();
            scatterModel.layers.Add(scatter);
            var scatterReport = AnalyzeLoadedLayer(scatter, StrictTextureRequest(new[]
            {
                LinearTexture("data", 1, 8, 1, 1, 1),
                LinearTexture("indices", 2, 1, 3, 1, 1, "Int32"),
                LinearTexture("updates", 1, 3, 1, 1, 1)
            }));
            Assert.That(scatterReport.strictEligible, Is.True, DescribeReport(scatterReport));
        }

        [Test]
        public void GatherNd_StrictProfileRejectsMissingDtypeOrRangeProof()
        {
            var layer = GatherNdLayer();
            var validInputs = new[]
            {
                LinearTexture("data", 1, 8, 1, 1, 1),
                LinearTexture("indices", 2, 1, 3, 1, 1, "Int32")
            };

            layer.stringParams.Remove("indices_in_range");
            Assert.That(AnalyzeLoadedLayer(layer, StrictTextureRequest(validInputs)).strictEligible, Is.False);

            layer = GatherNdLayer();
            var floatIndices = new[]
            {
                LinearTexture("data", 1, 8, 1, 1, 1),
                LinearTexture("indices", 2, 1, 3, 1, 1, "Float32")
            };
            Assert.That(AnalyzeLoadedLayer(layer, StrictTextureRequest(floatIndices)).strictEligible, Is.False);
        }

        [Test]
        public void ScatterNd_StrictProfileRejectsMissingConflictRangeOrReductionProof()
        {
            var inputs = new[]
            {
                LinearTexture("data", 1, 8, 1, 1, 1),
                LinearTexture("indices", 2, 1, 3, 1, 1, "Int32"),
                LinearTexture("updates", 1, 3, 1, 1, 1)
            };

            foreach (var proof in new[] { "unique_indices", "indices_in_range", "reduction" })
            {
                var layer = ScatterNdLayer();
                layer.stringParams.Remove(proof);
                Assert.That(AnalyzeLoadedLayer(layer, StrictTextureRequest(inputs)).strictEligible, Is.False, proof);
            }
        }

        [Test]
        public void OnnxLowering_GatherFamilyRequiresExplicitInRangeProof()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("data", TensorDataType.Float32, 1, 2, 3));
            model.graph.inputs.Add(Value("indices", TensorDataType.Int32, 1, 2, 2));
            var gatherElements = Node("gather_elements", "GatherElements", new[] { "data", "indices" }, new[] { "output" });
            gatherElements.attributes["axis"] = Int(2);
            model.graph.nodes.Add(gatherElements);

            var rejected = AexisOnnxGraphLowering.Lower(model);
            var accepted = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                verifiedInRangeIndexNodes = new[] { "gather_elements" }
            });

            Assert.That(rejected.IsEligible, Is.False);
            Assert.That(rejected.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-gather-profile" && diagnostic.blocking));
            Assert.That(accepted.IsEligible, Is.True);
            var layer = accepted.graph.layers.Single(candidate => candidate.typeName == "GatherElements");
            Assert.That(layer.GetInt(0), Is.EqualTo(1));
            Assert.That(layer.GetString("index_dtype"), Is.EqualTo("Int32"));
            Assert.That(layer.GetString("indices_in_range"), Is.EqualTo("1"));
        }

        [Test]
        public void OnnxLowering_RejectsIntermediateInt64GatherAndScatterIndices()
        {
            var gatherModel = new OnnxModel { opset = 18 };
            gatherModel.graph.inputs.Add(Value("data", TensorDataType.Float32, 4));
            gatherModel.graph.inputs.Add(Value("raw_indices", TensorDataType.Int32, 2, 1));
            var gatherCast = Node("gather_cast", "Cast", new[] { "raw_indices" }, new[] { "indices64" });
            gatherCast.attributes["to"] = Int(7);
            gatherModel.graph.nodes.Add(gatherCast);
            gatherModel.graph.nodes.Add(Node("gather", "GatherND", new[] { "data", "indices64" }, new[] { "output" }));

            var scatterModel = new OnnxModel { opset = 18 };
            scatterModel.graph.inputs.Add(Value("data", TensorDataType.Float32, 4));
            scatterModel.graph.inputs.Add(Value("raw_indices", TensorDataType.Int32, 2, 1));
            scatterModel.graph.inputs.Add(Value("updates", TensorDataType.Float32, 2));
            var scatterCast = Node("scatter_cast", "Cast", new[] { "raw_indices" }, new[] { "indices64" });
            scatterCast.attributes["to"] = Int(7);
            scatterModel.graph.nodes.Add(scatterCast);
            scatterModel.graph.nodes.Add(Node("scatter", "ScatterND", new[] { "data", "indices64", "updates" }, new[] { "output" }));

            var gather = AexisOnnxGraphLowering.Lower(gatherModel, new AexisOnnxGraphLoweringOptions
            {
                verifiedInRangeIndexNodes = new[] { "gather" }
            });
            var scatter = AexisOnnxGraphLowering.Lower(scatterModel, new AexisOnnxGraphLoweringOptions
            {
                verifiedInRangeIndexNodes = new[] { "scatter" },
                verifiedUniqueScatterNodes = new[] { "scatter" }
            });

            Assert.That(gather.IsEligible, Is.False);
            Assert.That(gather.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-gathernd-profile" && diagnostic.blocking));
            Assert.That(scatter.IsEligible, Is.False);
            Assert.That(scatter.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-scatter-profile" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_WhereAndTopKRequireExactGpuDtypes()
        {
            var validWhere = new OnnxModel { opset = 18 };
            var condition = Value("condition", TensorDataType.Int32, 2, 1);
            condition.onnxDataType = 9;
            var a = Value("a", TensorDataType.Float32, 2, 3);
            a.onnxDataType = 1;
            var b = Value("b", TensorDataType.Float32, 1, 3);
            b.onnxDataType = 1;
            validWhere.graph.inputs.Add(condition);
            validWhere.graph.inputs.Add(a);
            validWhere.graph.inputs.Add(b);
            validWhere.graph.nodes.Add(Node("where", "Where", new[] { "condition", "a", "b" }, new[] { "output" }));

            var invalidWhere = new OnnxModel { opset = 18 };
            invalidWhere.graph.inputs.Add(Value("condition", TensorDataType.Float32, 2, 1));
            invalidWhere.graph.inputs.Add(Value("a", TensorDataType.Float32, 2, 3));
            invalidWhere.graph.inputs.Add(Value("b", TensorDataType.Float32, 1, 3));
            invalidWhere.graph.nodes.Add(Node("where", "Where", new[] { "condition", "a", "b" }, new[] { "output" }));

            var topKModel = new OnnxModel { opset = 18 };
            var intData = Value("data", TensorDataType.Int32, 4);
            intData.onnxDataType = 6;
            topKModel.graph.inputs.Add(intData);
            topKModel.graph.initializers["k"] = IntTensor("k", new[] { 2 }, 1);
            topKModel.graph.nodes.Add(Node("topk", "TopK", new[] { "data", "k" }, new[] { "values", "indices" }));

            var accepted = AexisOnnxGraphLowering.Lower(validWhere);
            var rejectedWhere = AexisOnnxGraphLowering.Lower(invalidWhere);
            var rejectedTopK = AexisOnnxGraphLowering.Lower(topKModel);

            Assert.That(accepted.IsEligible, Is.True);
            Assert.That(Find(accepted, "output").onnxDataType, Is.EqualTo(1));
            Assert.That(rejectedWhere.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-where-profile" && diagnostic.blocking));
            Assert.That(rejectedTopK.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-topk-input-dtype" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_ReshapeResolvesZeroAndMinusOneBeforeRuntimeEncoding()
        {
            var validModel = new OnnxModel { opset = 18 };
            validModel.graph.inputs.Add(Value("data", TensorDataType.Float32, 1, 2, 3));
            validModel.graph.initializers["shape"] = IntTensor("shape", new[] { 0, -1 }, 2);
            validModel.graph.nodes.Add(Node("reshape", "Reshape", new[] { "data", "shape" }, new[] { "output" }));

            var invalidModel = new OnnxModel { opset = 18 };
            invalidModel.graph.inputs.Add(Value("data", TensorDataType.Float32, 1, 2, 3));
            invalidModel.graph.initializers["shape"] = IntTensor("shape", new[] { -1, -1 }, 2);
            invalidModel.graph.nodes.Add(Node("reshape", "Reshape", new[] { "data", "shape" }, new[] { "output" }));

            var valid = AexisOnnxGraphLowering.Lower(validModel);
            var invalid = AexisOnnxGraphLowering.Lower(invalidModel);

            Assert.That(valid.IsEligible, Is.True);
            Assert.That(Find(valid, "output").shape, Is.EqualTo(new long[] { 1, 6 }));
            var reshape = valid.graph.layers.Single(layer => layer.typeName == "Reshape");
            Assert.That(reshape.GetInt(0), Is.EqualTo(6));
            Assert.That(invalid.IsEligible, Is.False);
            Assert.That(invalid.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "invalid-reshape-shape" && diagnostic.blocking));
        }

        [Test]
        public void OnnxLowering_RejectsInt64InitializerOutsideExactTextureRange()
        {
            var model = new OnnxModel { opset = 18 };
            model.graph.initializers["indices"] = Int64Tensor("indices", new long[] { 16777217 }, 1);
            model.graph.nodes.Add(Node("identity", "Identity", new[] { "indices" }, new[] { "output" }));

            var result = AexisOnnxGraphLowering.Lower(model);

            Assert.That(result.IsEligible, Is.False);
            Assert.That(result.diagnostics, Has.Some.Matches<AexisOnnxLoweringDiagnostic>(diagnostic =>
                diagnostic.code == "unsupported-runtime-initializer" && diagnostic.blocking
                && diagnostic.message.Contains("FP32-exact Int32")));
        }

        [Test]
        public void AexisStaticTextureOperators_PassOnlyExactLoadedProfiles()
        {
            var fp32 = LinearTexture("data", 2, 3, 2, 1, 1);
            var intIndices = LinearTexture("indices", 1, 2, 1, 1, 1, "Int32");

            var shape = SentisLayer("Shape", new[] { "data" }, new[] { "output" });
            shape.intParams[0] = "0";
            shape.intParams[1] = "2";
            var shapeReport = AnalyzeLoadedLayer(shape, StrictTextureRequest(new[] { fp32 }));
            Assert.That(shapeReport.strictEligible, Is.True, DescribeReport(shapeReport));
            Assert.That(shapeReport.nodes[0].outputs[0].logicalDtype, Is.EqualTo("Int32"));

            var size = SentisLayer("Size", new[] { "data" }, new[] { "output" });
            Assert.That(AnalyzeLoadedLayer(size, StrictTextureRequest(new[] { fp32 })).strictEligible, Is.True);

            var range = SentisLayer("Range", Array.Empty<string>(), new[] { "output" });
            range.stringParams["start"] = "0";
            range.stringParams["limit"] = "4";
            range.stringParams["delta"] = "1";
            range.stringParams["logical_dtype"] = "Int32";
            var rangeReport = AnalyzeLoadedLayer(range, StrictTextureRequest(Array.Empty<AexisTexturePlanTensorDescriptor>()));
            Assert.That(rangeReport.strictEligible, Is.True, DescribeReport(rangeReport));

            var constant = SentisLayer("ConstantOfShape", Array.Empty<string>(), new[] { "output" });
            constant.stringParams["shape"] = "2,3";
            constant.stringParams["value"] = "7";
            constant.stringParams["logical_dtype"] = "Int32";
            Assert.That(AnalyzeLoadedLayer(constant, StrictTextureRequest(Array.Empty<AexisTexturePlanTensorDescriptor>())).strictEligible, Is.True);

            var expand = SentisLayer("Expand", new[] { "data" }, new[] { "output" });
            expand.stringParams["shape"] = "2,3";
            Assert.That(AnalyzeLoadedLayer(expand, StrictTextureRequest(new[] { fp32 })).strictEligible, Is.True);

            var where = SentisLayer("Where", new[] { "condition", "a", "b" }, new[] { "output" });
            var condition = LinearTexture("condition", 2, 1, 2, 1, 1, "Int32");
            var a = LinearTexture("a", 2, 3, 2, 1, 1);
            var b = LinearTexture("b", 1, 3, 1, 1, 1);
            Assert.That(AnalyzeLoadedLayer(where, StrictTextureRequest(new[] { condition, a, b })).strictEligible, Is.True);

            var gather = SentisLayer("Gather", new[] { "data", "indices" }, new[] { "output" });
            gather.intParams[0] = "1";
            gather.stringParams["indices_in_range"] = "1";
            gather.stringParams["index_dtype"] = "Int32";
            Assert.That(AnalyzeLoadedLayer(gather, StrictTextureRequest(new[] { fp32, intIndices })).strictEligible, Is.True);
            gather.stringParams.Remove("indices_in_range");
            Assert.That(AnalyzeLoadedLayer(gather, StrictTextureRequest(new[] { fp32, intIndices })).strictEligible, Is.False);

            var gatherElements = SentisLayer("GatherElements", new[] { "data", "indices" }, new[] { "output" });
            gatherElements.intParams[0] = "1";
            gatherElements.stringParams["indices_in_range"] = "1";
            gatherElements.stringParams["index_dtype"] = "Int32";
            var elementIndices = LinearTexture("indices", 2, 2, 2, 1, 1, "Int32");
            Assert.That(AnalyzeLoadedLayer(gatherElements, StrictTextureRequest(new[] { fp32, elementIndices })).strictEligible, Is.True);

            foreach (var op in new[] { "ArgMax", "ArgMin" })
            {
                var arg = SentisLayer(op, new[] { "data" }, new[] { "output" });
                arg.intParams[0] = "1";
                arg.intParams[1] = "0";
                arg.intParams[2] = "1";
                Assert.That(AnalyzeLoadedLayer(arg, StrictTextureRequest(new[] { fp32 })).strictEligible, Is.True, op);
            }

            var topK = SentisLayer("TopK", new[] { "data" }, new[] { "values", "indices" });
            topK.intParams[0] = "1";
            topK.intParams[1] = "2";
            topK.intParams[2] = "1";
            topK.intParams[3] = "1";
            topK.stringParams["k"] = "2";
            var topKReport = AnalyzeLoadedLayer(topK, StrictTextureRequest(new[] { fp32 }));
            Assert.That(topKReport.strictEligible, Is.True, DescribeReport(topKReport));
            Assert.That(topKReport.nodes[0].outputs[1].logicalDtype, Is.EqualTo("Int32"));

            var oneHot = SentisLayer("OneHot", new[] { "indices" }, new[] { "output" });
            oneHot.intParams[1] = "3";
            oneHot.intParams[2] = "1";
            oneHot.stringParams["logical_dtype"] = "Float32";
            Assert.That(AnalyzeLoadedLayer(oneHot, StrictTextureRequest(new[] { intIndices })).strictEligible, Is.True);
            Assert.That(AnalyzeLoadedLayer(oneHot, StrictTextureRequest(new[] { LinearTexture("indices", 1, 2, 1, 1, 1) })).strictEligible, Is.False);
        }

        [Test]
        public void NonMaxSuppression_CommandBufferUsesBoundedIndexAndGpuCountTextures()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat texture profile required by this NMS golden.");

            var layer = SentisLayer("Nms", new[] { "boxes", "scores" }, new[] { "indices", "count" });
            layer.intParams[0] = "4";
            layer.intParams[1] = "2";
            layer.intParams[2] = "0";
            layer.intParams[3] = "0.5";
            layer.intParams[4] = "0.2";
            layer.stringParams["capacity"] = "4";
            layer.stringParams["max_output_boxes_per_class"] = "2";
            layer.stringParams["center_point_box"] = "0";
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(SerializeSingleLayer(layer), reader);
            using var boxesUpload = new ComputeBuffer(12, sizeof(float));
            using var scoresUpload = new ComputeBuffer(6, sizeof(float));
            boxesUpload.SetData(new[]
            {
                0f, 0f, 2f, 2f,
                0.2f, 0.2f, 2.2f, 2.2f,
                3f, 3f, 5f, 5f
            });
            scoresUpload.SetData(new[] { 0.9f, 0.8f, 0.7f, 0.1f, 0.95f, 0.6f });
            using var commandBuffer = new CommandBuffer { name = "AexisBoundedNmsPack4Golden" };
            var boxes = session.RentTempMat(commandBuffer, 4, 3, RenderTextureFormat.RFloat);
            var scores = session.RentTempMat(commandBuffer, 3, 2, RenderTextureFormat.RFloat);
            var indicesReadback = CreateRFloatTarget(3, 4);
            var countReadback = CreateRFloatTarget(1, 1);
            try
            {
                ops.FillLinearMatFromBuffer(commandBuffer, boxesUpload, 4, 3, boxes);
                ops.FillLinearMatFromBuffer(commandBuffer, scoresUpload, 3, 2, scores);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["boxes"] = boxes, ["scores"] = scores },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["boxes"] = new AexisGraphSession.BufferShape(2, 4, 3, 1, 1),
                        ["scores"] = new AexisGraphSession.BufferShape(2, 3, 2, 1, 1)
                    },
                    new[] { "indices", "count" }))
                {
                    Assert.That(result.GetLogicalShape("indices"), Is.EqualTo(new AexisGraphSession.BufferShape(2, 3, 4, 1, 1)));
                    Assert.That(result.GetLogicalShape("count"), Is.EqualTo(new AexisGraphSession.BufferShape(1, 1, 1, 1, 1)));
                    commandBuffer.CopyTexture(new RenderTargetIdentifier(result.GetTexture("indices").nameID), new RenderTargetIdentifier(indicesReadback));
                    commandBuffer.CopyTexture(new RenderTargetIdentifier(result.GetTexture("count").nameID), new RenderTargetIdentifier(countReadback));
                }
                session.ReturnTempArray(commandBuffer, boxes);
                session.ReturnTempArray(commandBuffer, scores);
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                var indices = ReadRFloat(indicesReadback, 12);
                Assert.That(indices, Is.EqualTo(new[]
                {
                    0f, 0f, 0f,
                    0f, 0f, 2f,
                    0f, 1f, 1f,
                    0f, 1f, 2f
                }).Within(1e-6f), "actual=" + string.Join(",", indices));
                Assert.That(ReadRFloat(countReadback, 1), Is.EqualTo(new[] { 4f }).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(indicesReadback);
                UnityEngine.Object.DestroyImmediate(countReadback);
            }
        }

        [Test]
        public void BinaryOpScalar_Pack4LinearMatProfilePassesStrictPlanAndDispatchesOnGpu()
        {
            RequireArgbFloatCompute();
            var layer = new AexisGraphModel.Layer
            {
                name = "pack4_linear_scalar",
                typeName = "BinaryOp",
                type = AexisLayerTypes.BinaryOp,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "input" },
                topNames = new[] { "output" },
                intParams = new Dictionary<int, string> { [0] = "2", [1] = "1", [2] = "0.5" }
            };
            var report = AnalyzeLoadedLayer(layer, StrictTextureRequest(new[]
            {
                new AexisTexturePlanTensorDescriptor
                {
                    blob = "input",
                    logicalShape = new[] { 2, 8, 2, 1, 1 },
                    storageShape = new[] { 3, 2, 2, 1, 4 },
                    layout = AexisTexturePlanLayout.Packed4,
                    dtype = "FP32",
                    logicalDtype = "Float32",
                    aliasGroup = "external:input",
                    textureBacked = true
                }
            }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.nodes.Last().executionPath, Is.EqualTo("command-buffer-pack4:binary-pack4-linear-scalar"));

            using var ops = new AexisOps();
            using var commandBuffer = new CommandBuffer { name = "AexisBinaryPack4LinearScalarGolden" };
            var input = CreatePack4Array(new[]
            {
                2f, 4f, 6f, 8f,
                10f, 12f, 14f, 16f,
                18f, 20f, 22f, 24f,
                26f, 28f, 30f, 32f
            }, 2, 2, 1);
            var output = CreatePack4Target(2, 2, 1);
            try
            {
                // The golden validates the same Pack4 scalar kernel used by the command-buffer profile.
                ops.BinaryOpScalarPack4(commandBuffer, input, 0.5f, 1, 2, output);
                Graphics.ExecuteCommandBuffer(commandBuffer);
                Assert.That(ReadPack4Slice(output, 0), Is.EqualTo(new[]
                {
                    1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,
                    9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f
                }).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
                UnityEngine.Object.DestroyImmediate(output);
            }
        }

        [Test]
        public void SliceAndMaxUnPooling_LoadedProfilesAdmitNativeMultiOutputAndWideIndexTextures()
        {
            var slice = SentisLayer("Slice", new[] { "data" }, new[] { "left", "right" });
            slice.intParams[-23300] = "2,6";
            slice.intParams[1] = "0";
            var sliceReport = AnalyzeLoadedLayer(
                slice,
                StrictTextureRequest(new[] { Texture("data", 3, 4, 2, 1, 8) }));
            Assert.That(sliceReport.strictEligible, Is.True, DescribeReport(sliceReport));
            Assert.That(sliceReport.texturePlan.nodes[0].outputs.Select(output => output.logicalShape[4]),
                Is.EqualTo(new[] { 2, 6 }));

            var maxUnpool = SentisLayer("MaxUnPooling", new[] { "values", "indices" }, new[] { "output" });
            maxUnpool.intParams[1] = "2";
            maxUnpool.intParams[11] = "2";
            maxUnpool.intParams[2] = "2";
            maxUnpool.intParams[12] = "2";
            var indices = Texture("indices", 3, 16, 16, 1, 512);
            indices.sourceLogicalShape = new[] { 3, 32, 32, 1, 512 };
            var unpoolReport = AnalyzeLoadedLayer(
                maxUnpool,
                StrictTextureRequest(new[] { Texture("values", 3, 16, 16, 1, 256), indices }));
            Assert.That(unpoolReport.strictEligible, Is.True, DescribeReport(unpoolReport));
            Assert.That(unpoolReport.texturePlan.nodes[0].outputs[0].logicalShape,
                Is.EqualTo(new[] { 3, 32, 32, 1, 256 }));
        }

        [Test]
        public void ShuffleChannelAndFp16InnerProduct_AdmitRealCommandBufferTextureProfiles()
        {
            var shuffle = SentisLayer("ShuffleChannel", new[] { "data" }, new[] { "output" });
            shuffle.intParams[0] = "3";
            var shuffleReport = AnalyzeLoadedLayer(
                shuffle,
                StrictTextureRequest(new[] { Texture("data", 3, 4, 4, 1, 12) }));
            Assert.That(shuffleReport.strictEligible, Is.True, DescribeReport(shuffleReport));
            Assert.That(shuffleReport.texturePlan.nodes[0].executionPath,
                Is.EqualTo("command-buffer-pack4:shuffle-channel"));

            var innerProduct = SentisLayer("InnerProduct", new[] { "data" }, new[] { "output" });
            innerProduct.intParams[0] = "2";
            innerProduct.intParams[1] = "0";
            innerProduct.intParams[2] = "8";
            var fp16Input = LinearTexture("data", 1, 4, 1, 1, 1);
            fp16Input.dtype = "FP16";
            fp16Input.logicalDtype = "Float16";
            var fp16Request = StrictTextureRequest(new[] { fp16Input });
            fp16Request.targetDtype = "FP16";
            var innerReport = AnalyzeLoadedLayer(
                innerProduct,
                fp16Request,
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
            Assert.That(innerReport.strictEligible, Is.True, DescribeReport(innerReport));
            Assert.That(innerReport.texturePlan.nodes[0].executionPath,
                Is.EqualTo("command-buffer-pack4:inner-product-linear-mat"));
        }

        [Test]
        public void CodeFormerPack4LinearInnerProductAndQwenDescriptorInterp_AdmitNativeCommandBufferProfiles()
        {
            // CodeFormer encoder MatMul_911: [K=512,M=256] Pack4Linear input
            // followed by an immutable [N=1024,K=512] InnerProduct. This must
            // use the Pack4 source shader variant, not reinterpret K/4 as K or
            // materialize the activation into a buffer.
            var innerProduct = SentisLayer("InnerProduct", new[] { "data" }, new[] { "output" });
            innerProduct.name = "MatMul_911";
            innerProduct.intParams[0] = "1024";
            innerProduct.intParams[1] = "1";
            innerProduct.intParams[2] = "524288";
            var codeFormerInput = new AexisTexturePlanTensorDescriptor
            {
                blob = "data",
                logicalShape = new[] { 2, 512, 256, 1, 1 },
                storageShape = new[] { 3, 128, 256, 1, 4 },
                layout = AexisTexturePlanLayout.Packed4,
                dtype = "FP16",
                logicalDtype = "Float16",
                aliasGroup = "external:data",
                textureBacked = true
            };
            var codeFormerRequest = StrictTextureRequest(new[] { codeFormerInput });
            codeFormerRequest.targetDtype = "FP16";
            // This profile is only valid for an actual FP16 activation contract.
            // Do not let a default FP32 test session claim the half Pack4-linear
            // projection variant merely because the request says FP16.
            using var codeFormerOps = new AexisOps();
            using var codeFormerSession = new AexisGraphSession(codeFormerOps);
            using var codeFormerReader = new AexisFloatArrayWeightReader(new float[524288 + 1024]);
            codeFormerSession.ApplyModelManifest(new ModelManifest
            {
                modelId = "codeformer-pack4-linear-profile",
                precision = new ModelPrecisionContract
                {
                    activationDataType = TensorDataType.Float16,
                    weightDataType = TensorDataType.Float16,
                    sensitiveOutputDataType = TensorDataType.Float16,
                    requireStrictTexturePlan = true
                }
            });
            codeFormerSession.LoadModel(SerializeSingleLayer(innerProduct), codeFormerReader);
            var codeFormerReport = codeFormerSession.AnalyzeLoadedModelPreflight(codeFormerRequest);
            var codeFormerNode = codeFormerReport.texturePlan.nodes.Single(node => node.operatorName == "InnerProduct");
            Assert.That(codeFormerReport.strictEligible, Is.True, DescribeReport(codeFormerReport));
            Assert.That(codeFormerNode.executionPath,
                Is.EqualTo("command-buffer-pack4:inner-product-pack4-linear-mat"));
            Assert.That(codeFormerNode.outputs[0].logicalShape,
                Is.EqualTo(new[] { 2, 1024, 256, 1, 1 }));
            Assert.That(codeFormerNode.outputs[0].storageShape,
                Is.EqualTo(new[] { 3, 256, 256, 1, 4 }));

            // QWEN vision position embedding: size_expr=1w,1h only reads the
            // second texture's logical descriptor. At 48x48 it is a legal GPU
            // alias; it must not be classified as data-dependent resize.
            var interp = SentisLayer("Interp", new[] { "template", "grid" }, new[] { "position" });
            interp.name = "F.upsample_6";
            interp.intParams[0] = "2";
            interp.intParams[5] = "1";
            interp.intParams[6] = "1";
            interp.intParams[9] = "1w,1h";
            var qwenReport = AnalyzeLoadedLayerWithDeclaredInputs(
                interp,
                StrictTextureRequest(new[]
                {
                    Texture("template", 3, 48, 48, 1, 768),
                    LinearTexture("grid", 2, 48, 48, 1, 1)
                }));
            var qwenNode = qwenReport.texturePlan.nodes.Single(node => node.operatorName == "Interp");
            Assert.That(qwenReport.strictEligible, Is.True, DescribeReport(qwenReport));
            Assert.That(qwenNode.usesDescriptorAlias, Is.True);
            Assert.That(qwenNode.outputs[0].logicalShape,
                Is.EqualTo(new[] { 3, 48, 48, 1, 768 }));
        }

        [Test]
        public void BinaryOpPack4Broadcast_PreflightMatchesTextureKernelProfiles()
        {
            var binary = SentisLayer("BinaryOp", new[] { "image", "mask" }, new[] { "output" });
            binary.intParams[0] = "2";

            // DeepFillV2 prepare_image_masked: RGB * (1 - mask).  The runtime
            // shader replicates the single-channel mask lane directly on GPU.
            var deepFillReport = AnalyzeLoadedLayerWithDeclaredInputs(
                binary,
                StrictTextureRequest(new[]
                {
                    Texture("image", 3, 400, 512, 1, 3),
                    Texture("mask", 3, 400, 512, 1, 1)
                }));
            Assert.That(deepFillReport.strictEligible, Is.True, DescribeReport(deepFillReport));
            var deepFillNode = deepFillReport.texturePlan.nodes.Single(node => node.operatorName == "BinaryOp");
            Assert.That(deepFillNode.executionPath,
                Is.EqualTo("command-buffer-pack4:binary-spatial-broadcast"));
            Assert.That(deepFillNode.outputs[0].logicalShape,
                Is.EqualTo(new[] { 3, 400, 512, 1, 3 }));

            // The same kernel also supports width-1 row broadcasting (modes 5/6).
            var rowReport = AnalyzeLoadedLayerWithDeclaredInputs(
                binary,
                StrictTextureRequest(new[]
                {
                    Texture("image", 3, 1, 5, 1, 8),
                    Texture("mask", 3, 7, 5, 1, 8)
                }));
            Assert.That(rowReport.strictEligible, Is.True, DescribeReport(rowReport));
            var rowNode = rowReport.texturePlan.nodes.Single(node => node.operatorName == "BinaryOp");
            Assert.That(rowNode.outputs[0].logicalShape,
                Is.EqualTo(new[] { 3, 7, 5, 1, 8 }));

            // CodeFormer applies SFT scale/shift tensors as exact 2D LinearMat
            // BinaryOps. The runtime dispatches BinaryOpLinearMat directly, so
            // strict planning must accept the same descriptor contract.
            var codeFormerBinary = SentisLayer("BinaryOp", new[] { "scale", "shift" }, new[] { "output" });
            codeFormerBinary.intParams[0] = "0";
            var codeFormerReport = AnalyzeLoadedLayerWithDeclaredInputs(
                codeFormerBinary,
                StrictTextureRequest(new[]
                {
                    LinearTexture("scale", 2, 16, 8, 1, 1),
                    LinearTexture("shift", 2, 16, 8, 1, 1)
                }));
            Assert.That(codeFormerReport.strictEligible, Is.True, DescribeReport(codeFormerReport));
            Assert.That(codeFormerReport.texturePlan.nodes.Single(node => node.operatorName == "BinaryOp").executionPath,
                Is.EqualTo("command-buffer-linearmat:binary-exact"));

            // Legacy MHA returns a scalar FP32 matrix in a one-slice array,
            // while CodeFormer's projection residual is an RFloat LinearMat.
            // This must resolve to the dedicated texture kernel and retain the
            // LinearMat descriptor rather than materializing an activation buffer.
            var legacyAttentionOutput = new AexisTexturePlanTensorDescriptor
            {
                blob = "attention",
                logicalShape = new[] { 2, 16, 8, 1, 1 },
                storageShape = new[] { 3, 16, 8, 1, 1 },
                layout = AexisTexturePlanLayout.Packed4,
                dtype = "FP32",
                logicalDtype = "Float32",
                aliasGroup = "external:attention",
                textureBacked = true
            };
            var legacyResidualBinary = SentisLayer("BinaryOp", new[] { "projection", "attention" }, new[] { "output" });
            legacyResidualBinary.intParams[0] = "0";
            var legacyResidualReport = AnalyzeLoadedLayerWithDeclaredInputs(
                legacyResidualBinary,
                StrictTextureRequest(new[]
                {
                    LinearTexture("projection", 2, 16, 8, 1, 1),
                    legacyAttentionOutput
                }));
            var legacyResidualNode = legacyResidualReport.texturePlan.nodes.Single(node => node.operatorName == "BinaryOp");
            Assert.That(legacyResidualReport.strictEligible, Is.True, DescribeReport(legacyResidualReport));
            Assert.That(legacyResidualNode.executionPath,
                Is.EqualTo("command-buffer-linearmat:binary-scalar-array"));
            Assert.That(legacyResidualNode.outputs[0].storageShape,
                Is.EqualTo(new[] { 2, 16, 8, 1, 1 }));
        }

        // Direct Unity batch entry point used where the Unity Test Framework CLI
        // is not installed.  It keeps the production admission checks executable
        // without introducing a CPU fallback test harness.
        public static void RunSliceAndMaxUnPoolingBatchValidation()
        {
            var tests = new AexisP0ProductionBaselineTests();
            tests.SliceAndMaxUnPooling_LoadedProfilesAdmitNativeMultiOutputAndWideIndexTextures();
            tests.ShuffleChannelAndFp16InnerProduct_AdmitRealCommandBufferTextureProfiles();
            tests.CodeFormerPack4LinearInnerProductAndQwenDescriptorInterp_AdmitNativeCommandBufferProfiles();
            tests.BinaryOpPack4Broadcast_PreflightMatchesTextureKernelProfiles();
            Debug.Log("[AexisP0ProductionBaselineTests] Slice/MaxUnPooling/ShuffleChannel/InnerProduct/Qwen Interp/BinaryOp Pack4 profiles passed");
        }

        [Test]
        public void NcnnTextureNativeLayers_RequireExactLoadedStrictProfiles()
        {
            var pack4 = Texture("data", 3, 4, 4, 1, 4);
            var request = StrictTextureRequest(new[] { pack4 });

            var aten = SentisLayer("aten::to", new[] { "data" }, new[] { "output" });
            Assert.That(AnalyzeLoadedLayer(aten, request).strictEligible, Is.True);

            var crop = SentisLayer("Crop", new[] { "data" }, new[] { "output" });
            crop.intParams[0] = "1";
            crop.intParams[1] = "1";
            crop.intParams[2] = "1";
            crop.intParams[3] = "2";
            crop.intParams[4] = "2";
            crop.intParams[5] = "3";
            var cropReport = AnalyzeLoadedLayer(crop, request);
            Assert.That(cropReport.strictEligible, Is.True, DescribeReport(cropReport));
            Assert.That(cropReport.nodes[0].outputs[0].logicalShape, Is.EqualTo(new[] { 3, 2, 2, 1, 3 }));

            var padding = SentisLayer("Padding", new[] { "data" }, new[] { "output" });
            padding.intParams[0] = "1";
            padding.intParams[1] = "2";
            padding.intParams[2] = "1";
            padding.intParams[3] = "2";
            padding.intParams[4] = "0";
            var paddingReport = AnalyzeLoadedLayer(padding, request);
            Assert.That(paddingReport.strictEligible, Is.True, DescribeReport(paddingReport));
            Assert.That(paddingReport.nodes[0].outputs[0].logicalShape, Is.EqualTo(new[] { 3, 7, 7, 1, 4 }));
            padding.intParams[7] = "1";
            Assert.That(AnalyzeLoadedLayer(padding, request).strictEligible, Is.False);

            var reorg = SentisLayer("Reorg", new[] { "data" }, new[] { "output" });
            reorg.intParams[0] = "2";
            reorg.intParams[1] = "0";
            var reorgReport = AnalyzeLoadedLayer(reorg, request);
            Assert.That(reorgReport.strictEligible, Is.True, DescribeReport(reorgReport));
            Assert.That(reorgReport.nodes[0].outputs[0].logicalShape, Is.EqualTo(new[] { 3, 2, 2, 1, 16 }));
            reorg.intParams[0] = "1";
            Assert.That(AnalyzeLoadedLayer(reorg, request).strictEligible, Is.False);

            var unfold = SentisLayer("Unfold", new[] { "data" }, new[] { "output" });
            unfold.intParams[1] = "2";
            unfold.intParams[11] = "2";
            unfold.intParams[2] = "1";
            unfold.intParams[12] = "1";
            unfold.intParams[3] = "2";
            unfold.intParams[13] = "2";
            var unfoldReport = AnalyzeLoadedLayer(unfold, request);
            Assert.That(unfoldReport.strictEligible, Is.True, DescribeReport(unfoldReport));
            Assert.That(unfoldReport.nodes[0].outputs[0].logicalShape, Is.EqualTo(new[] { 2, 4, 16, 1, 1 }));

            var groupNorm = SentisLayer("GroupNorm", new[] { "data" }, new[] { "output" });
            groupNorm.intParams[0] = "2";
            groupNorm.intParams[1] = "4";
            groupNorm.intParams[2] = "0.00001";
            groupNorm.intParams[3] = "1";
            var groupReport = AnalyzeLoadedLayer(groupNorm, request, 1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f);
            Assert.That(groupReport.strictEligible, Is.True, DescribeReport(groupReport));
            Assert.That(groupReport.texturePlan.nodes[0].scratch, Has.Length.EqualTo(2));
            Assert.That(groupReport.texturePlan.nodes[0].scratch.All(scratch => scratch.storageShape.SequenceEqual(new[] { 3, 2, 1, 1, 4 })), Is.True);
            var fp16Input = Texture("data", 3, 4, 4, 1, 4);
            fp16Input.dtype = "FP16";
            fp16Input.logicalDtype = "Float16";
            var fp16Request = StrictTextureRequest(new[] { fp16Input });
            fp16Request.targetDtype = "FP16";
            var fp16GroupReport = AnalyzeLoadedLayer(groupNorm, fp16Request, 1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f);
            Assert.That(fp16GroupReport.strictEligible, Is.True, DescribeReport(fp16GroupReport));
            Assert.That(fp16GroupReport.texturePlan.nodes[0].scratch.All(scratch => scratch.dtype == "FP32"), Is.True);
            groupNorm.intParams[1] = "3";
            Assert.That(AnalyzeLoadedLayer(groupNorm, request, 1f, 1f, 1f, 0f, 0f, 0f).strictEligible, Is.False);

            var scale = SentisLayer("Scale", new[] { "data" }, new[] { "output" });
            scale.intParams[0] = "1";
            scale.intParams[1] = "0";
            var scaleReport = AnalyzeLoadedLayer(scale, request, 2f);
            Assert.That(scaleReport.strictEligible, Is.True, DescribeReport(scaleReport));
            scale.intParams[0] = "4";
            Assert.That(AnalyzeLoadedLayer(scale, request, 1f, 1f, 1f, 1f).strictEligible, Is.False);

            var quantize = SentisLayer("Quantize", new[] { "data" }, new[] { "output" });
            quantize.intParams[0] = "1";
            var quantReport = AnalyzeLoadedLayer(quantize, request, 2f);
            Assert.That(quantReport.strictEligible, Is.True, DescribeReport(quantReport));
            Assert.That(quantReport.nodes[0].outputs[0].logicalDtype, Is.EqualTo("Int8"));

            var int8Input = Texture("data", 3, 4, 4, 1, 4);
            int8Input.logicalDtype = "Int8";
            var int8Request = StrictTextureRequest(new[] { int8Input });
            var dequantize = SentisLayer("Dequantize", new[] { "data" }, new[] { "output" });
            dequantize.intParams[0] = "1";
            dequantize.intParams[1] = "0";
            Assert.That(AnalyzeLoadedLayer(dequantize, int8Request, 0.5f).strictEligible, Is.True);

            var requantize = SentisLayer("Requantize", new[] { "data" }, new[] { "output" });
            requantize.intParams[0] = "1";
            requantize.intParams[1] = "1";
            requantize.intParams[2] = "0";
            requantize.intParams[3] = "0";
            var requantReport = AnalyzeLoadedLayer(requantize, int8Request, 0.5f, 2f);
            Assert.That(requantReport.strictEligible, Is.True, DescribeReport(requantReport));
            Assert.That(requantReport.nodes[0].outputs[0].logicalDtype, Is.EqualTo("Int8"));
        }

        [Test]
        public void AexisGatherTopKAndOneHot_LinearMatGpuMatchesReference()
        {
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                Assert.Ignore("The graphics device does not support the RFloat compute profile required by this golden.");

            using var ops = new AexisOps();
            var dataShape = new AexisGraphSession.BufferShape(2, 3, 2, 1, 1);
            var dataStorage = new AexisGraphSession.BufferShape(2, 3, 2, 1, 1);
            var data = CreateLinearMat(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 3, 2);
            var gatherIndices = CreateLinearMat(new[] { -1f, 0f });
            var gatherOutput = CreateRFloatTarget(2, 2);
            var gatherElementsIndices = CreateLinearMat(new[] { -1f, 0f, 1f, -1f, 0f, 1f }, 3, 2);
            var gatherElementsOutput = CreateRFloatTarget(3, 2);
            var topKInput = CreateLinearMat(new[] { 1f, 3f, 3f, 2f });
            var topKValues = CreateRFloatTarget(2, 1);
            var topKIndices = CreateRFloatTarget(2, 1);
            var oneHotIndices = CreateLinearMat(new[] { -1f, 0f, 2f });
            var oneHotOutput = CreateRFloatTarget(3, 3);
            try
            {
                var gatherIndexShape = new AexisGraphSession.BufferShape(1, 2, 1, 1, 1);
                var gatherOutShape = new AexisGraphSession.BufferShape(2, 2, 2, 1, 1);
                var gatherOutStorage = new AexisGraphSession.BufferShape(2, 2, 2, 1, 1);
                ops.AexisGatherLinearMat(data, dataShape, dataStorage, gatherIndices, gatherIndexShape, new AexisGraphSession.BufferShape(2, 2, 1, 1, 1), 1, gatherOutShape, gatherOutStorage, gatherOutput);
                Assert.That(ReadRFloat(gatherOutput, 4), Is.EqualTo(new[] { 3f, 1f, 6f, 4f }).Within(1e-6f));

                var elementIndexShape = new AexisGraphSession.BufferShape(2, 3, 2, 1, 1);
                ops.AexisGatherElementsLinearMat(data, dataShape, dataStorage, gatherElementsIndices, elementIndexShape, dataStorage, 1, elementIndexShape, dataStorage, gatherElementsOutput);
                Assert.That(ReadRFloat(gatherElementsOutput, 6), Is.EqualTo(new[] { 3f, 1f, 2f, 6f, 4f, 5f }).Within(1e-6f));

                var topKInputShape = new AexisGraphSession.BufferShape(1, 4, 1, 1, 1);
                var topKOutputShape = new AexisGraphSession.BufferShape(1, 2, 1, 1, 1);
                var topKOutputStorage = new AexisGraphSession.BufferShape(2, 2, 1, 1, 1);
                ops.AexisTopKLinearMat(topKInput, topKInputShape, new AexisGraphSession.BufferShape(2, 4, 1, 1, 1), 0, 2, true, topKOutputShape, topKOutputStorage, topKValues, topKIndices);
                Assert.That(ReadRFloat(topKValues, 2), Is.EqualTo(new[] { 3f, 3f }).Within(1e-6f));
                Assert.That(ReadRFloat(topKIndices, 2), Is.EqualTo(new[] { 1f, 2f }).Within(1e-6f));

                var oneHotInputShape = new AexisGraphSession.BufferShape(1, 3, 1, 1, 1);
                var oneHotOutputShape = new AexisGraphSession.BufferShape(2, 3, 3, 1, 1);
                ops.AexisOneHotLinearMat(oneHotIndices, oneHotInputShape, new AexisGraphSession.BufferShape(2, 3, 1, 1, 1), 1, 3, 0f, 1f, oneHotOutputShape, new AexisGraphSession.BufferShape(2, 3, 3, 1, 1), oneHotOutput);
                Assert.That(ReadRFloat(oneHotOutput, 9), Is.EqualTo(new[] { 0f, 0f, 1f, 1f, 0f, 0f, 0f, 0f, 1f }).Within(1e-6f));
            }
            finally
            {
                foreach (var texture in new[] { data, gatherIndices, gatherOutput, gatherElementsIndices, gatherElementsOutput, topKInput, topKValues, topKIndices, oneHotIndices, oneHotOutput })
                    UnityEngine.Object.DestroyImmediate(texture);
            }
        }

        [Test]
        public void OnnxLowering_GatherNdUsesStandardNByOneIndicesShape()
        {
            var model = new OnnxModel();
            model.graph.inputs.Add(Value("data", TensorDataType.Float32, 8));
            model.graph.inputs.Add(Value("indices", TensorDataType.Int32, 3, 1));
            model.graph.nodes.Add(Node("gather", "GatherND", new[] { "data", "indices" }, new[] { "result" }));

            var result = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                verifiedInRangeIndexNodes = new[] { "gather" }
            });

            Assert.That(result.IsEligible, Is.True);
            Assert.That(Find(result, "result").shape, Is.EqualTo(new long[] { 3 }));
            Assert.That(result.graph.layers.Last().GetString("indices_in_range"), Is.EqualTo("1"));
        }

        [Test]
        public void OnnxRoiAlign_LowersStaticMultiRoiToThePack4TextureProfile()
        {
            var model = new OnnxModel { opset = 16 };
            model.graph.inputs.Add(Value("features", TensorDataType.Float32, 1, 3, 8, 8));
            model.graph.inputs.Add(Value("rois", TensorDataType.Float32, 2, 4));
            model.graph.initializers["batch_indices"] = IntTensor("batch_indices", new[] { 0, 0 }, 2);
            var node = Node("roi_align", "RoiAlign", new[] { "features", "rois", "batch_indices" }, new[] { "pooled" });
            node.attributes["mode"] = StringAttribute("avg");
            node.attributes["coordinate_transformation_mode"] = StringAttribute("half_pixel");
            node.attributes["output_height"] = Int(2);
            node.attributes["output_width"] = Int(3);
            node.attributes["sampling_ratio"] = Int(2);
            node.attributes["spatial_scale"] = new OnnxAttribute { f = 0.5f };
            model.graph.nodes.Add(node);
            model.graph.outputs.Add(Value("pooled", TensorDataType.Float32, 2, 3, 2, 3));

            var result = AexisOnnxGraphLowering.Lower(model);
            var layer = result.graph.layers.Last();

            Assert.That(result.IsEligible, Is.True, DescribeLowering(result));
            Assert.That(layer.typeName, Is.EqualTo("ROIAlign"));
            Assert.That(layer.bottomNames, Is.EqualTo(new[] { "features", "rois" }));
            Assert.That(layer.GetInt(0), Is.EqualTo(3));
            Assert.That(layer.GetInt(1), Is.EqualTo(2));
            Assert.That(layer.GetInt(3), Is.EqualTo(2));
            Assert.That(layer.GetInt(4), Is.EqualTo(1));
            Assert.That(Find(result, "pooled").runtimeShape, Is.EqualTo(new long[] { 2, 3, 2, 3 }));
        }

        [Test]
        public void RoiAlign_StaticMultiRoiProfilePassesTheLoadedPack4PlanVerifier()
        {
            var layer = new AexisGraphModel.Layer
            {
                name = "roi_align",
                typeName = "ROIAlign",
                type = AexisLayerTypes.ROIAlign,
                bottoms = 2,
                tops = 1,
                bottomNames = new[] { "features", "rois" },
                topNames = new[] { "pooled" },
                intParams = new Dictionary<int, string>
                {
                    [0] = "3", [1] = "2", [2] = "1", [3] = "2", [4] = "1", [5] = "1"
                }
            };

            var report = AnalyzeLoadedLayerWithDeclaredInputs(layer, StrictTextureRequest(new[]
            {
                Texture("features", 3, 8, 8, 1, 3),
                LinearTexture("rois", 2, 4, 2, 1, 1)
            }));
            Assert.That(report.strictEligible, Is.True, DescribeReport(report));
            Assert.That(report.texturePlan.nodes.Last().outputs[0].logicalShape, Is.EqualTo(new[] { 4, 3, 2, 2, 3 }));
            Assert.That(report.texturePlan.nodes.Last().executionPath, Is.EqualTo("command-buffer-pack4:p1-roialign"));
            Assert.That(AexisOperatorCapabilities.TryGet("ROIAlign", out var capability), Is.True);
            Assert.That(capability.verifiedModels, Does.Contain("command-buffer-pack4-roialign-multi-roi-golden"));
        }

        [Test]
        public void TextureExecutionPlan_ReportsPack4LivenessAndStaticRtArena()
        {
            var model = new AexisGraphModel();
            model.layers.Add(SentisLayer("ReLU", new[] { "input" }, new[] { "middle" }));
            model.layers.Add(SentisLayer("ReLU", new[] { "middle" }, new[] { "output" }));
            var plan = AexisTextureExecutionPlanner.Analyze(model, new AexisTextureExecutionPlanRequest
            {
                strict = false,
                debugOracleRelaxed = true,
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                targetLayout = AexisTexturePlanLayout.Packed4,
                inputs = new[] { Texture("input", 3, 4, 4, 1, 3) }
            });

            Assert.That(plan.dispatchAllowed, Is.True);
            Assert.That(plan.memory, Is.Not.Null);
            Assert.That(plan.memory.resources, Has.Length.EqualTo(3));
            Assert.That(plan.memory.peakLiveBytes, Is.GreaterThan(0));
            Assert.That(plan.memory.temporaryArenaBytes, Is.GreaterThan(0));
            var middle = plan.memory.resources.Single(resource => resource.representativeBlob == "middle");
            Assert.That(middle.temporary, Is.True);
            Assert.That(middle.allocationSlot, Is.GreaterThanOrEqualTo(0));
        }

        [Test]
        public void TextureExecutionPlan_TerminalDescriptorAliasRetainsItsSingleProducedRt()
        {
            var model = new AexisGraphModel();
            model.layers.Add(SentisLayer("ReLU", new[] { "input" }, new[] { "middle" }));
            // Identity Permute preserves the exact Pack4 logical-to-physical
            // mapping, so this is a legitimate descriptor alias unlike Flatten.
            model.layers.Add(SentisLayer("Permute", new[] { "middle" }, new[] { "output" }));
            var plan = AexisTextureExecutionPlanner.Analyze(model, new AexisTextureExecutionPlanRequest
            {
                strict = false,
                debugOracleRelaxed = true,
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                targetLayout = AexisTexturePlanLayout.Packed4,
                inputs = new[] { Texture("input", 3, 4, 4, 1, 3) }
            });

            Assert.That(plan.dispatchAllowed, Is.True);
            Assert.That(plan.nodes.Last().usesDescriptorAlias, Is.True);
            Assert.That(plan.memory.resources.Count(resource => resource.producedByGraph), Is.EqualTo(1));
            var outputStorage = plan.memory.resources.Single(resource => resource.producedByGraph);
            Assert.That(outputStorage.representativeBlob, Is.EqualTo("middle"));
            Assert.That(outputStorage.persistent, Is.True);
            Assert.That(outputStorage.temporary, Is.False);
        }

        [Test]
        public void TextureExecutionPlan_ExplicitInputNodeRemainsCallerOwned()
        {
            var model = new AexisGraphModel();
            model.layers.Add(SentisLayer("Input", Array.Empty<string>(), new[] { "data" }));
            model.layers.Add(SentisLayer("ReLU", new[] { "data" }, new[] { "output" }));
            var plan = AexisTextureExecutionPlanner.Analyze(model, new AexisTextureExecutionPlanRequest
            {
                strict = false,
                debugOracleRelaxed = true,
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                targetLayout = AexisTexturePlanLayout.Packed4,
                inputs = new[] { Texture("data", 3, 4, 4, 1, 3) }
            });

            Assert.That(plan.dispatchAllowed, Is.True);
            Assert.That(plan.nodes[0].usesDescriptorAlias, Is.True);
            var inputStorage = plan.memory.resources.Single(resource => resource.representativeBlob == "data");
            Assert.That(inputStorage.producedByGraph, Is.False);
            Assert.That(inputStorage.externalInput, Is.True);
            Assert.That(inputStorage.allocationSlot, Is.EqualTo(-1));
            Assert.That(plan.memory.resources.Count(resource => resource.producedByGraph), Is.EqualTo(1));
        }

        [Test]
        public void CommandBufferPack4_StaticRtArenaBindsAndReleasesCompiledIntermediateSlot()
        {
            RequireArgbFloatCompute();
            const string model = "7767517\n"
                + "2 3\n"
                + "ReLU relu_0 1 1 data middle\n"
                + "ReLU relu_1 1 1 middle output\n";
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(model, reader);
            using var commandBuffer = new CommandBuffer { name = "AexisStaticRtArenaGolden" };
            var inputUpload = CreatePack4Array(new[]
            {
                -2f, 1f, -3f, 4f,
                5f, 0f, 0f, 0f,
                5f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f
            }, 2, 1, 2);
            var outputReadback = CreatePack4Target(2, 1, 2);
            ComputeTexture input = null;
            try
            {
                input = session.RentTempArray(commandBuffer, 2, 1, 2, RenderTextureFormat.ARGBFloat);
                commandBuffer.CopyTexture(inputUpload, 0, 0, input.nameID, 0, 0);
                commandBuffer.CopyTexture(inputUpload, 1, 0, input.nameID, 1, 0);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(3, 2, 1, 1, 5)
                    },
                    new[] { "output" }))
                {
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, 0, 0, outputReadback, 0, 0);
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, 1, 0, outputReadback, 1, 0);
                }
                session.ReturnTempArray(commandBuffer, input); input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastCommandBufferRtArena, Is.Not.Null);
                Assert.That(session.LastCommandBufferRtArena.plannedTemporaryResources, Is.EqualTo(1));
                Assert.That(session.LastCommandBufferRtArena.plannedPersistentResources, Is.EqualTo(1));
                Assert.That(session.LastCommandBufferRtArena.plannedSlots, Is.EqualTo(2));
                Assert.That(session.LastCommandBufferRtArena.boundResources, Is.EqualTo(2));
                Assert.That(session.LastCommandBufferRtArena.releasedResources, Is.EqualTo(1));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.Zero);
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                Assert.That(ReadPack4Slice(outputReadback, 0), Is.EqualTo(new[]
                {
                    0f, 1f, 0f, 4f, 5f, 0f, 0f, 0f
                }).Within(1e-6f));
                Assert.That(ReadPack4Slice(outputReadback, 1), Is.EqualTo(new[]
                {
                    5f, 0f, 0f, 0f, 0f, 0f, 0f, 0f
                }).Within(1e-6f));
            }
            finally
            {
                if (input != null) session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void CommandBufferPack4_MaxPoolingIndBindsSameDescriptorValueAndIndexByBlobIdentity()
        {
            RequireArgbFloatCompute();
            const string model = "7767517\n"
                + "3 4\n"
                + "MaxPoolingInd max_pool 1 2 data values indices 1=2 2=2 3=0 5=0\n"
                + "ReLU relu_0 1 1 values middle\n"
                + "ReLU relu_1 1 1 middle output\n";
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(model, reader);
            using var commandBuffer = new CommandBuffer { name = "AexisMaxPoolingIdentityRtArenaGolden" };
            var inputValues = new float[4 * 4 * 4];
            for (var pixel = 0; pixel < 16; pixel++)
            {
                for (var lane = 0; lane < 4; lane++)
                    inputValues[pixel * 4 + lane] = pixel + 1;
            }
            var inputUpload = CreatePack4Array(inputValues, 4, 4, 1);
            var outputReadback = CreatePack4Target(2, 2, 1);
            ComputeTexture input = null;
            try
            {
                input = session.RentTempArray(commandBuffer, 4, 4, 1, RenderTextureFormat.ARGBFloat);
                commandBuffer.CopyTexture(inputUpload, 0, 0, input.nameID, 0, 0);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(3, 4, 4, 1, 4)
                    },
                    new[] { "output" }))
                {
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, 0, 0, outputReadback, 0, 0);
                }
                session.ReturnTempArray(commandBuffer, input); input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                var resources = session.LastTextureExecutionPlan.memory.resources;
                var values = resources.Single(resource => resource.representativeBlob == "values");
                var indices = resources.Single(resource => resource.representativeBlob == "indices");
                Assert.That(values.allocationSlot, Is.Not.EqualTo(indices.allocationSlot),
                    "Same-descriptor MaxPoolingInd outputs must be bound by their declared graph identities.");
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.Zero);
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                Assert.That(ReadPack4Slice(outputReadback, 0), Is.EqualTo(new[]
                {
                    6f, 6f, 6f, 6f,
                    8f, 8f, 8f, 8f,
                    14f, 14f, 14f, 14f,
                    16f, 16f, 16f, 16f
                }).Within(1e-6f));
            }
            finally
            {
                if (input != null) session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void CommandBufferPack4_MvnDeclaresAndExecutesItsScratchRtArena()
        {
            RequireArgbFloatCompute();
            const string model = "7767517\n"
                + "1 2\n"
                + "MVN mvn_0 1 1 data output 0=1 1=0 2=0.0001\n";
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(Array.Empty<float>());
            session.LoadModel(model, reader);
            using var commandBuffer = new CommandBuffer { name = "AexisMvnScratchRtArenaGolden" };
            var inputUpload = CreatePack4Array(new[]
            {
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
            }, 2, 1, 1);
            var outputReadback = CreatePack4Target(2, 1, 1);
            ComputeTexture input = null;
            try
            {
                input = session.RentTempArray(commandBuffer, 2, 1, 1, RenderTextureFormat.ARGBFloat);
                commandBuffer.CopyTexture(inputUpload, 0, 0, input.nameID, 0, 0);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(3, 2, 1, 1, 4)
                    },
                    new[] { "output" }))
                {
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, 0, 0, outputReadback, 0, 0);
                }
                session.ReturnTempArray(commandBuffer, input); input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                var node = session.LastTextureExecutionPlan.nodes.Single(candidate => candidate.layer == "mvn_0");
                Assert.That(node.executionPath, Is.EqualTo("command-buffer-pack4:mvn-per-channel"));
                Assert.That(node.scratch, Has.Length.EqualTo(2));
                Assert.That(node.scratch.All(scratch => scratch.aliasGroup.StartsWith("scratch:mvn_0:", StringComparison.Ordinal)), Is.True);
                Assert.That(session.LastTextureExecutionPlan.memory.resources.Count(resource => resource.scratch), Is.EqualTo(2));
                Assert.That(session.LastCommandBufferRtArena.plannedTemporaryResources, Is.EqualTo(2));
                Assert.That(session.LastCommandBufferRtArena.plannedPersistentResources, Is.EqualTo(1));
                Assert.That(session.LastCommandBufferRtArena.plannedSlots, Is.EqualTo(3));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.Zero);
                Assert.That(ReadPack4Slice(outputReadback, 0).All(value => !float.IsNaN(value) && !float.IsInfinity(value)), Is.True);
            }
            finally
            {
                if (input != null) session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void CommandBufferPack4_GroupNormBindsFp32ScratchByPlannedIdentity()
        {
            RequireArgbFloatCompute();
            const string model = "7767517\n"
                + "1 2\n"
                + "GroupNorm groupnorm_0 1 1 data output 0=2 1=4 2=0.0001 3=1\n";
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32",
                UseNcnnStyleGroupNorm = true
            };
            using var reader = new AexisFloatArrayWeightReader(new[]
            {
                1f, 1f, 1f, 1f,
                0f, 0f, 0f, 0f
            });
            session.LoadModel(model, reader);
            using var commandBuffer = new CommandBuffer { name = "AexisGroupNormScratchRtArenaGolden" };
            var inputUpload = CreatePack4Array(new[]
            {
                1f, 2f, 5f, 6f,
                3f, 4f, 7f, 8f
            }, 2, 1, 1);
            var outputReadback = CreatePack4Target(2, 1, 1);
            ComputeTexture input = null;
            try
            {
                input = session.RentTempArray(commandBuffer, 2, 1, 1, RenderTextureFormat.ARGBFloat);
                commandBuffer.CopyTexture(inputUpload, 0, 0, input.nameID, 0, 0);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(3, 2, 1, 1, 4)
                    },
                    new[] { "output" }))
                {
                    commandBuffer.CopyTexture(result.GetTexture("output").nameID, 0, 0, outputReadback, 0, 0);
                }
                session.ReturnTempArray(commandBuffer, input); input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                var node = session.LastTextureExecutionPlan.nodes.Single(candidate => candidate.layer == "groupnorm_0");
                Assert.That(node.executionPath, Is.EqualTo("command-buffer-pack4:groupnorm"));
                Assert.That(node.scratch.Select(scratch => scratch.blob), Is.EquivalentTo(new[]
                {
                    "scratch:groupnorm_0:groupnorm-stats-a",
                    "scratch:groupnorm_0:groupnorm-stats-b"
                }));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.Zero);
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                Assert.That(ReadPack4Slice(outputReadback, 0), Is.EqualTo(new[]
                {
                    -1.341587f, -0.447196f, -1.341587f, -0.447196f,
                    0.447196f, 1.341587f, 0.447196f, 1.341587f
                }).Within(3e-4f));
            }
            finally
            {
                if (input != null) session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(inputUpload);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        [Test]
        public void CommandBufferPack4_RoiAlignProcessesEachStaticRoiDepthSlice()
        {
            RequireArgbFloatCompute();
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops) { TensorTextureFormat = RenderTextureFormat.ARGBFloat };
            using var commandBuffer = new CommandBuffer { name = "AexisRoiAlignMultiRoiPack4Golden" };
            var featureUpload = CreatePack4Array(new[]
            {
                1f, 0f, 0f, 0f, 2f, 0f, 0f, 0f,
                3f, 0f, 0f, 0f, 4f, 0f, 0f, 0f
            }, 2, 2, 1);
            var roiValues = new float[4 * 2 * 4];
            // half_pixel coordinates.  The shader removes 0.5 before sampling.
            roiValues[0] = 0.5f; roiValues[4] = 0.5f; roiValues[8] = 2.5f; roiValues[12] = 2.5f;
            roiValues[16] = 0.5f; roiValues[20] = 0.5f; roiValues[24] = 1.5f; roiValues[28] = 1.5f;
            var roiUpload = CreatePack4Array(roiValues, 4, 2, 1);
            var persistent = CreatePack4Target(2, 2, 2);
            ComputeTexture feature = null;
            ComputeTexture rois = null;
            ComputeTexture output = null;
            try
            {
                feature = session.RentTempArray(commandBuffer, 2, 2, 1, RenderTextureFormat.ARGBFloat);
                rois = session.RentTempArray(commandBuffer, 4, 2, 1, RenderTextureFormat.ARGBFloat);
                output = session.RentTempArray(commandBuffer, 2, 2, 2, RenderTextureFormat.ARGBFloat);
                commandBuffer.CopyTexture(featureUpload, 0, 0, feature.nameID, 0, 0);
                commandBuffer.CopyTexture(roiUpload, 0, 0, rois.nameID, 0, 0);
                ops.P1VisionPack4(commandBuffer, new AexisP1VisionDispatch
                {
                    kernel = AexisP1VisionKernel.RoiAlign,
                    input0 = new AexisGraphSession.BufferShape(3, 2, 2, 1, 1),
                    input1 = new AexisGraphSession.BufferShape(2, 4, 2, 1, 1),
                    input2 = new AexisGraphSession.BufferShape(3, 1, 1, 1, 0),
                    output = new AexisGraphSession.BufferShape(4, 2, 2, 2, 1),
                    pooledW = 2,
                    pooledH = 2,
                    samplingRatio = 1,
                    aligned = 1,
                    spatialScale = 1f
                }, feature, rois, null, output);
                commandBuffer.CopyTexture(output.nameID, 0, 0, persistent, 0, 0);
                commandBuffer.CopyTexture(output.nameID, 1, 0, persistent, 1, 0);
                session.ReturnTempArray(commandBuffer, output); output = null;
                session.ReturnTempArray(commandBuffer, rois); rois = null;
                session.ReturnTempArray(commandBuffer, feature); feature = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(ReadPack4Slice(persistent, 0).Where((_, index) => index % 4 == 0).Take(4).ToArray(),
                    Is.EqualTo(new[] { 2.5f, 3f, 3.5f, 4f }).Within(1e-5f));
                Assert.That(ReadPack4Slice(persistent, 1).Where((_, index) => index % 4 == 0).Take(4).ToArray(),
                    Is.EqualTo(new[] { 1.75f, 2.25f, 2.75f, 3.25f }).Within(1e-5f));
            }
            finally
            {
                if (output != null) session.ReturnTempArray(commandBuffer, output);
                if (rois != null) session.ReturnTempArray(commandBuffer, rois);
                if (feature != null) session.ReturnTempArray(commandBuffer, feature);
                UnityEngine.Object.DestroyImmediate(featureUpload);
                UnityEngine.Object.DestroyImmediate(roiUpload);
                UnityEngine.Object.DestroyImmediate(persistent);
            }
        }

        private static AexisModelPreflightRequest StrictTextureRequest(AexisTexturePlanTensorDescriptor[] textureInputs)
        {
            var inputs = new AexisPreflightTensorDescriptor[textureInputs.Length];
            for (var i = 0; i < textureInputs.Length; i++)
            {
                inputs[i] = new AexisPreflightTensorDescriptor
                {
                    blob = textureInputs[i].blob,
                    logicalShape = textureInputs[i].logicalShape,
                    storageShape = textureInputs[i].storageShape,
                    layout = "Packed4",
                    dtype = textureInputs[i].dtype,
                    logicalDtype = textureInputs[i].logicalDtype
                };
            }
            return new AexisModelPreflightRequest
            {
                strict = true,
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                targetLayout = "Packed4",
                inputs = inputs,
                textureInputs = textureInputs
            };
        }

        private static AexisTexturePlanTensorDescriptor Texture(string blob, params int[] shape)
        {
            return new AexisTexturePlanTensorDescriptor
            {
                blob = blob,
                logicalShape = shape,
                storageShape = shape,
                layout = AexisTexturePlanLayout.Packed4,
                dtype = "FP32",
                logicalDtype = "Float32",
                aliasGroup = "external:" + blob,
                textureBacked = true
            };
        }

        private static AexisTexturePlanTensorDescriptor LinearTexture(
            string blob,
            int dims,
            int w,
            int h,
            int d,
            int c,
            string logicalDtype = "Float32")
        {
            return new AexisTexturePlanTensorDescriptor
            {
                blob = blob,
                logicalShape = new[] { dims, w, h, d, c },
                storageShape = new[] { 2, w, dims >= 2 ? h : 1, 1, 1 },
                layout = AexisTexturePlanLayout.Packed4,
                dtype = "FP32",
                logicalDtype = logicalDtype,
                aliasGroup = "external:" + blob,
                textureBacked = true
            };
        }

        private static AexisGraphModel.Layer SentisLayer(string typeName, string[] bottoms, string[] tops)
        {
            return new AexisGraphModel.Layer
            {
                name = typeName.ToLowerInvariant(),
                typeName = typeName,
                type = AexisLayerTypeKey.FromString(typeName),
                bottoms = bottoms.Length,
                tops = tops.Length,
                bottomNames = bottoms,
                topNames = tops
            };
        }

        private static AexisGraphModel.Layer GatherNdLayer()
        {
            return new AexisGraphModel.Layer
            {
                name = "gather",
                typeName = "GatherND",
                type = AexisLayerTypes.GatherND,
                bottoms = 2,
                tops = 1,
                bottomNames = new[] { "data", "indices" },
                topNames = new[] { "result" },
                intParams = { [0] = "0", [1] = "1" },
                stringParams = { ["batch_dims"] = "0", ["index_depth"] = "1", ["index_dtype"] = "Int32", ["indices_in_range"] = "1" }
            };
        }

        private static AexisGraphModel.Layer ScatterNdLayer()
        {
            return new AexisGraphModel.Layer
            {
                name = "scatter",
                typeName = "ScatterND",
                type = AexisLayerTypes.ScatterND,
                bottoms = 3,
                tops = 1,
                bottomNames = new[] { "data", "indices", "updates" },
                topNames = new[] { "result" },
                intParams = { [-1] = "1", [1] = "1" },
                stringParams = { ["unique_indices"] = "1", ["indices_in_range"] = "1", ["index_dtype"] = "Int32", ["reduction"] = "none", ["index_depth"] = "1" }
            };
        }

        private static AexisGraphModel.Layer SoftmaxLayer(int mode)
        {
            return new AexisGraphModel.Layer
            {
                name = "softmax",
                typeName = "Softmax",
                type = AexisLayerTypes.Softmax,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams = { [0] = "1", [10] = mode.ToString(CultureInfo.InvariantCulture) }
            };
        }

        private static RenderTexture CreateLinearMat(float[] values)
        {
            return CreateLinearMat(values, values.Length, 1);
        }

        private static RenderTexture CreateLinearMat(float[] values, int width, int height)
        {
            Assert.That(values, Has.Length.EqualTo(width * height));
            var source = new Texture2D(width, height, TextureFormat.RFloat, false, true);
            try
            {
                source.SetPixelData(values, 0);
                source.Apply(false, false);
                var target = CreateRFloatTarget(width, height);
                Graphics.Blit(source, target);
                return target;
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(source);
            }
        }

        private static RenderTexture CreateRFloatTarget(int width, int height)
        {
            var texture = new RenderTexture(new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0)
            {
                enableRandomWrite = true,
                sRGB = false
            });
            Assert.That(texture.Create(), Is.True);
            return texture;
        }

        private static float[] ReadRFloat(RenderTexture source, int count)
        {
            var previous = RenderTexture.active;
            var readback = new Texture2D(source.width, source.height, TextureFormat.RFloat, false, true);
            try
            {
                RenderTexture.active = source;
                readback.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0, false);
                readback.Apply(false, false);
                return readback.GetPixelData<float>(0).ToArray().Take(count).ToArray();
            }
            finally
            {
                RenderTexture.active = previous;
                UnityEngine.Object.DestroyImmediate(readback);
            }
        }

        private static void RequireArgbFloatCompute()
        {
            if (!SystemInfo.supportsComputeShaders
                || !SystemInfo.supports2DArrayTextures
                || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.ARGBFloat)
                || SystemInfo.copyTextureSupport == CopyTextureSupport.None)
            {
                Assert.Ignore("The graphics device does not support the ARGBFloat texture-array compute profile required by this golden.");
            }
        }

        private static RenderTexture CreatePack4Array(float[] values, int width, int height, int depth)
        {
            Assert.That(values, Has.Length.EqualTo(width * height * depth * 4));
            var upload = new Texture2DArray(width, height, depth, TextureFormat.RGBAFloat, false, true);
            var target = CreatePack4Target(width, height, depth);
            try
            {
                var sliceLength = width * height * 4;
                for (var slice = 0; slice < depth; slice++)
                    upload.SetPixelData(values, 0, slice, slice * sliceLength);
                upload.Apply(false, false);
                for (var slice = 0; slice < depth; slice++)
                    Graphics.CopyTexture(upload, slice, 0, target, slice, 0);
                return target;
            }
            catch
            {
                UnityEngine.Object.DestroyImmediate(target);
                throw;
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(upload);
            }
        }

        private static RenderTexture CreatePack4Target(int width, int height, int depth)
        {
            var texture = new RenderTexture(new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGBFloat, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = depth,
                enableRandomWrite = true,
                sRGB = false
            });
            Assert.That(texture.Create(), Is.True);
            return texture;
        }

        private static float[] ReadPack4Slice(RenderTexture source, int slice)
        {
            Assert.That(source.dimension, Is.EqualTo(TextureDimension.Tex2DArray));
            Assert.That(slice, Is.InRange(0, source.volumeDepth - 1));
            var staging = new RenderTexture(new RenderTextureDescriptor(source.width, source.height, RenderTextureFormat.ARGBFloat, 0)
            {
                sRGB = false
            });
            Assert.That(staging.Create(), Is.True);
            var previous = RenderTexture.active;
            var readback = new Texture2D(source.width, source.height, TextureFormat.RGBAFloat, false, true);
            try
            {
                Graphics.CopyTexture(source, slice, 0, staging, 0, 0);
                RenderTexture.active = staging;
                readback.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0, false);
                readback.Apply(false, false);
                return readback.GetPixelData<float>(0).ToArray();
            }
            finally
            {
                RenderTexture.active = previous;
                UnityEngine.Object.DestroyImmediate(readback);
                UnityEngine.Object.DestroyImmediate(staging);
            }
        }

        private static void ExecuteLoadedLinearLayerAndAssert(
            AexisGraphModel.Layer layer,
            float[] immutableWeights,
            RenderTexture input,
            AexisGraphSession.BufferShape inputShape,
            float[] expected,
            bool enableGeneralConvolution = false)
        {
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                EnableDepthWiseTextureConvolution = true,
                EnableGeneralTextureConvolution = enableGeneralConvolution
            };
            using var reader = new AexisFloatArrayWeightReader(immutableWeights);
            var implementation = AexisLayerFactory.Create(layer);
            implementation.LoadLayer(session, layer, reader);
            var context = new AexisLayerBufferContext
            {
                textureBlobs = new Dictionary<string, AexisGraphSession.TensorRef>(StringComparer.Ordinal)
                {
                    ["input"] = new AexisGraphSession.TensorRef
                    {
                        texture = input,
                        width = input.width,
                        height = input.height,
                        packs = 1,
                        refs = 1,
                        owned = false,
                        hasLogicalShape = true,
                        logicalShape = inputShape,
                        hasStorageShape = true,
                        storageShape = inputShape,
                        layoutKind = AexisTextureTensorLayoutKind.LinearMat
                    }
                },
                textureShapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal) { ["input"] = inputShape },
                bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal),
                bufferRefs = new Dictionary<string, AexisGraphSession.BufferRef>(StringComparer.Ordinal),
                bufferViews = new Dictionary<string, AexisTensorBuffer>(StringComparer.Ordinal),
                indexBlobs = new Dictionary<string, AexisGraphSession.IndexRef>(StringComparer.Ordinal),
                remaining = new Dictionary<string, int>(StringComparer.Ordinal) { ["input"] = 1 },
                tempOwned = new List<IDisposable>()
            };

            try
            {
                implementation.ExecuteRenderTexturePath(session, layer, context);
                Assert.That(context.textureBlobs.TryGetValue("output", out var output), Is.True);
                Assert.That(output.logicalShape.dims, Is.EqualTo(2));
                Assert.That(output.logicalShape.h, Is.EqualTo(inputShape.h));
                Assert.That(output.texture.dimension, Is.EqualTo(TextureDimension.Tex2D));
                Assert.That(ReadRFloat(output.texture, expected.Length), Is.EqualTo(expected).Within(4e-5f), layer.typeName);
                session.ReturnTempArray(output.texture);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
            }
        }

        private static AexisModelPreflightReport AnalyzeLoadedLayer(
            AexisGraphModel.Layer layer,
            AexisModelPreflightRequest request,
            params float[] immutableWeights)
        {
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops);
            using var reader = new AexisFloatArrayWeightReader(immutableWeights ?? Array.Empty<float>());
            session.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
            session.UseNcnnStyleGroupNorm = true;
            session.LoadModel(SerializeSingleLayer(layer), reader);
            return session.AnalyzeLoadedModelPreflight(request);
        }

        private static AexisModelPreflightReport AnalyzeLoadedLayerWithDeclaredInputs(
            AexisGraphModel.Layer layer,
            AexisModelPreflightRequest request,
            params float[] immutableWeights)
        {
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops);
            using var reader = new AexisFloatArrayWeightReader(immutableWeights ?? Array.Empty<float>());
            session.TensorTextureFormat = RenderTextureFormat.ARGBFloat;
            session.UseNcnnStyleGroupNorm = true;
            session.LoadModel(SerializeSingleLayerWithDeclaredInputs(layer), reader);
            return session.AnalyzeLoadedModelPreflight(request);
        }

        private static string DescribeReport(AexisModelPreflightReport report)
        {
            var diagnostics = report?.texturePlan?.diagnostics == null
                ? string.Empty
                : string.Join(" | ", report.texturePlan.diagnostics.Select(diagnostic =>
                    diagnostic.code + ": " + diagnostic.reason + " action=" + diagnostic.recommendedAction));
            var nodeIssues = report?.nodes == null
                ? string.Empty
                : string.Join(" | ", report.nodes.SelectMany(node => node.issues ?? Array.Empty<string>()));
            return (report?.summary ?? "<no-report>") + " diagnostics=[" + diagnostics + "] nodeIssues=[" + nodeIssues + "]";
        }

        private static string DescribeLowering(AexisOnnxGraphLoweringResult result)
        {
            return string.Join(" | ", (result?.diagnostics ?? Array.Empty<AexisOnnxLoweringDiagnostic>()).Select(diagnostic =>
                diagnostic.code + ": " + diagnostic.message + " blocking=" + diagnostic.blocking));
        }

        private static void ExecuteBoundedRecurrentGolden(
            string typeName,
            AexisLayerTypeKey type,
            AexisRecurrentKind kind,
            int inputSize,
            int hiddenSize,
            float[] immutableWeights,
            float[] inputByChannel)
        {
            RequireArgbFloatCompute();
            Assert.That(inputByChannel.Length % inputSize, Is.EqualTo(0));
            var sequenceLength = inputByChannel.Length / inputSize;
            var layer = new AexisGraphModel.Layer
            {
                name = typeName.ToLowerInvariant() + "_pack4_golden",
                typeName = typeName,
                type = type,
                bottoms = 1,
                tops = 1,
                bottomNames = new[] { "data" },
                topNames = new[] { "output" },
                intParams = { [0] = inputSize.ToString(CultureInfo.InvariantCulture), [1] = hiddenSize.ToString(CultureInfo.InvariantCulture), [2] = "0", [3] = "0", [4] = "0" }
            };
            using var ops = new AexisOps();
            using var session = new AexisGraphSession(ops)
            {
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32"
            };
            using var reader = new AexisFloatArrayWeightReader(immutableWeights);
            session.LoadModel(SerializeSingleLayer(layer), reader);
            using var upload = new ComputeBuffer(inputByChannel.Length, sizeof(float));
            upload.SetData(inputByChannel);
            using var commandBuffer = new CommandBuffer { name = "Aexis" + typeName + "Pack4Golden" };
            ComputeTexture input = null;
            var outputReadback = CreatePack4Target(sequenceLength, 1, Mathf.CeilToInt(hiddenSize / 4f));
            try
            {
                input = session.RentTempArray(commandBuffer, sequenceLength, 1, Mathf.CeilToInt(inputSize / 4f), RenderTextureFormat.ARGBFloat);
                ops.FillPack4FromBufferCHW(commandBuffer, upload, sequenceLength, 1, inputSize, input);
                using (var result = session.ForwardPack4Outputs(
                    commandBuffer,
                    new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { ["data"] = input },
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["data"] = new AexisGraphSession.BufferShape(3, sequenceLength, 1, 1, inputSize)
                    },
                    new[] { "output" }))
                {
                    Assert.That(result.GetLogicalShape("output"), Is.EqualTo(new AexisGraphSession.BufferShape(3, sequenceLength, 1, 1, hiddenSize)));
                    for (var slice = 0; slice < Mathf.CeilToInt(hiddenSize / 4f); slice++)
                        commandBuffer.CopyTexture(result.GetTexture("output").nameID, slice, 0, outputReadback, slice, 0);
                }
                session.ReturnTempArray(commandBuffer, input);
                input = null;
                Graphics.ExecuteCommandBuffer(commandBuffer);

                Assert.That(session.LastTextureExecutionPlan.strictEligible, Is.True, session.LastTextureExecutionPlan.summary);
                Assert.That(session.LastTextureExecutionPlan.nodes.Last().executionPath,
                    Is.EqualTo("command-buffer-pack4:bounded-" + typeName.ToLowerInvariant() + "-fp32"));
                Assert.That(session.LastCommandBufferRtArena.unplannedTextureAllocations, Is.EqualTo(0));
                Assert.That(session.LastCommandBufferRtArena.allPlannedResourcesBound, Is.True);
                Assert.That(session.LastCommandBufferRtArena.allBoundResourcesReleased, Is.True);
                Assert.That(ReadPack4Slice(outputReadback, 0), Is.EqualTo(BoundedRecurrentReference(kind, sequenceLength, inputSize, hiddenSize, immutableWeights, inputByChannel)).Within(2e-5f));
            }
            finally
            {
                if (input != null)
                    session.ReturnTempArray(commandBuffer, input);
                UnityEngine.Object.DestroyImmediate(outputReadback);
            }
        }

        // Explicit test oracle only. Runtime recurrence is the Pack4 CommandBuffer kernel.
        private static float[] BoundedRecurrentReference(
            AexisRecurrentKind kind,
            int sequenceLength,
            int inputSize,
            int hiddenSize,
            float[] weights,
            float[] inputByChannel)
        {
            var gates = kind == AexisRecurrentKind.Gru ? 3 : kind == AexisRecurrentKind.Lstm ? 4 : 1;
            var inputWeightCount = gates * hiddenSize * inputSize;
            var stateWeightCount = gates * hiddenSize * hiddenSize;
            var state = new float[hiddenSize];
            var cell = new float[hiddenSize];
            var next = new float[hiddenSize];
            var nextCell = new float[hiddenSize];
            var output = new float[sequenceLength * 4];
            float DotInput(int gate, int unit, int token)
            {
                var sum = 0f;
                var offset = (gate * hiddenSize + unit) * inputSize;
                for (var index = 0; index < inputSize; index++)
                    sum += weights[offset + index] * inputByChannel[index * sequenceLength + token];
                return sum;
            }
            float DotState(int gate, int unit, float multiplier)
            {
                var sum = 0f;
                var offset = inputWeightCount + (gate * hiddenSize + unit) * hiddenSize;
                for (var index = 0; index < hiddenSize; index++)
                    sum += weights[offset + index] * state[index] * multiplier;
                return sum;
            }
            float Bias(int gate, int unit) => weights[inputWeightCount + stateWeightCount + gate * hiddenSize + unit];
            float Sigmoid(float value) => 1f / (1f + Mathf.Exp(-Mathf.Clamp(value, -60f, 60f)));
            float Tanh(float value) => (float)Math.Tanh(Mathf.Clamp(value, -16f, 16f));

            for (var token = 0; token < sequenceLength; token++)
            {
                for (var unit = 0; unit < hiddenSize; unit++)
                {
                    if (kind == AexisRecurrentKind.Rnn)
                        next[unit] = Tanh(DotInput(0, unit, token) + DotState(0, unit, 1f) + Bias(0, unit));
                    else if (kind == AexisRecurrentKind.Gru)
                    {
                        var update = Sigmoid(DotInput(0, unit, token) + DotState(0, unit, 1f) + Bias(0, unit));
                        var reset = Sigmoid(DotInput(1, unit, token) + DotState(1, unit, 1f) + Bias(1, unit));
                        var candidate = Tanh(DotInput(2, unit, token) + DotState(2, unit, reset) + Bias(2, unit));
                        next[unit] = (1f - update) * candidate + update * state[unit];
                    }
                    else
                    {
                        var inputGate = Sigmoid(DotInput(0, unit, token) + DotState(0, unit, 1f) + Bias(0, unit));
                        var outputGate = Sigmoid(DotInput(1, unit, token) + DotState(1, unit, 1f) + Bias(1, unit));
                        var forgetGate = Sigmoid(DotInput(2, unit, token) + DotState(2, unit, 1f) + Bias(2, unit));
                        var candidate = Tanh(DotInput(3, unit, token) + DotState(3, unit, 1f) + Bias(3, unit));
                        nextCell[unit] = forgetGate * cell[unit] + inputGate * candidate;
                        next[unit] = outputGate * Tanh(nextCell[unit]);
                    }
                }
                for (var unit = 0; unit < hiddenSize; unit++)
                {
                    state[unit] = next[unit];
                    if (kind == AexisRecurrentKind.Lstm)
                        cell[unit] = nextCell[unit];
                    output[token * 4 + unit] = state[unit];
                }
            }
            return output;
        }

        private static string SerializeSingleLayer(AexisGraphModel.Layer layer)
        {
            return SerializeLayers(layer);
        }

        private static string SerializeLayers(params AexisGraphModel.Layer[] layers)
        {
            if (layers == null || layers.Length == 0)
                throw new ArgumentException("At least one layer is required.", nameof(layers));

            var blobs = new HashSet<string>(StringComparer.Ordinal);
            var builder = new StringBuilder();
            builder.AppendLine("7767517");
            foreach (var layer in layers)
            {
                if (layer == null)
                    throw new ArgumentException("Layers cannot contain null.", nameof(layers));
                foreach (var name in layer.bottomNames ?? Array.Empty<string>()) blobs.Add(name);
                foreach (var name in layer.topNames ?? Array.Empty<string>()) blobs.Add(name);
            }
            builder.Append(layers.Length.ToString(CultureInfo.InvariantCulture)).Append(' ')
                .Append(blobs.Count.ToString(CultureInfo.InvariantCulture)).AppendLine();
            foreach (var layer in layers)
            {
                builder.Append(layer.typeName).Append(' ').Append(layer.name).Append(' ')
                    .Append(layer.bottoms.ToString(CultureInfo.InvariantCulture)).Append(' ')
                    .Append(layer.tops.ToString(CultureInfo.InvariantCulture));
                foreach (var name in layer.bottomNames ?? Array.Empty<string>()) builder.Append(' ').Append(name);
                foreach (var name in layer.topNames ?? Array.Empty<string>()) builder.Append(' ').Append(name);
                foreach (var pair in layer.intParams.OrderBy(pair => pair.Key)) builder.Append(' ').Append(pair.Key.ToString(CultureInfo.InvariantCulture)).Append('=').Append(pair.Value);
                foreach (var pair in layer.stringParams.OrderBy(pair => pair.Key, StringComparer.Ordinal)) builder.Append(' ').Append(pair.Key).Append('=').Append(pair.Value);
                builder.AppendLine();
            }
            return builder.ToString();
        }

        private static string SerializeSingleLayerWithDeclaredInputs(AexisGraphModel.Layer layer)
        {
            var bottoms = (layer.bottomNames ?? Array.Empty<string>())
                .Where(name => !string.IsNullOrWhiteSpace(name))
                .Distinct(StringComparer.Ordinal)
                .ToArray();
            var blobs = new HashSet<string>(bottoms, StringComparer.Ordinal);
            foreach (var top in layer.topNames ?? Array.Empty<string>())
                blobs.Add(top);

            var builder = new StringBuilder();
            builder.AppendLine("7767517");
            builder.Append((bottoms.Length + 1).ToString(CultureInfo.InvariantCulture))
                .Append(' ')
                .Append(blobs.Count.ToString(CultureInfo.InvariantCulture))
                .AppendLine();
            foreach (var bottom in bottoms)
                builder.Append("Input input_").Append(bottom).Append(" 0 1 ").Append(bottom).AppendLine();
            builder.Append(layer.typeName).Append(' ').Append(layer.name).Append(' ')
                .Append(layer.bottoms.ToString(CultureInfo.InvariantCulture)).Append(' ')
                .Append(layer.tops.ToString(CultureInfo.InvariantCulture));
            foreach (var name in layer.bottomNames ?? Array.Empty<string>()) builder.Append(' ').Append(name);
            foreach (var name in layer.topNames ?? Array.Empty<string>()) builder.Append(' ').Append(name);
            foreach (var pair in layer.intParams.OrderBy(pair => pair.Key)) builder.Append(' ').Append(pair.Key.ToString(CultureInfo.InvariantCulture)).Append('=').Append(pair.Value);
            foreach (var pair in layer.stringParams.OrderBy(pair => pair.Key, StringComparer.Ordinal)) builder.Append(' ').Append(pair.Key).Append('=').Append(pair.Value);
            builder.AppendLine();
            return builder.ToString();
        }

        private static OnnxValueInfo Value(string name, TensorDataType type, params long[] shape)
        {
            return new OnnxValueInfo { name = name, dataType = type, dims = shape };
        }

        private static OnnxNode Node(string name, string op, string[] inputs, string[] outputs)
        {
            var node = new OnnxNode { name = name, opType = op };
            node.inputs.AddRange(inputs);
            node.outputs.AddRange(outputs);
            return node;
        }

        private static OnnxTensor Tensor(string name, TensorDataType type, params long[] shape)
        {
            return new OnnxTensor { name = name, dataType = type, dims = shape };
        }

        private static OnnxTensor FloatTensor(string name, float[] values, params long[] shape)
        {
            return new OnnxTensor
            {
                name = name,
                dataType = TensorDataType.Float32,
                onnxDataType = 1,
                dims = shape,
                floatData = values
            };
        }

        private static OnnxTensor IntTensor(string name, int[] values, params long[] shape)
        {
            return new OnnxTensor
            {
                name = name,
                dataType = TensorDataType.Int32,
                onnxDataType = 6,
                dims = shape,
                int32Data = values
            };
        }

        private static OnnxTensor Int64Tensor(string name, long[] values, params long[] shape)
        {
            return new OnnxTensor
            {
                name = name,
                dataType = TensorDataType.Int32,
                onnxDataType = 7,
                dims = shape,
                int64Data = values
            };
        }

        private static OnnxAttribute Int(long value)
        {
            return new OnnxAttribute { type = 2, i = value };
        }

        private static OnnxAttribute FloatAttribute(float value)
        {
            return new OnnxAttribute { type = 1, f = value };
        }

        private static OnnxAttribute StringAttribute(string value)
        {
            return new OnnxAttribute { type = 3, s = Encoding.UTF8.GetBytes(value ?? string.Empty) };
        }

        private static OnnxAttribute TensorAttribute(OnnxTensor tensor)
        {
            return new OnnxAttribute { type = 4, tensor = tensor };
        }

        private static OnnxAttribute Ints(params long[] values)
        {
            var attribute = new OnnxAttribute { type = 7 };
            attribute.ints.AddRange(values);
            return attribute;
        }

        private static AexisOnnxTensorDescriptor Find(AexisOnnxGraphLoweringResult result, string name)
        {
            foreach (var tensor in result.tensors)
                if (string.Equals(tensor.name, name, StringComparison.Ordinal))
                    return tensor;
            Assert.Fail("Tensor was not inferred: " + name);
            return null;
        }

        private sealed class MissingCommandBufferLayer : AexisBaseLayer
        {
            public MissingCommandBufferLayer()
                : base(AexisLayerTypeKey.FromString("MissingCmd"), supportsBufferPath: false, supportsCommandBufferPath: false)
            {
            }
        }
    }
}
