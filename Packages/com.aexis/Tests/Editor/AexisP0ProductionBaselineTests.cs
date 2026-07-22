using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Aexis;
using Aexis.Execution;
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
        public void OnnxLowering_RejectsUnsupportedOpsetAndOperatorBeforeIntroduction()
        {
            var unsupported = new OnnxModel { opset = 20 };
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
                ops.SentisNonZeroLinearMat(nonZeroInput, vectorShape, vectorStorage, 4, nonZeroOutput, nonZeroCount);
                ops.SentisCompressLinearMat(nonZeroInput, vectorShape, vectorStorage, condition, vectorShape, vectorStorage, 4, compressOutput, compressCount);

                var indexShape = new AexisGraphSession.BufferShape(2, 1, 2, 1, 1);
                var indexStorage = new AexisGraphSession.BufferShape(2, 1, 2, 1, 1);
                var outputShape = new AexisGraphSession.BufferShape(1, 2, 1, 1, 1);
                var outputStorage = new AexisGraphSession.BufferShape(2, 2, 1, 1, 1);
                ops.SentisGatherNdLinearMat(data, vectorShape, vectorStorage, indices, indexShape, indexStorage, outputShape, outputStorage, gatherOutput);
                ops.SentisScatterLinearMat(data, vectorShape, vectorStorage, indices, indexShape, indexStorage, updates, outputShape, outputStorage, 2, scatterOutput);

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
        public void SentisStaticTextureOperators_PassOnlyExactLoadedProfiles()
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
        public void SentisGatherTopKAndOneHot_LinearMatGpuMatchesReference()
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
                ops.SentisGatherLinearMat(data, dataShape, dataStorage, gatherIndices, gatherIndexShape, new AexisGraphSession.BufferShape(2, 2, 1, 1, 1), 1, gatherOutShape, gatherOutStorage, gatherOutput);
                Assert.That(ReadRFloat(gatherOutput, 4), Is.EqualTo(new[] { 3f, 1f, 6f, 4f }).Within(1e-6f));

                var elementIndexShape = new AexisGraphSession.BufferShape(2, 3, 2, 1, 1);
                ops.SentisGatherElementsLinearMat(data, dataShape, dataStorage, gatherElementsIndices, elementIndexShape, dataStorage, 1, elementIndexShape, dataStorage, gatherElementsOutput);
                Assert.That(ReadRFloat(gatherElementsOutput, 6), Is.EqualTo(new[] { 3f, 1f, 2f, 6f, 4f, 5f }).Within(1e-6f));

                var topKInputShape = new AexisGraphSession.BufferShape(1, 4, 1, 1, 1);
                var topKOutputShape = new AexisGraphSession.BufferShape(1, 2, 1, 1, 1);
                var topKOutputStorage = new AexisGraphSession.BufferShape(2, 2, 1, 1, 1);
                ops.SentisTopKLinearMat(topKInput, topKInputShape, new AexisGraphSession.BufferShape(2, 4, 1, 1, 1), 0, 2, true, topKOutputShape, topKOutputStorage, topKValues, topKIndices);
                Assert.That(ReadRFloat(topKValues, 2), Is.EqualTo(new[] { 3f, 3f }).Within(1e-6f));
                Assert.That(ReadRFloat(topKIndices, 2), Is.EqualTo(new[] { 1f, 2f }).Within(1e-6f));

                var oneHotInputShape = new AexisGraphSession.BufferShape(1, 3, 1, 1, 1);
                var oneHotOutputShape = new AexisGraphSession.BufferShape(2, 3, 3, 1, 1);
                ops.SentisOneHotLinearMat(oneHotIndices, oneHotInputShape, new AexisGraphSession.BufferShape(2, 3, 1, 1, 1), 1, 3, 0f, 1f, oneHotOutputShape, new AexisGraphSession.BufferShape(2, 3, 3, 1, 1), oneHotOutput);
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

        private static string SerializeSingleLayer(AexisGraphModel.Layer layer)
        {
            var blobs = new HashSet<string>((layer.bottomNames ?? Array.Empty<string>()).Concat(layer.topNames ?? Array.Empty<string>()), StringComparer.Ordinal);
            var builder = new StringBuilder();
            builder.AppendLine("7767517");
            builder.Append("1 ").Append(blobs.Count.ToString(CultureInfo.InvariantCulture)).AppendLine();
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
