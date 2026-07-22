using System;
using System.Collections.Generic;
using Aexis.Execution;
using Aexis.Ncnn;
using Aexis.Onnx;
using NUnit.Framework;

namespace Aexis.Tests.Editor
{
    public sealed class AexisP1InfrastructureTests
    {
        [Test]
        public void BinaryParam_RoundTripsLayersParametersAndExtensionDeclarations()
        {
            var graph = new AexisGraphModel
            {
                magic = "7767517",
                layerCount = 1,
                blobCount = 2,
                extensionDeclarations = new[]
                {
                    new AexisModelExtensionDeclaration
                    {
                        typeName = "com.example.Pack4Kernel",
                        schemaVersion = 2,
                        kernelId = "ExamplePack4",
                        textureNativeRequired = true
                    }
                }
            };
            graph.layers.Add(new AexisGraphModel.Layer
            {
                typeName = "GridSample",
                type = AexisLayerTypes.GridSample,
                name = "sample",
                bottoms = 2,
                tops = 1,
                bottomNames = new[] { "input", "grid" },
                topNames = new[] { "output" },
                intParams = new Dictionary<int, string> { { 0, "1" }, { 1, "2" }, { 2, "0" } },
                stringParams = new Dictionary<string, string>(StringComparer.Ordinal) { { "source", "test" } }
            });

            var roundTrip = NcnnParamParser.Parse(NcnnParamParser.WriteBinary(graph));

            Assert.That(roundTrip.magic, Is.EqualTo(graph.magic));
            Assert.That(roundTrip.layers, Has.Count.EqualTo(1));
            Assert.That(roundTrip.layers[0].typeName, Is.EqualTo("GridSample"));
            Assert.That(roundTrip.layers[0].GetInt(1), Is.EqualTo(2));
            Assert.That(roundTrip.layers[0].GetString("source"), Is.EqualTo("test"));
            Assert.That(roundTrip.extensionDeclarations[0].kernelId, Is.EqualTo("ExamplePack4"));
        }

        [Test]
        public void VersionedArchive_RoundTripsCompiledPayload()
        {
            var graph = NcnnParamParser.Parse("7767517\n0 0\n");
            var source = new AexisCompiledModel
            {
                modelId = "detector",
                sourceFormat = "ncnn-param",
                eligible = true,
                binaryParam = AexisNcnnBinaryParam.Serialize(graph),
                weights = new byte[] { 1, 2, 3 },
                source = new byte[] { 4, 5 },
                diagnosticJson = "{}"
            };

            var roundTrip = AexisModelArchive.Deserialize(AexisModelArchive.Serialize(source));

            Assert.That(roundTrip.modelId, Is.EqualTo("detector"));
            Assert.That(roundTrip.weights, Is.EqualTo(source.weights));
            Assert.That(roundTrip.ReadGraph().layers, Is.Empty);
        }

        [Test]
        public void CustomLayerRegistry_ValidatesSchemaBeforeFactoryCreation()
        {
            const string typeName = "com.example.TestLayer";
            AexisCustomLayerRegistry.Register(new AexisCustomLayerDefinition
            {
                typeName = typeName,
                kernelId = "ExampleKernel",
                schema = new AexisCustomLayerSchema
                {
                    minimumInputs = 1,
                    maximumInputs = 1,
                    minimumOutputs = 1,
                    maximumOutputs = 1,
                    textureNativeRequired = false,
                    parameters = new[] { new AexisLayerParameterSchema { hasNcnnKey = true, ncnnKey = 9, required = true } }
                },
                createLayer = () => new TestLayer()
            });
            try
            {
                var valid = new AexisGraphModel.Layer
                {
                    typeName = typeName,
                    name = "custom",
                    bottomNames = new[] { "input" },
                    topNames = new[] { "output" },
                    intParams = new Dictionary<int, string> { { 9, "7" } }
                };

                Assert.That(AexisLayerFactory.Create(valid), Is.TypeOf<TestLayer>());
                valid.intParams.Clear();
                Assert.Throws<InvalidOperationException>(() => AexisLayerFactory.Create(valid));
            }
            finally
            {
                AexisCustomLayerRegistry.Unregister(typeName);
            }
        }

        [Test]
        public void OnnxGridSample_LowersTheExactNcnnParameterAbi()
        {
            var model = new OnnxModel { opset = 16 };
            model.graph.inputs.Add(Value("input", 1, 3, 8, 8));
            model.graph.inputs.Add(Value("grid", 1, 5, 6, 2));
            var node = new OnnxNode { name = "sample", opType = "GridSample" };
            node.inputs.Add("input");
            node.inputs.Add("grid");
            node.outputs.Add("output");
            node.attributes["mode"] = new OnnxAttribute { s = new byte[] { 110, 101, 97, 114, 101, 115, 116 } };
            node.attributes["padding_mode"] = new OnnxAttribute { s = new byte[] { 114, 101, 102, 108, 101, 99, 116, 105, 111, 110 } };
            node.attributes["align_corners"] = new OnnxAttribute { i = 1 };
            model.graph.nodes.Add(node);
            model.graph.outputs.Add(Value("output", 1, 3, 5, 6));

            var result = AexisOnnxGraphLowering.Lower(model);
            var layer = result.graph.layers[result.graph.layers.Count - 1];

            Assert.That(result.IsEligible, Is.True);
            Assert.That(layer.typeName, Is.EqualTo("GridSample"));
            Assert.That(layer.GetInt(0), Is.EqualTo(2));
            Assert.That(layer.GetInt(1), Is.EqualTo(3));
            Assert.That(layer.GetInt(2), Is.EqualTo(1));
            Assert.That(result.tensors, Has.Some.Matches<AexisOnnxTensorDescriptor>(tensor => tensor.name == "output" && tensor.shape[2] == 5 && tensor.shape[3] == 6));
        }

        [Test]
        public void ShaderKernelDeclarations_RequireARegisteredTextureNativeKernel()
        {
            const string kernelId = "Aexis.Tests.GridSample";
            AexisShaderKernelRegistry.Register(new TestKernel(kernelId));
            try
            {
                Assert.DoesNotThrow(() => AexisCustomLayerRegistry.ValidateDeclarations(new[]
                {
                    new AexisModelExtensionDeclaration
                    {
                        typeName = "GridSample",
                        schemaVersion = 1,
                        kernelId = kernelId,
                        textureNativeRequired = true
                    }
                }));
            }
            finally
            {
                AexisShaderKernelRegistry.Unregister(kernelId);
            }
        }

        [Test]
        public void Bfloat16MixedPrecisionAndGate_AreValidated()
        {
            var manifest = new ModelManifest
            {
                modelId = "detector",
                precision = new ModelPrecisionContract
                {
                    activationDataType = TensorDataType.BFloat16,
                    weightDataType = TensorDataType.BFloat16,
                    sensitiveOutputDataType = TensorDataType.Float32
                },
                mixedPrecision = new ModelMixedPrecisionContract
                {
                    planVersion = "p1",
                    nodePlans = new[]
                    {
                        new MixedPrecisionNodePlan
                        {
                            layerName = "head",
                            operatorName = "Convolution",
                            activationDataType = TensorDataType.BFloat16,
                            weightDataType = TensorDataType.BFloat16,
                            accumulationDataType = TensorDataType.Float32
                        }
                    }
                },
                precisionGate = new ModelPrecisionGateContract
                {
                    gateVersion = "p1",
                    maximumAbsoluteError = 0.1f,
                    maximumMeanAbsoluteError = 0.01f,
                    minimumCosineSimilarity = 0.99f
                }
            };

            manifest.Validate();
            Assert.That(manifest.precisionGate.Accepts(new PrecisionGateMeasurement
            {
                outputName = "boxes",
                maximumAbsoluteError = 0.05f,
                meanAbsoluteError = 0.005f,
                cosineSimilarity = 0.999f
            }, out _), Is.True);
        }

        [Test]
        public void DetectionOutputAndYoloDecoders_ApplyClassAwareNms()
        {
            var detectionOutput = AexisDetectionPostprocessing.DecodeDetectionOutput(
                new[] { 0f, 0f, 10f, 10f, 1f, 1f, 9f, 9f },
                new[] { 0f, 0.9f, 0f, 0.8f },
                proposalCount: 2,
                classCount: 2,
                scoreThreshold: 0.1f,
                nmsThreshold: 0.5f,
                keepTopK: 10);
            var yolo = AexisDetectionPostprocessing.DecodeYolo(
                new[] { 5f, 5f, 10f, 10f, 1f, 0.9f, 0.1f, 5f, 5f, 10f, 10f, 1f, 0.8f, 0.2f },
                candidateCount: 2,
                classCount: 2,
                scoreThreshold: 0.3f,
                nmsThreshold: 0.5f,
                keepTopK: 10);

            Assert.That(detectionOutput, Has.Length.EqualTo(1));
            Assert.That(detectionOutput[0].classIndex, Is.EqualTo(1));
            Assert.That(yolo, Has.Length.EqualTo(1));
            Assert.That(yolo[0].score, Is.EqualTo(0.9f).Within(1e-5f));
        }

        private static OnnxValueInfo Value(string name, params long[] shape)
        {
            return new OnnxValueInfo
            {
                name = name,
                dataType = TensorDataType.Float32,
                onnxDataType = 1,
                dims = shape
            };
        }

        private sealed class TestLayer : AexisBaseLayer
        {
            public TestLayer() : base(AexisLayerTypeKey.FromString("TestLayer"), false, false) { }
        }

        private sealed class TestKernel : IAexisShaderKernelExtension
        {
            public TestKernel(string kernelId) { KernelId = kernelId; }
            public string KernelId { get; }
            public void ExecuteRenderTexture(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context) { }
            public void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context) { }
        }
    }
}
