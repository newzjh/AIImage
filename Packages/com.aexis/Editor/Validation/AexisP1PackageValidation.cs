using System;
using System.Collections.Generic;
using Aexis;
using Aexis.Execution;
using Aexis.Ncnn;
using Aexis.Onnx;
using UnityEditor;
using UnityEngine;

namespace Aexis.Editor
{
    public static class AexisP1PackageValidation
    {
        [MenuItem("Aexis/Validation/Run P1 Package Smoke")]
        public static void RunBatchSmoke()
        {
            ValidateBinaryModelFormat();
            ValidateCustomLayerSchema();
            ValidateKernelDeclaration();
            ValidateGridSampleLowering();
            ValidateP1CapabilityCatalog();
            ValidateStrictP1Pack4Profile();
            ValidatePrecisionContracts();
            ValidateBfloat16CastLowering();
            ValidateDetectionPostprocessing();
            Debug.Log("[AexisP1PackageValidation] passed");
        }

        private static void ValidateBinaryModelFormat()
        {
            var graph = NcnnParamParser.Parse("7767517\n0 0\n");
            graph.extensionDeclarations = new[]
            {
                new AexisModelExtensionDeclaration { typeName = "com.aexis.validation", schemaVersion = 1, kernelId = "Validation", textureNativeRequired = true }
            };
            var encoded = AexisNcnnBinaryParam.Serialize(graph);
            var decoded = AexisNcnnBinaryParam.Deserialize(encoded);
            Require(decoded.extensionDeclarations.Length == 1 && decoded.extensionDeclarations[0].kernelId == "Validation", "Binary param extension declaration round trip failed.");

            var archive = AexisModelArchive.Deserialize(AexisModelArchive.Serialize(new AexisCompiledModel
            {
                modelId = "validation",
                sourceFormat = "ncnn-param",
                eligible = true,
                binaryParam = encoded,
                weights = new byte[] { 1, 2, 3 }
            }));
            Require(archive.ReadGraph().extensionDeclarations.Length == 1 && archive.weights.Length == 3, "Aexis archive round trip failed.");
        }

        private static void ValidateCustomLayerSchema()
        {
            const string typeName = "com.aexis.ValidationLayer";
            AexisCustomLayerRegistry.Register(new AexisCustomLayerDefinition
            {
                typeName = typeName,
                kernelId = "Validation",
                schema = new AexisCustomLayerSchema
                {
                    minimumInputs = 1,
                    maximumInputs = 1,
                    minimumOutputs = 1,
                    maximumOutputs = 1,
                    textureNativeRequired = false,
                    parameters = new[] { new AexisLayerParameterSchema { hasNcnnKey = true, ncnnKey = 0, required = true } }
                },
                createLayer = () => new ValidationLayer()
            });
            try
            {
                var layer = new AexisGraphModel.Layer
                {
                    typeName = typeName,
                    name = "validation",
                    bottomNames = new[] { "input" },
                    topNames = new[] { "output" },
                    intParams = new Dictionary<int, string> { { 0, "1" } }
                };
                Require(AexisLayerFactory.Create(layer) is ValidationLayer, "Custom layer was not created.");
                layer.intParams.Clear();
                var rejected = false;
                try { AexisLayerFactory.Create(layer); }
                catch (InvalidOperationException) { rejected = true; }
                Require(rejected, "Custom layer schema did not reject a missing required parameter.");
            }
            finally
            {
                AexisCustomLayerRegistry.Unregister(typeName);
            }
        }

        private static void ValidateGridSampleLowering()
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

            var lowered = AexisOnnxGraphLowering.Lower(model);
            var layer = lowered.graph.layers[lowered.graph.layers.Count - 1];
            Require(lowered.IsEligible && layer.typeName == "GridSample" && layer.GetInt(0) == 2 && layer.GetInt(1) == 3 && layer.GetInt(2) == 1,
                "GridSample lowering did not preserve the NCNN ABI.");
        }

        private static void ValidateKernelDeclaration()
        {
            const string kernelId = "Aexis.ValidationGridSample";
            AexisShaderKernelRegistry.Register(new ValidationKernel(kernelId));
            try
            {
                AexisCustomLayerRegistry.ValidateDeclarations(new[]
                {
                    new AexisModelExtensionDeclaration
                    {
                        typeName = "GridSample",
                        schemaVersion = 1,
                        kernelId = kernelId,
                        textureNativeRequired = true
                    }
                });
            }
            finally
            {
                AexisShaderKernelRegistry.Unregister(kernelId);
            }
        }

        private static void ValidatePrecisionContracts()
        {
            var manifest = new ModelManifest
            {
                modelId = "validation",
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
                    },
                    activationPlans = new[]
                    {
                        new QuantizedActivationPlan
                        {
                            layerName = "head",
                            operatorName = "Convolution",
                            packing = ActivationQuantizationPacking.Pack4UnsignedInt8,
                            calibration = new ActivationCalibrationRange
                            {
                                layerName = "head",
                                tensorName = "head_input",
                                minimum = -1f,
                                maximum = 2f,
                                sampleCount = 32,
                                method = CalibrationMethod.Percentile
                            }
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
            Require(Math.Abs(manifest.mixedPrecision.activationPlans[0].calibration.SymmetricScale - 2f / 127f) < 1e-6f,
                "INT8 activation calibration scale is not deterministic.");
            var measurement = AexisPrecisionGateEvaluator.Measure("output", new[] { 1f, 2f }, new[] { 1.01f, 1.99f });
            Require(manifest.precisionGate.Accepts(measurement, out _), "Precision gate rejected a compliant output.");
        }

        private static void ValidateBfloat16CastLowering()
        {
            var model = new OnnxModel { opset = 16 };
            model.graph.inputs.Add(new OnnxValueInfo
            {
                name = "input",
                dataType = TensorDataType.Float16,
                onnxDataType = 10,
                dims = new long[] { 1, 3, 4, 4 }
            });
            var node = new OnnxNode { name = "to_bfloat16", opType = "Cast" };
            node.inputs.Add("input");
            node.outputs.Add("output");
            node.attributes["to"] = new OnnxAttribute { i = 16 };
            model.graph.nodes.Add(node);
            model.graph.outputs.Add(new OnnxValueInfo
            {
                name = "output",
                dataType = TensorDataType.BFloat16,
                onnxDataType = 16,
                dims = new long[] { 1, 3, 4, 4 }
            });

            var lowered = AexisOnnxGraphLowering.Lower(model);
            var layer = lowered.graph.layers[lowered.graph.layers.Count - 1];
            Require(lowered.IsEligible && layer.typeName == "Cast" && layer.GetInt(0) == 2 && layer.GetInt(1) == 4,
                "ONNX BFLOAT16 Cast did not lower to the Pack4 BF16 ABI.");
        }

        private static void ValidateP1CapabilityCatalog()
        {
            foreach (var operatorName in new[]
            {
                "GridSample", "DeformableConv2D", "Fold", "Flip", "GLU", "Einsum", "Diag", "SPP",
                "ROIAlign", "ROIPooling", "PSROIPooling", "Proposal", "DetectionOutput", "YoloDetectionOutput"
            })
            {
                Require(AexisOperatorCapabilities.TryGet(operatorName, out var capability) && capability.importSupported,
                    "P1 operator is missing from the capability catalog: " + operatorName);
            }
        }

        private static void ValidateStrictP1Pack4Profile()
        {
            // Exercise the loaded-runtime verifier rather than only the static
            // capability catalog. Flip has no immutable weights, so this validates
            // the texture-only P1 admission path without creating a model fixture.
            using (var ops = new AexisOps())
            using (var session = new AexisGraphSession(ops) { TensorTextureFormat = RenderTextureFormat.ARGBFloat })
            using (var reader = new AexisFloatArrayWeightReader(Array.Empty<float>()))
            {
                session.LoadModel("7767517\n1 2\nFlip p1_flip 1 1 data output\n", reader);
                var descriptor = new AexisTexturePlanTensorDescriptor
                {
                    blob = "data",
                    logicalShape = new[] { 3, 4, 4, 1, 4 },
                    storageShape = new[] { 3, 4, 4, 1, 4 },
                    layout = AexisTexturePlanLayout.Packed4,
                    dtype = "FP32",
                    logicalDtype = "Float32",
                    textureBacked = true
                };
                var report = session.AnalyzeLoadedModelPreflight(new AexisModelPreflightRequest
                {
                    strict = true,
                    targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                    targetDtype = "BF16",
                    targetLayout = AexisTexturePlanLayout.Packed4,
                    inputs = new[]
                    {
                        new AexisPreflightTensorDescriptor
                        {
                            blob = "data",
                            logicalShape = descriptor.logicalShape,
                            storageShape = descriptor.storageShape,
                            layout = descriptor.layout,
                            dtype = descriptor.dtype,
                            logicalDtype = descriptor.logicalDtype
                        }
                    },
                    textureInputs = new[] { descriptor }
                });
                Require(report.strictEligible && report.texturePlan.dispatchAllowed,
                    "Loaded P1 Pack4/BF16 profile was not admitted by strict preflight: "
                    + (report.summary ?? string.Empty) + " | " + (report.texturePlan?.summary ?? string.Empty));
            }
        }

        private static void ValidateDetectionPostprocessing()
        {
            var output = AexisDetectionPostprocessing.DecodeYolo(
                new[] { 5f, 5f, 10f, 10f, 1f, 0.9f, 0.1f, 5f, 5f, 10f, 10f, 1f, 0.8f, 0.2f },
                2, 2, 0.3f, 0.5f, 10);
            Require(output.Length == 1 && Math.Abs(output[0].score - 0.9f) < 1e-5f, "YOLO class-aware NMS failed.");
        }

        private static OnnxValueInfo Value(string name, params long[] shape)
        {
            return new OnnxValueInfo { name = name, dataType = TensorDataType.Float32, onnxDataType = 1, dims = shape };
        }

        private static void Require(bool condition, string message)
        {
            if (!condition)
                throw new InvalidOperationException(message);
        }

        private sealed class ValidationLayer : AexisBaseLayer
        {
            public ValidationLayer() : base(AexisLayerTypeKey.FromString("ValidationLayer"), false, false) { }
        }

        private sealed class ValidationKernel : IAexisShaderKernelExtension
        {
            public ValidationKernel(string kernelId) { KernelId = kernelId; }
            public string KernelId { get; }
            public void ExecuteRenderTexture(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context) { }
            public void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context) { }
        }
    }
}
