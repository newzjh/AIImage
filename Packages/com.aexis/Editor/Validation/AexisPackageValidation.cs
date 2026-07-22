using System;
using System.Linq;
using Aexis.Execution;
using Aexis.Ncnn;
using Aexis.Onnx;
using UnityEngine;

namespace Aexis.Editor
{
    public static class AexisPackageValidation
    {
        public static void RunBatchSmoke()
        {
            InferencePackageBoundary.RunSmoke();

            var diagnostic = OnnxExecutionShapePlanner.Validate(new OnnxExecutionNodeContract
            {
                name = "aexis_dynamic_topk",
                opType = "TopK",
                inputRank = 1,
                dynamicParameter = true,
                outputShape = new GpuShapeTensorContract
                {
                    rank = 1,
                    capacity = 8,
                    lengthPolicy = GpuShapeLengthPolicy.CapacityBounded,
                    overflowPolicy = "reject"
                }
            });

            if (diagnostic != null)
                throw new InferenceContractException("Aexis ONNX execution smoke failed: " + diagnostic.message);

            ValidateOnnxCompilerSmoke();
            ValidateNcnnAliasCompatibilitySmoke();

            Debug.Log("[Aexis.Editor] package batch smoke passed.");
        }

        private static void ValidateOnnxCompilerSmoke()
        {
            var model = new OnnxModel { opset = 13 };
            model.graph.inputs.Add(new OnnxValueInfo
            {
                name = "input",
                dataType = TensorDataType.Float32,
                onnxDataType = 1,
                dims = new long[] { 1, 4 }
            });
            model.graph.initializers["offset"] = new OnnxTensor
            {
                name = "offset",
                dataType = TensorDataType.Float32,
                onnxDataType = 1,
                dims = new long[] { 1 },
                floatData = new[] { 0.25f }
            };
            model.graph.nodes.Add(new OnnxNode
            {
                name = "add_offset",
                opType = "Add",
                inputs = { "input", "offset" },
                outputs = { "shifted" }
            });
            model.graph.nodes.Add(new OnnxNode
            {
                name = "relu",
                opType = "Relu",
                inputs = { "shifted" },
                outputs = { "output" }
            });
            model.graph.outputs.Add(new OnnxValueInfo
            {
                name = "output",
                dataType = TensorDataType.Float32,
                onnxDataType = 1,
                dims = new long[] { 1, 4 }
            });

            var lowering = AexisOnnxGraphLowering.Lower(model, new AexisOnnxGraphLoweringOptions
            {
                requireDeclaredGraphOutputs = true
            });
            if (!lowering.IsEligible)
                throw new InferenceContractException("Aexis ONNX lowering smoke failed: " + DescribeLoweringFailure(lowering));

            var compiled = AexisOnnxGraphCompiler.Compile(lowering);
            if (compiled.immutableWeights.Length != 1
                || compiled.immutableWeights[0] != 0.25f
                || compiled.paramText.IndexOf("MemoryData onnx_const_offset", StringComparison.Ordinal) < 0
                || compiled.paramText.IndexOf("BinaryOp add_offset", StringComparison.Ordinal) < 0
                || compiled.paramText.IndexOf("ReLU relu", StringComparison.Ordinal) < 0)
            {
                throw new InferenceContractException("Aexis ONNX compiler smoke did not preserve the expected texture-native constant and canonical graph layers.");
            }
        }

        private static void ValidateNcnnAliasCompatibilitySmoke()
        {
            var model = NcnnParamParser.Parse(
                "7767517\n"
                + "12 13\n"
                + "Input input 0 1 data\n"
                + "Bias bias 1 1 data bias_out 0=4\n"
                + "BNLL bnll 1 1 bias_out bnll_out\n"
                + "Exp exp 1 1 bnll_out exp_out\n"
                + "Log log 1 1 exp_out log_out\n"
                + "Power power 1 1 log_out power_out 0=1 1=1 2=0\n"
                + "Threshold threshold 1 1 power_out threshold_out 0=0\n"
                + "MVN mvn 1 1 threshold_out mvn_out 0=1 1=1 2=0\n"
                + "Pooling1D pool 1 1 mvn_out pool_out 0=0 1=2 2=2\n"
                + "CumulativeSum cumulative 1 1 pool_out cumulative_out 0=0\n"
                + "ConvolutionDepthWise1D depthwise 1 1 cumulative_out depthwise_out 0=4 1=3 6=12\n"
                + "Deconvolution1D deconvolution 1 1 depthwise_out output 0=4 1=3 6=12\n");

            var report = AexisModelPreflight.Analyze(model, new AexisModelPreflightRequest
            {
                modelName = "p0-ncnn-alias-smoke",
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = "FP32",
                strict = true,
                inputs = new[]
                {
                    new AexisPreflightTensorDescriptor
                    {
                        blob = "data",
                        logicalShape = new[] { 3, 16, 1, 1, 4 },
                        storageShape = new[] { 3, 16, 1, 1, 4 },
                        layout = AexisTexturePlanLayout.Packed4,
                        dtype = "FP32",
                        logicalDtype = "Float32"
                    }
                }
            });
            if (report.missingNodes.Length != 0
                || report.nodes.Any(node => string.Equals(node.status, AexisOperatorCapabilityStatus.Unsupported, StringComparison.Ordinal))
                || report.nodes.Single(node => node.operatorName == "CumulativeSum").canonicalOperator != "CumSum"
                || report.nodes.Single(node => node.operatorName == "ConvolutionDepthWise1D").canonicalOperator != "DepthwiseConv1D"
                || report.nodes.Single(node => node.operatorName == "Deconvolution1D").canonicalOperator != "ConvTranspose1D")
            {
                throw new InferenceContractException("Aexis NCNN alias compatibility smoke failed: " + report.summary);
            }

            var json = AexisModelPreflight.ToStableJson(report);
            if (!string.Equals(json, AexisModelPreflight.ToStableJson(report), StringComparison.Ordinal))
                throw new InferenceContractException("Aexis model preflight JSON is not deterministic.");
        }

        private static string DescribeLoweringFailure(AexisOnnxGraphLoweringResult lowering)
        {
            if (lowering?.diagnostics == null)
                return "no diagnostics";
            return string.Join(" | ", lowering.diagnostics
                .Where(diagnostic => diagnostic != null && diagnostic.blocking)
                .Select(diagnostic => diagnostic.code + ":" + diagnostic.message));
        }
    }
}
