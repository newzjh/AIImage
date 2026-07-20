using System;
using System.Collections.Generic;
using Aexis;

namespace Aexis.Onnx
{
    public enum CanonicalShapeIndexOp
    {
        Shape, Size, Rank, NonZero, Compress, GatherND,
        Scatter, ScatterElements, ScatterND, TopK, OneHot
    }

    [Serializable]
    public sealed class CanonicalShapeIndexNode
    {
        public string name = string.Empty;
        public CanonicalShapeIndexOp op;
        public string[] inputs = Array.Empty<string>();
        public string[] outputs = Array.Empty<string>();
        public OnnxExecutionNodeContract contract = new OnnxExecutionNodeContract();
    }

    // Minimal ONNX/Sentis-facing node representation. Parsing protobuf/Sentis model
    // assets remains outside core; adapters feed this lossless metadata to planning.
    [Serializable]
    public sealed class OnnxExecutionImportedNode
    {
        public string name = string.Empty;
        public string opType = string.Empty;
        public string[] inputs = Array.Empty<string>();
        public string[] outputs = Array.Empty<string>();
        public int inputRank;
        public int axis;
        public int batchDims;
        public TensorDataType indexDataType = TensorDataType.Int32;
        public bool parameterComesFromTensor;
        public bool uniqueIndices;
        public int outputCapacity;
        public string scatterConflictPolicy = "reject";
    }

    public static class OnnxExecutionAdapter
    {
        public static bool TryAdapt(OnnxExecutionImportedNode source, out CanonicalShapeIndexNode node, out OnnxExecutionPlanDiagnostic diagnostic)
        {
            node = null;
            diagnostic = null;
            if (source == null)
            {
                diagnostic = new OnnxExecutionPlanDiagnostic { code = "null-node", message = "ONNX execution import node is null.", recommendedAction = "Re-export the graph." };
                return false;
            }
            if (!Enum.TryParse(source.opType, false, out CanonicalShapeIndexOp op))
            {
                diagnostic = new OnnxExecutionPlanDiagnostic { code = "unsupported-op", message = "ONNX operator " + source.opType + " is outside the shape/index subset.", recommendedAction = "Lower it before importing." };
                return false;
            }

            var contract = new OnnxExecutionNodeContract
            {
                name = source.name ?? string.Empty,
                opType = source.opType,
                inputRank = source.inputRank,
                axis = source.axis,
                batchDims = source.batchDims,
                indexDataType = source.indexDataType,
                dynamicParameter = source.parameterComesFromTensor,
                uniqueIndices = source.uniqueIndices,
                scatterConflictPolicy = source.scatterConflictPolicy ?? "reject",
                outputCapacity = source.outputCapacity
            };
            if (source.parameterComesFromTensor || op == CanonicalShapeIndexOp.NonZero || op == CanonicalShapeIndexOp.Compress)
            {
                contract.outputShape = new GpuShapeTensorContract
                {
                    rank = Math.Max(1, source.inputRank), capacity = source.outputCapacity,
                    lengthPolicy = GpuShapeLengthPolicy.CapacityBounded, overflowPolicy = "reject",
                    lengthTensor = (source.outputs != null && source.outputs.Length > 0 ? source.outputs[0] : source.name) + ".shape"
                };
            }
            diagnostic = OnnxExecutionShapePlanner.Validate(contract);
            if (diagnostic != null)
                return false;
            node = new CanonicalShapeIndexNode { name = source.name ?? string.Empty, op = op, inputs = source.inputs ?? Array.Empty<string>(), outputs = source.outputs ?? Array.Empty<string>(), contract = contract };
            return true;
        }
    }
}
