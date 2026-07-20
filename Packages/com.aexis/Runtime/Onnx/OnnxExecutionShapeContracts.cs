using System;
using System.Collections.Generic;
using Aexis;

namespace Aexis.Onnx
{
    public enum GpuShapeLengthPolicy
    {
        Static = 0,
        GpuResident = 1,
        CapacityBounded = 2
    }

    // Logical length remains on the GPU. Consumers must use the shape texture rather than CPU readback.
    public sealed class GpuShapeTensorContract
    {
        public int rank;
        public int capacity;
        public GpuShapeLengthPolicy lengthPolicy;
        public TensorDataType dataType = TensorDataType.Int32;
        public string lengthTensor = string.Empty;
        public string overflowPolicy = string.Empty;

        public void Validate(string nodeName)
        {
            if (rank < 0 || rank > 4)
                throw new InferenceContractException("Shape tensor " + nodeName + " requires rank in [0,4].");
            if (capacity <= 0)
                throw new InferenceContractException("Shape tensor " + nodeName + " requires a positive texture capacity.");
            if (dataType != TensorDataType.Int32)
                throw new InferenceContractException("Shape tensor " + nodeName + " requires Int32 values.");
            if (lengthPolicy == GpuShapeLengthPolicy.GpuResident && string.IsNullOrWhiteSpace(lengthTensor))
                throw new InferenceContractException("Shape tensor " + nodeName + " requires a GPU-resident length tensor.");
            if (lengthPolicy == GpuShapeLengthPolicy.CapacityBounded
                && !string.Equals(overflowPolicy, "reject", StringComparison.Ordinal))
            {
                throw new InferenceContractException("Shape tensor " + nodeName + " only supports overflowPolicy=reject.");
            }
        }
    }

    public sealed class OnnxExecutionNodeContract
    {
        public string name = string.Empty;
        public string opType = string.Empty;
        public int inputRank;
        public int axis;
        public int batchDims;
        public TensorDataType indexDataType = TensorDataType.Int32;
        public bool dynamicParameter;
        public bool uniqueIndices;
        public GpuShapeTensorContract outputShape;
        // Importers set this only after proving the producer cannot emit duplicate
        // destinations. The texture backend deliberately has no unordered writes.
        public string scatterConflictPolicy = "reject";
        public int outputCapacity;
    }

    public sealed class OnnxExecutionPlanDiagnostic
    {
        public string code = string.Empty;
        public string message = string.Empty;
        public string recommendedAction = string.Empty;
    }

    // Import-time contract checks. This is deliberately backend-neutral and never reads tensor data.
    public static class OnnxExecutionShapePlanner
    {
        public static OnnxExecutionPlanDiagnostic Validate(OnnxExecutionNodeContract node)
        {
            if (node == null)
                return Reject("null-node", "ONNX/Sentis node is null.", "Re-export the graph with a named node.");
            if (string.IsNullOrWhiteSpace(node.opType))
                return Reject("missing-op", "ONNX/Sentis node " + NodeName(node) + " has no opType.", "Preserve the ONNX operator type during import.");
            if (node.inputRank < 1 || node.inputRank > 4)
                return Reject("unsupported-rank", node.opType + " node " + NodeName(node) + " supports input rank 1..4 only.", "Lower or partition rank>4 tensors before this node.");

            try
            {
                node.outputShape?.Validate(NodeName(node));
            }
            catch (InferenceContractException exception)
            {
                return Reject("invalid-shape-contract", exception.Message, "Declare an Int32 GPU shape tensor with a positive capacity.");
            }

            switch (node.opType)
            {
                case "Shape":
                case "Size":
                case "Rank":
                    return null;
                case "TopK":
                case "OneHot":
                    return ValidateDynamicCapacity(node);
                case "GatherND":
                    return node.inputRank <= 4 && node.batchDims == 0 && node.indexDataType == TensorDataType.Int32
                        ? null
                        : Reject("unsupported-gathernd-profile", "GatherND node " + NodeName(node) + " requires batchDims=0 and Int32 indices.", "Cast indices to Int32 and lower batch_dims before import.");
                case "Scatter":
                case "ScatterElements":
                case "ScatterND":
                    return node.uniqueIndices
                           && node.indexDataType == TensorDataType.Int32
                           && string.Equals(node.scatterConflictPolicy, "reject", StringComparison.Ordinal)
                        ? null
                        : Reject("scatter-conflict-undefined", node.opType + " node " + NodeName(node) + " only accepts provably unique Int32 indices; duplicate writes have no portable texture ordering.", "Deduplicate indices before Scatter or use an explicitly supported reduction operator.");
                case "NonZero":
                case "Compress":
                    return node.outputShape != null
                           && node.outputShape.lengthPolicy == GpuShapeLengthPolicy.CapacityBounded
                           && node.outputShape.capacity == node.outputCapacity
                        ? null
                        : Reject("missing-gpu-compaction-contract", node.opType + " node " + NodeName(node) + " has data-dependent length.", "Provide a capacity-bounded GPU shape tensor and overflowPolicy=reject; do not read back the count.");
                default:
                    return Reject("unsupported-op", "ONNX/Sentis operator " + node.opType + " is outside the D3 shape/index subset.", "Lower the node to a supported canonical operator.");
            }
        }

        private static OnnxExecutionPlanDiagnostic ValidateDynamicCapacity(OnnxExecutionNodeContract node)
        {
            if (!node.dynamicParameter)
                return null;
            if (node.outputShape != null
                && node.outputShape.lengthPolicy == GpuShapeLengthPolicy.CapacityBounded
                && node.outputShape.capacity > 0)
            {
                return null;
            }
            return Reject("missing-dynamic-capacity", node.opType + " node " + NodeName(node) + " has a GPU-driven parameter but no bounded output contract.", "Declare output capacity and overflowPolicy=reject, then keep the actual length in a GPU shape tensor.");
        }

        private static string NodeName(OnnxExecutionNodeContract node) => string.IsNullOrWhiteSpace(node.name) ? "<unnamed>" : node.name;

        private static OnnxExecutionPlanDiagnostic Reject(string code, string message, string action)
        {
            return new OnnxExecutionPlanDiagnostic { code = code, message = message, recommendedAction = action };
        }
    }
}
