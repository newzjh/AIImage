using System;
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

            Debug.Log("[Aexis.Editor] package batch smoke passed.");
        }
    }
}
