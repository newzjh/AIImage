using System;
using AIImage.Inference.Core;
using UnityEngine;

namespace AIImage.Inference.Validation
{
    public static class InferencePackageBoundary
    {
        public static void AssertCoreIsUnityIndependent(Type contractType)
        {
            if (contractType == null)
                throw new ArgumentNullException(nameof(contractType));
            if (contractType.Assembly.GetName().Name != "AIImage.Inference.Core")
                throw new InferenceContractException("Expected a core contract type.");
        }

        public static void RunSmoke()
        {
            AssertCoreIsUnityIndependent(typeof(TensorDescriptor));
            foreach (var reference in typeof(TensorDescriptor).Assembly.GetReferencedAssemblies())
            {
                if (reference.Name.StartsWith("UnityEngine", StringComparison.Ordinal))
                    throw new InferenceContractException("Core assembly must not reference UnityEngine.");
            }
            Debug.Log("[AIImage.Inference.Validation] core boundary smoke passed.");
        }
    }
}
