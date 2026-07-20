using System;
using Aexis;
using UnityEngine;

namespace Aexis.Editor
{
    public static class InferencePackageBoundary
    {
        public static void AssertCoreIsUnityIndependent(Type contractType)
        {
            if (contractType == null)
                throw new ArgumentNullException(nameof(contractType));
            if (contractType.Assembly.GetName().Name != "Aexis")
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
            Debug.Log("[Aexis.Editor] core boundary smoke passed.");
        }
    }
}
