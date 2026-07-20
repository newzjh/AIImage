using Aexis;
using NUnit.Framework;

namespace Aexis.Tests.Editor
{
    public sealed class InferencePackageBoundaryTests
    {
        [Test]
        public void CoreContractAssembly_DoesNotReferenceUnityEngine()
        {
            var references = typeof(TensorDescriptor).Assembly.GetReferencedAssemblies();
            foreach (var reference in references)
                Assert.That(reference.Name, Does.Not.StartWith("UnityEngine"));
        }
    }
}
