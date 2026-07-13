using AIImage.Inference.Core;
using NUnit.Framework;

namespace AIImage.Inference.Validation.Tests
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
