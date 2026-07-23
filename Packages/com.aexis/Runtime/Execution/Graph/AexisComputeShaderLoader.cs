using System;
using UnityEngine;

namespace Aexis.Execution
{
    internal static class AexisComputeShaderLoader
    {
        private const string ResourceName = "Aexis/AexisCommon";

        public static ComputeShader LoadOrThrow()
        {
            var shader = Resources.Load<ComputeShader>(ResourceName);
            if (shader == null)
                throw new InvalidOperationException("Aexis common compute shader is missing from Runtime/Resources/Aexis.");
            return shader;
        }
    }
}
